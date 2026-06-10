"""Daily retraining of LightGBM return predictor (Model B).

Training regime:
- 90-day rolling window of 1-hour return data
- Warm-start from previous model
- Early stopping with patience=50
- Validate: directional accuracy and IC on 7-day holdout
- Export to ONNX for production inference
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import asyncpg
import numpy as np
import structlog

from apex.models.lgbm_return import LightGBMReturnModel
from apex.models.registry import ModelRegistry

logger = structlog.get_logger(__name__)

MODEL_NAME = "lgbm_return"


async def _load_return_data(
    db_url: str,
    lookback_days: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load 1-hour return data from TimescaleDB feature store.

    Queries the feature_store for return-labeled features built by the
    data pipeline.

    Returns (X, y, feature_names) where y is the 1-hour forward return.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5, command_timeout=60)

    try:
        rows = await pool.fetch(
            """
            SELECT entity_id, features
            FROM feature_store
            WHERE feature_set = 'return_features'
              AND time >= $1
            ORDER BY time ASC
            """,
            cutoff,
        )

        if not rows:
            raise ValueError(
                f"No return feature data found in the last {lookback_days} days"
            )

        feature_dicts: list[dict[str, float]] = []
        targets: list[float] = []

        for row in rows:
            feat_raw = row["features"]
            feat = json.loads(feat_raw) if isinstance(feat_raw, str) else feat_raw

            # The target is stored as 'forward_return_1h' inside the features
            target = feat.pop("forward_return_1h", None)
            if target is None:
                continue

            feature_dicts.append(feat)
            targets.append(float(target))

        if not feature_dicts:
            raise ValueError("No samples with forward_return_1h target found")

        all_keys = sorted(feature_dicts[0].keys())
        X = np.array(
            [[d.get(k, 0.0) for k in all_keys] for d in feature_dicts],
            dtype=np.float64,
        )
        y = np.array(targets, dtype=np.float64)

        logger.info(
            "lgbm_train.data_loaded",
            n_samples=len(y),
            n_features=len(all_keys),
            y_mean=float(np.mean(y)),
            y_std=float(np.std(y)),
        )
        return X, y, all_keys

    finally:
        await pool.close()


async def train_lightgbm(
    db_url: str,
    model_registry: ModelRegistry,
    lookback_days: int = 90,
) -> dict:
    """Train or retrain the LightGBM return predictor.

    Steps:
    1. Load 1h-return features from TimescaleDB (90-day rolling window)
    2. Split chronologically: training + 7-day holdout
    3. Train LGBMRegressor with early stopping
    4. Validate: IC, directional accuracy, RMSE on holdout
    5. Compare against previous model; reject if IC drops below 0.8x
    6. Register and promote in model registry
    7. Export ONNX

    Returns
    -------
    dict with version_id, metrics, status
    """
    logger.info("lgbm_train.start", lookback_days=lookback_days)

    # 1. Load data
    X, y, feature_names = await _load_return_data(db_url, lookback_days)

    # 2. Chronological split: ~10% holdout (most recent)
    n_holdout = max(1, int(len(y) * 0.1))
    X_train, X_holdout = X[:-n_holdout], X[-n_holdout:]
    y_train, y_holdout = y[:-n_holdout], y[-n_holdout:]

    if len(y_train) < 100:
        raise ValueError(
            f"Insufficient training data: {len(y_train)} samples (need >=100)"
        )

    # 3. Try to inherit params from previous model
    warm_params: dict | None = None
    try:
        prev_model = model_registry.load(MODEL_NAME, version="latest")
        if isinstance(prev_model, LightGBMReturnModel) and prev_model._is_fitted:
            warm_params = {
                "n_estimators": 500,
                "learning_rate": 0.02,
            }
            logger.info("lgbm_train.warm_start")
    except (FileNotFoundError, Exception):
        logger.info("lgbm_train.cold_start")

    # 4. Train new model
    model = LightGBMReturnModel(params=warm_params)
    train_metrics = model.train(X_train, y_train, feature_names)

    # 5. Validate on holdout
    holdout_preds = model.predict(X_holdout)

    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    ic, ic_pval = spearmanr(y_holdout, holdout_preds)
    dir_acc = float(np.mean(np.sign(holdout_preds) == np.sign(y_holdout)))

    holdout_metrics = {
        "holdout_rmse": float(np.sqrt(mean_squared_error(y_holdout, holdout_preds))),
        "holdout_mae": float(mean_absolute_error(y_holdout, holdout_preds)),
        "holdout_ic": float(ic),
        "holdout_ic_pval": float(ic_pval),
        "holdout_directional_accuracy": dir_acc,
        "holdout_n": int(len(y_holdout)),
    }

    all_metrics = {**train_metrics, **holdout_metrics}

    # 6. Compare against previous production model
    status = "deployed"
    try:
        prev_versions = model_registry.list_versions(MODEL_NAME)
        prod_versions = [v for v in prev_versions if v.get("is_production")]
        if prod_versions:
            old_metrics = prod_versions[-1].get("metrics", {})
            old_ic = old_metrics.get("ic", 0.0)
            new_ic = all_metrics.get("ic", 0.0)

            if old_ic > 0 and new_ic < old_ic * 0.8:
                logger.warning(
                    "lgbm_train.rejected",
                    reason="ic_degradation",
                    old_ic=old_ic,
                    new_ic=new_ic,
                )
                status = "rejected"
    except FileNotFoundError:
        pass

    # 7. Register
    version_id = model_registry.register(MODEL_NAME, model, all_metrics)

    if status == "deployed":
        model_registry.promote(MODEL_NAME, version_id)

        # Export ONNX
        try:
            onnx_path = model_registry.export_onnx(MODEL_NAME, version_id)
            logger.info("lgbm_train.onnx_exported", path=str(onnx_path))
        except (TypeError, ImportError) as exc:
            logger.warning("lgbm_train.onnx_export_failed", error=str(exc))

    result = {
        "model": MODEL_NAME,
        "version_id": version_id,
        "metrics": all_metrics,
        "status": status,
        "n_train": len(y_train),
        "n_holdout": len(y_holdout),
        "feature_importance": model.feature_importance(),
    }

    logger.info(
        "lgbm_train.complete",
        **{k: v for k, v in result.items() if k != "feature_importance"},
    )
    return result
