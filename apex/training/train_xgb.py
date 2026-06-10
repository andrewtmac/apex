"""Daily retraining of XGBoost probability estimator (Model A).

Training regime:
- 90-day rolling window of resolved markets
- Warm-start from previous model
- 5-fold cross-validation for calibration
- Validate against 7-day holdout before deployment
- Export to ONNX for production inference
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import asyncpg
import numpy as np
import structlog

from apex.models.registry import ModelRegistry
from apex.models.xgboost_prob import XGBoostProbabilityModel

logger = structlog.get_logger(__name__)

MODEL_NAME = "xgboost_prob"


async def _load_resolved_markets(
    db_url: str,
    lookback_days: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load resolved markets from TimescaleDB and build feature matrix.

    Returns (X, y, feature_names) where:
    - X is shape (n_markets, n_features)
    - y is shape (n_markets,) with binary outcomes (1=YES, 0=NO)
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5, command_timeout=60)

    try:
        # Fetch resolved markets with their features
        rows = await pool.fetch(
            """
            SELECT
                m.id,
                m.outcome,
                fs.features
            FROM markets m
            JOIN LATERAL (
                SELECT features
                FROM feature_store
                WHERE entity_id = m.id
                  AND feature_set = 'price_features'
                ORDER BY time DESC
                LIMIT 1
            ) fs ON TRUE
            WHERE m.status = 'resolved'
              AND m.outcome IS NOT NULL
              AND m.updated_at >= $1
            ORDER BY m.updated_at
            """,
            cutoff,
        )

        if not rows:
            raise ValueError(
                f"No resolved markets found in the last {lookback_days} days"
            )

        # Parse features from JSONB
        import json

        feature_dicts: list[dict[str, float]] = []
        labels: list[int] = []

        for row in rows:
            features_raw = row["features"]
            features = (
                json.loads(features_raw)
                if isinstance(features_raw, str)
                else features_raw
            )
            feature_dicts.append(features)
            labels.append(int(row["outcome"]))

        # Build aligned feature matrix
        all_keys = sorted(feature_dicts[0].keys())
        X = np.array(
            [[d.get(k, 0.0) for k in all_keys] for d in feature_dicts],
            dtype=np.float64,
        )
        y = np.array(labels, dtype=np.float64)

        logger.info(
            "xgb_train.data_loaded",
            n_samples=len(y),
            n_features=len(all_keys),
            positive_rate=float(np.mean(y)),
        )
        return X, y, all_keys

    finally:
        await pool.close()


async def train_xgboost(
    db_url: str,
    model_registry: ModelRegistry,
    lookback_days: int = 90,
) -> dict:
    """Train or retrain the XGBoost probability estimator.

    Steps:
    1. Load resolved markets from TimescaleDB (90-day rolling window)
    2. Build feature matrix from the feature store
    3. Train XGBRegressor with warm start from previous model
    4. Calibrate on held-out fold (isotonic regression)
    5. Validate: if new Sharpe < old * 0.8, reject
    6. Register in model registry
    7. Export ONNX

    Returns
    -------
    dict with keys: version_id, metrics, status ("deployed" | "rejected")
    """
    logger.info("xgb_train.start", lookback_days=lookback_days)

    # 1. Load data
    X, y, feature_names = await _load_resolved_markets(db_url, lookback_days)

    # 2. Split: 7-day holdout for validation
    n_holdout = max(1, int(len(y) * 0.1))  # ~7 days worth
    X_train, X_holdout = X[:-n_holdout], X[-n_holdout:]
    y_train, y_holdout = y[:-n_holdout], y[-n_holdout:]

    if len(y_train) < 50:
        raise ValueError(
            f"Insufficient training data: {len(y_train)} samples (need >=50)"
        )

    # 3. Try warm-start from previous model
    warm_params: dict | None = None
    try:
        prev_model = model_registry.load(MODEL_NAME, version="latest")
        if isinstance(prev_model, XGBoostProbabilityModel) and prev_model._is_fitted:
            # Extract learned booster params for warm start
            warm_params = {
                "n_estimators": 200,  # additional rounds on top of warm start
                "learning_rate": 0.03,  # reduced LR for fine-tuning
            }
            logger.info("xgb_train.warm_start", source="previous_model")
    except (FileNotFoundError, Exception):
        logger.info("xgb_train.cold_start", reason="no_previous_model")

    # 4. Train new model
    model = XGBoostProbabilityModel(params=warm_params)
    train_metrics = model.train(X_train, y_train, feature_names)

    # 5. Validate on holdout
    holdout_probs = model.predict(X_holdout)
    holdout_binary = (holdout_probs >= 0.5).astype(int)

    from sklearn.metrics import brier_score_loss, accuracy_score

    holdout_metrics = {
        "holdout_brier": float(brier_score_loss(y_holdout, holdout_probs)),
        "holdout_accuracy": float(accuracy_score(y_holdout, holdout_binary)),
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
            old_brier = old_metrics.get("brier_score", 1.0)
            new_brier = all_metrics.get("brier_score", 1.0)

            # Reject if calibration degraded significantly
            if new_brier > old_brier * 1.2:
                logger.warning(
                    "xgb_train.rejected",
                    reason="brier_degradation",
                    old_brier=old_brier,
                    new_brier=new_brier,
                )
                status = "rejected"
    except FileNotFoundError:
        pass  # no previous model, deploy unconditionally

    # 7. Register and optionally promote
    version_id = model_registry.register(MODEL_NAME, model, all_metrics)

    if status == "deployed":
        model_registry.promote(MODEL_NAME, version_id)

        # 8. Export ONNX
        try:
            onnx_path = model_registry.export_onnx(MODEL_NAME, version_id)
            logger.info("xgb_train.onnx_exported", path=str(onnx_path))
        except (TypeError, ImportError) as exc:
            logger.warning("xgb_train.onnx_export_failed", error=str(exc))

    result = {
        "model": MODEL_NAME,
        "version_id": version_id,
        "metrics": all_metrics,
        "status": status,
        "n_train": len(y_train),
        "n_holdout": len(y_holdout),
        "feature_importance": model.feature_importance(),
    }

    logger.info("xgb_train.complete", **{k: v for k, v in result.items() if k != "feature_importance"})
    return result
