"""
APEX Model B: LightGBM Return Predictor

Predicts the 1-hour forward return of a contract price.  Used alongside
the XGBoost probability model to add a momentum / mean-reversion signal
to the ensemble.

Output: continuous float -- expected 1h % return of the contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 1000,
    "max_depth": 8,
    "learning_rate": 0.03,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "num_leaves": 63,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


class LightGBMReturnModel:
    """
    LightGBM regressor for 1-hour forward contract return prediction.

    Workflow
    --------
    1. ``train(X, y, feature_names)``  -- fits LGBMRegressor with early stopping.
    2. ``predict(X)``                  -- returns predicted 1h returns.
    3. ``export_onnx(path)``           -- ONNX export for low-latency serving.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        import lightgbm as lgb

        self._lgb = lgb
        merged = {**_DEFAULT_LGBM_PARAMS, **(params or {})}
        self.model: lgb.LGBMRegressor = lgb.LGBMRegressor(**merged)
        self.feature_names: list[str] = []
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        eval_fraction: float = 0.1,
        early_stopping_rounds: int = 50,
    ) -> dict[str, float]:
        """
        Train on historical 1h-return data.

        Parameters
        ----------
        X : (n_samples, n_features) feature matrix.
        y : (n_samples,) continuous target -- 1h forward return.
        feature_names : column names aligned with X columns.
        eval_fraction : fraction of data held out for early stopping.
        early_stopping_rounds : patience for LGBM callback.

        Returns
        -------
        Dict of validation metrics: rmse, mae, r2, ic (information coefficient).
        """
        self.feature_names = list(feature_names)

        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y, test_size=eval_fraction, random_state=42,
        )

        callbacks = [
            self._lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            self._lgb.log_evaluation(period=0),  # suppress per-round logging
        ]

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_eval, y_eval)],
            callbacks=callbacks,
        )
        self._is_fitted = True

        # Validation metrics
        preds = self.model.predict(X_eval)
        rmse = float(np.sqrt(mean_squared_error(y_eval, preds)))
        mae = float(mean_absolute_error(y_eval, preds))
        r2 = float(r2_score(y_eval, preds))

        # Information coefficient (rank correlation)
        from scipy.stats import spearmanr

        ic, ic_pval = spearmanr(y_eval, preds)

        # Directional accuracy: did we get the sign right?
        dir_correct = np.mean(np.sign(preds) == np.sign(y_eval))

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "ic": float(ic),
            "ic_pval": float(ic_pval),
            "directional_accuracy": float(dir_correct),
            "n_train": int(len(X_train)),
            "n_eval": int(len(X_eval)),
            "best_iteration": int(self.model.best_iteration_) if hasattr(self.model, "best_iteration_") else -1,
        }

        logger.info("LightGBMReturnModel trained: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict 1-hour forward returns.

        Returns
        -------
        np.ndarray of shape (n_samples,) -- predicted % returns.
        """
        self._check_fitted()
        return self.model.predict(X)

    def predict_with_confidence(
        self,
        X: np.ndarray,
        n_trees_fraction: float = 0.8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict returns and estimate prediction uncertainty via tree sub-sampling.

        Returns
        -------
        (predictions, std_estimates) each of shape (n_samples,).
        """
        self._check_fitted()
        total_trees = self.model.n_estimators_
        n_sub = max(1, int(total_trees * n_trees_fraction))

        # Get predictions from multiple sub-samples of trees
        preds_collection: list[np.ndarray] = []
        step = max(1, total_trees // 5)
        for end in range(step, total_trees + 1, step):
            p = self.model.predict(X, num_iteration=end)
            preds_collection.append(p)

        stacked = np.stack(preds_collection, axis=0)
        mean_preds = stacked[-1]  # full model prediction
        std_preds = np.std(stacked, axis=0)

        return mean_preds, std_preds

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        """
        Return feature importance scores.

        Parameters
        ----------
        importance_type : ``"gain"`` (default), ``"split"``, or ``"shap"``.
        """
        self._check_fitted()
        raw = self.model.feature_importances_
        total = float(np.sum(raw)) or 1.0
        return {
            name: float(val / total)
            for name, val in zip(self.feature_names, raw)
        }

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def export_onnx(self, path: Path) -> None:
        """
        Export to ONNX for low-latency inference.

        Requires ``onnxmltools`` and ``skl2onnx``.
        """
        self._check_fitted()
        try:
            from onnxmltools import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType

            initial_type = [
                ("features", FloatTensorType([None, len(self.feature_names)])),
            ]
            onnx_model = convert_lightgbm(
                self.model,
                initial_types=initial_type,
                target_opset=15,
            )
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(onnx_model.SerializeToString())
            logger.info("Exported LightGBM ONNX to %s", path)

        except ImportError:
            # Fallback: native LightGBM text format
            fallback = path.with_suffix(".txt")
            self.model.booster_.save_model(str(fallback))
            logger.warning(
                "onnxmltools not installed; saved native LightGBM model to %s",
                fallback,
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "feature_names": self.feature_names,
            "is_fitted": self._is_fitted,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self.model = state["model"]
        self.feature_names = state["feature_names"]
        self._is_fitted = state["is_fitted"]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "LightGBMReturnModel has not been trained. Call train() first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        n_feat = len(self.feature_names)
        return f"<LightGBMReturnModel [{status}, {n_feat} features]>"
