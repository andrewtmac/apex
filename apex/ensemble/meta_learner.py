"""
APEX Stacked Generalization Meta-Learner

Level 1 meta-learner that combines Level 0 model outputs (XGBoost probability,
LightGBM predicted return, TFT quantiles, regime, sentiment) into a single
ensemble score.

Uses LightGBM trained on out-of-fold predictions so the meta-learner learns
which models are reliable in which regimes, avoiding information leakage from
in-sample stacking.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

# Default hyper-parameters tuned for the meta-learner objective
_META_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "mse",
    "learning_rate": 0.05,
    "num_leaves": 16,
    "max_depth": 4,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}


class MetaLearner:
    """Level 1 meta-learner that combines Level 0 model outputs.

    The meta-learner is trained on *out-of-fold* predictions from Level 0
    models (XGBoost, LightGBM, TFT, etc.) so it can learn regime-dependent
    reliability patterns without overfitting.

    Input feature vector layout (per sample)::

        [xgb_prob, xgb_edge, lgbm_return, tft_q10, tft_q50, tft_q90,
         regime_encoded, regime_confidence, sentiment_score,
         calibrated_edge, edge_ci_lower, edge_ci_upper]

    The target is realised return (or PnL) over the trade horizon.

    Parameters
    ----------
    n_folds : int
        Number of cross-validation folds for out-of-fold generation.
    params : dict
        LightGBM parameters override.  Merged on top of defaults.
    n_rounds : int
        Maximum boosting rounds.
    early_stopping : int
        Early-stopping patience.
    """

    # Canonical column names for the stacked feature vector
    FEATURE_NAMES: list[str] = [
        "xgb_prob",
        "xgb_edge",
        "lgbm_return",
        "tft_q10",
        "tft_q50",
        "tft_q90",
        "regime_encoded",
        "regime_confidence",
        "sentiment_score",
        "calibrated_edge",
        "edge_ci_lower",
        "edge_ci_upper",
    ]

    def __init__(
        self,
        n_folds: int = 5,
        params: dict[str, Any] | None = None,
        n_rounds: int = 500,
        early_stopping: int = 50,
    ) -> None:
        self.n_folds = n_folds
        self.params = {**_META_PARAMS, **(params or {})}
        self.n_rounds = n_rounds
        self.early_stopping = early_stopping

        self.model: lgb.Booster | None = None
        self._feature_importances: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        model_outputs: np.ndarray,
        actual_returns: np.ndarray,
    ) -> dict[str, float]:
        """Train the meta-learner on stacked Level 0 model outputs.

        Parameters
        ----------
        model_outputs : np.ndarray
            Shape ``(n_samples, n_features)`` -- stacked Level 0 predictions.
            Column order must match :attr:`FEATURE_NAMES`.
        actual_returns : np.ndarray
            Shape ``(n_samples,)`` -- realised returns / PnL.

        Returns
        -------
        dict[str, float]
            Training metrics: ``oof_mse``, ``oof_corr``, ``best_iteration``.
        """
        n_samples, n_features = model_outputs.shape
        assert n_features == len(self.FEATURE_NAMES), (
            f"Expected {len(self.FEATURE_NAMES)} features, got {n_features}"
        )

        oof_preds = np.zeros(n_samples, dtype=np.float64)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        best_iteration = 0

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(model_outputs)):
            X_train = model_outputs[train_idx]
            y_train = actual_returns[train_idx]
            X_val = model_outputs[val_idx]
            y_val = actual_returns[val_idx]

            dtrain = lgb.Dataset(
                X_train, label=y_train, feature_name=self.FEATURE_NAMES
            )
            dval = lgb.Dataset(
                X_val, label=y_val, feature_name=self.FEATURE_NAMES, reference=dtrain
            )

            booster = lgb.train(
                self.params,
                dtrain,
                num_boost_round=self.n_rounds,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(self.early_stopping, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

            oof_preds[val_idx] = booster.predict(X_val)
            best_iteration = max(best_iteration, booster.best_iteration)

            logger.debug(
                "Meta-learner fold %d/%d: best_iter=%d val_mse=%.6f",
                fold_idx + 1,
                self.n_folds,
                booster.best_iteration,
                float(np.mean((oof_preds[val_idx] - y_val) ** 2)),
            )

        # Final model on all data
        dfull = lgb.Dataset(
            model_outputs, label=actual_returns, feature_name=self.FEATURE_NAMES
        )
        self.model = lgb.train(
            self.params,
            dfull,
            num_boost_round=best_iteration or self.n_rounds,
        )

        # Feature importances
        importance = self.model.feature_importance(importance_type="gain")
        total = float(importance.sum()) or 1.0
        self._feature_importances = {
            name: float(imp) / total
            for name, imp in zip(self.FEATURE_NAMES, importance)
        }

        # Metrics
        oof_mse = float(np.mean((oof_preds - actual_returns) ** 2))
        oof_corr = float(np.corrcoef(oof_preds, actual_returns)[0, 1]) if n_samples > 2 else 0.0

        metrics = {
            "oof_mse": oof_mse,
            "oof_corr": oof_corr,
            "best_iteration": float(best_iteration),
            "n_samples": float(n_samples),
        }

        logger.info(
            "Meta-learner trained: oof_mse=%.6f oof_corr=%.4f n=%d",
            oof_mse,
            oof_corr,
            n_samples,
        )

        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, model_outputs: np.ndarray) -> float:
        """Predict ensemble score from stacked model outputs.

        Parameters
        ----------
        model_outputs : np.ndarray
            Shape ``(n_features,)`` -- a single observation's Level 0 outputs,
            ordered as :attr:`FEATURE_NAMES`.

        Returns
        -------
        float
            Ensemble score (predicted return / edge).

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if self.model is None:
            raise RuntimeError("MetaLearner has not been trained. Call train() first.")

        if model_outputs.ndim == 1:
            model_outputs = model_outputs.reshape(1, -1)

        pred = self.model.predict(model_outputs)
        return float(pred[0])

    def predict_batch(self, model_outputs: np.ndarray) -> np.ndarray:
        """Predict ensemble scores for a batch of observations.

        Parameters
        ----------
        model_outputs : np.ndarray
            Shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples,)`` ensemble scores.
        """
        if self.model is None:
            raise RuntimeError("MetaLearner has not been trained. Call train() first.")

        return self.model.predict(model_outputs)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    @property
    def feature_importances(self) -> dict[str, float]:
        """Normalised feature importances (gain-based) from the last training run."""
        return self._feature_importances.copy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the trained model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "params": self.params,
                    "n_folds": self.n_folds,
                    "n_rounds": self.n_rounds,
                    "early_stopping": self.early_stopping,
                    "feature_importances": self._feature_importances,
                },
                f,
            )
        logger.info("MetaLearner saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> MetaLearner:
        """Load a trained meta-learner from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)  # noqa: S301

        instance = cls(
            n_folds=state["n_folds"],
            params=state["params"],
            n_rounds=state["n_rounds"],
            early_stopping=state["early_stopping"],
        )
        instance.model = state["model"]
        instance._feature_importances = state.get("feature_importances", {})
        logger.info("MetaLearner loaded from %s", path)
        return instance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_feature_vector(
        xgb_prob: float,
        xgb_edge: float,
        lgbm_return: float,
        tft_quantiles: dict[float, float],
        regime_encoded: float,
        regime_confidence: float,
        sentiment_score: float,
        calibrated_edge: float,
        edge_ci_lower: float,
        edge_ci_upper: float,
    ) -> np.ndarray:
        """Assemble a meta-learner feature vector from individual model outputs.

        This is the canonical way to build the input array for :meth:`predict`.
        """
        return np.array(
            [
                xgb_prob,
                xgb_edge,
                lgbm_return,
                tft_quantiles.get(0.1, 0.0),
                tft_quantiles.get(0.5, 0.0),
                tft_quantiles.get(0.9, 0.0),
                regime_encoded,
                regime_confidence,
                sentiment_score,
                calibrated_edge,
                edge_ci_lower,
                edge_ci_upper,
            ],
            dtype=np.float64,
        )
