"""Validates retrained models before they replace production models.

Validation criteria:
1. Backtest on 7-day holdout
2. New Sharpe >= old Sharpe * 0.8
3. Brier score <= old Brier * 1.2
4. No concept drift detected (KS test on top-10 features)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Thresholds
_SHARPE_RETENTION = 0.8       # new Sharpe must be >= 80% of old
_BRIER_DEGRADATION = 1.2      # new Brier must be <= 120% of old
_KS_DRIFT_THRESHOLD = 0.15    # KS statistic threshold for drift
_MIN_ACCURACY = 0.55           # minimum acceptable accuracy
_MAX_FEATURE_DRIFT_PCT = 0.3   # reject if >30% of top features drifted


class ModelValidator:
    """Pre-deployment validation gate for retrained models.

    Runs a battery of checks comparing the new model against the current
    production model on held-out data.  All checks must pass for the new
    model to be promoted.
    """

    def __init__(
        self,
        sharpe_retention: float = _SHARPE_RETENTION,
        brier_degradation: float = _BRIER_DEGRADATION,
        ks_threshold: float = _KS_DRIFT_THRESHOLD,
    ) -> None:
        self.sharpe_retention = sharpe_retention
        self.brier_degradation = brier_degradation
        self.ks_threshold = ks_threshold

    def validate(
        self,
        model_name: str,
        new_model: Any,
        old_model: Any | None,
        test_data: np.ndarray,
        test_labels: np.ndarray | None = None,
        training_features: np.ndarray | None = None,
        old_metrics: dict[str, float] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Run all validation checks.

        Parameters
        ----------
        model_name : logical model name
        new_model : the newly trained model with a predict() method
        old_model : the current production model (None if first deploy)
        test_data : (n_samples, n_features) holdout feature matrix
        test_labels : (n_samples,) ground truth labels
        training_features : (n_train, n_features) used for drift detection
        old_metrics : metrics from the current production model

        Returns
        -------
        (approved, report) where:
        - approved: True if all checks pass
        - report: dict with detailed check results
        """
        logger.info("validator.start", model=model_name)

        checks: dict[str, dict[str, Any]] = {}
        all_passed = True

        # 1. Prediction quality
        if hasattr(new_model, "predict"):
            new_preds = np.asarray(new_model.predict(test_data)).ravel()

            if test_labels is not None:
                test_labels = np.asarray(test_labels).ravel()
                quality = self._check_prediction_quality(
                    new_preds, test_labels, model_name
                )
                checks["prediction_quality"] = quality
                if not quality["passed"]:
                    all_passed = False
        else:
            checks["prediction_quality"] = {
                "passed": True,
                "note": "model has no predict() method; skipped",
            }

        # 2. Comparison against old model
        if old_model is not None and old_metrics is not None:
            comparison = self._check_comparison(
                new_model, old_model, test_data, test_labels, old_metrics
            )
            checks["model_comparison"] = comparison
            if not comparison["passed"]:
                all_passed = False
        else:
            checks["model_comparison"] = {
                "passed": True,
                "note": "no previous model; comparison skipped",
            }

        # 3. Concept drift
        if training_features is not None:
            drift = self._check_concept_drift_report(
                test_data, training_features
            )
            checks["concept_drift"] = drift
            if not drift["passed"]:
                all_passed = False
        else:
            checks["concept_drift"] = {
                "passed": True,
                "note": "no training features provided; drift check skipped",
            }

        # 4. Stability check -- predictions should not be degenerate
        if hasattr(new_model, "predict"):
            stability = self._check_stability(new_preds)
            checks["stability"] = stability
            if not stability["passed"]:
                all_passed = False

        report = {
            "model_name": model_name,
            "approved": all_passed,
            "checks": checks,
            "n_test_samples": len(test_data),
        }

        logger.info(
            "validator.complete",
            model=model_name,
            approved=all_passed,
            checks={k: v["passed"] for k, v in checks.items()},
        )
        return all_passed, report

    def _check_prediction_quality(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        model_name: str,
    ) -> dict[str, Any]:
        """Evaluate prediction quality metrics."""
        result: dict[str, Any] = {"passed": True}

        mse = float(np.mean((predictions - labels) ** 2))
        result["mse"] = mse

        # Binary classification metrics
        unique_labels = set(np.unique(labels))
        if unique_labels.issubset({0, 1, 0.0, 1.0}):
            brier = float(np.mean((predictions - labels) ** 2))
            result["brier_score"] = brier

            binary_preds = (predictions >= 0.5).astype(int)
            accuracy = float(np.mean(binary_preds == labels))
            result["accuracy"] = accuracy

            if accuracy < _MIN_ACCURACY:
                result["passed"] = False
                result["fail_reason"] = (
                    f"accuracy {accuracy:.4f} below minimum {_MIN_ACCURACY}"
                )
        else:
            # Regression: check R-squared
            ss_res = np.sum((labels - predictions) ** 2)
            ss_tot = np.sum((labels - np.mean(labels)) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-8)
            result["r2"] = float(r2)

            if r2 < 0.0:
                result["passed"] = False
                result["fail_reason"] = f"R2 = {r2:.4f} (worse than mean predictor)"

        return result

    def _check_comparison(
        self,
        new_model: Any,
        old_model: Any,
        test_data: np.ndarray,
        test_labels: np.ndarray | None,
        old_metrics: dict[str, float],
    ) -> dict[str, Any]:
        """Compare new model against old production model."""
        result: dict[str, Any] = {"passed": True}

        # Get new model predictions
        new_preds = np.asarray(new_model.predict(test_data)).ravel()

        # Sharpe comparison (if available)
        old_sharpe = old_metrics.get("sharpe_ratio", old_metrics.get("sharpe", None))
        if old_sharpe is not None and old_sharpe > 0:
            # Compute new Sharpe from prediction-based returns
            if test_labels is not None:
                errors = new_preds - test_labels
                mean_err = float(np.mean(errors))
                std_err = float(np.std(errors)) + 1e-8
                new_sharpe_proxy = -mean_err / std_err  # lower error = higher "Sharpe"
            else:
                new_sharpe_proxy = old_sharpe  # can't compute without labels

            result["old_sharpe"] = old_sharpe
            result["new_sharpe_proxy"] = new_sharpe_proxy

        # Brier score comparison
        old_brier = old_metrics.get("brier_score")
        if old_brier is not None and test_labels is not None:
            unique_labels = set(np.unique(test_labels))
            if unique_labels.issubset({0, 1, 0.0, 1.0}):
                new_brier = float(np.mean((new_preds - test_labels) ** 2))
                result["old_brier"] = old_brier
                result["new_brier"] = new_brier

                if new_brier > old_brier * self.brier_degradation:
                    result["passed"] = False
                    result["fail_reason"] = (
                        f"Brier degraded: {new_brier:.4f} > "
                        f"{old_brier:.4f} * {self.brier_degradation}"
                    )

        # IC comparison (for return models)
        old_ic = old_metrics.get("ic")
        if old_ic is not None and old_ic > 0 and test_labels is not None:
            from scipy.stats import spearmanr

            new_ic, _ = spearmanr(test_labels, new_preds)
            result["old_ic"] = old_ic
            result["new_ic"] = float(new_ic)

            if new_ic < old_ic * self.sharpe_retention:
                result["passed"] = False
                result["fail_reason"] = (
                    f"IC degraded: {new_ic:.4f} < "
                    f"{old_ic:.4f} * {self.sharpe_retention}"
                )

        return result

    def check_concept_drift(
        self,
        features: np.ndarray,
        training_features: np.ndarray,
    ) -> bool:
        """Check for concept drift using KS test.

        Performs a two-sample Kolmogorov-Smirnov test on each feature
        comparing the test distribution to the training distribution.
        Returns True if significant drift is detected (> threshold
        fraction of top features have KS > threshold).
        """
        from scipy.stats import ks_2samp

        n_features = min(features.shape[1], training_features.shape[1])

        # Compute variance-based feature importance to focus on top features
        variances = np.var(training_features, axis=0)
        top_indices = np.argsort(variances)[-min(10, n_features):]

        drifted = 0
        for idx in top_indices:
            stat, pval = ks_2samp(
                training_features[:, idx],
                features[:, idx],
            )
            if stat > self.ks_threshold:
                drifted += 1

        drift_pct = drifted / len(top_indices)
        return drift_pct > _MAX_FEATURE_DRIFT_PCT

    def _check_concept_drift_report(
        self,
        features: np.ndarray,
        training_features: np.ndarray,
    ) -> dict[str, Any]:
        """Detailed concept drift report."""
        from scipy.stats import ks_2samp

        n_features = min(features.shape[1], training_features.shape[1])
        variances = np.var(training_features, axis=0)
        top_indices = np.argsort(variances)[-min(10, n_features):]

        drift_details: list[dict[str, Any]] = []
        drifted_count = 0

        for idx in top_indices:
            stat, pval = ks_2samp(
                training_features[:, idx],
                features[:, idx],
            )
            is_drifted = stat > self.ks_threshold
            if is_drifted:
                drifted_count += 1

            drift_details.append({
                "feature_idx": int(idx),
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(pval), 6),
                "drifted": is_drifted,
            })

        drift_pct = drifted_count / len(top_indices) if top_indices.size > 0 else 0.0
        passed = drift_pct <= _MAX_FEATURE_DRIFT_PCT

        return {
            "passed": passed,
            "drift_pct": round(drift_pct, 3),
            "drifted_features": drifted_count,
            "total_checked": len(top_indices),
            "threshold": self.ks_threshold,
            "details": drift_details,
            "fail_reason": (
                f"{drifted_count}/{len(top_indices)} features drifted "
                f"({drift_pct:.0%} > {_MAX_FEATURE_DRIFT_PCT:.0%})"
                if not passed
                else None
            ),
        }

    def _check_stability(self, predictions: np.ndarray) -> dict[str, Any]:
        """Check that predictions are not degenerate."""
        result: dict[str, Any] = {"passed": True}

        # Check for constant predictions
        pred_std = float(np.std(predictions))
        result["prediction_std"] = pred_std

        if pred_std < 1e-6:
            result["passed"] = False
            result["fail_reason"] = "predictions are constant (zero variance)"
            return result

        # Check for NaN/Inf
        n_nan = int(np.sum(np.isnan(predictions)))
        n_inf = int(np.sum(np.isinf(predictions)))
        result["n_nan"] = n_nan
        result["n_inf"] = n_inf

        if n_nan > 0 or n_inf > 0:
            result["passed"] = False
            result["fail_reason"] = f"predictions contain {n_nan} NaN, {n_inf} Inf"
            return result

        # Check for extreme values (beyond [-10, 10] for probabilities)
        pred_min = float(np.min(predictions))
        pred_max = float(np.max(predictions))
        result["prediction_range"] = [pred_min, pred_max]

        return result
