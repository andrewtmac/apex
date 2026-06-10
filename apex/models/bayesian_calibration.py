"""
APEX Model G: Bayesian Calibration

Wraps Model A's (XGBoost) raw probability output in uncertainty quantification.

Pipeline:
    raw_prob -> Platt scaling -> Isotonic regression -> Beta-Binomial posterior

Output: (mean_probability, credible_interval_lower, credible_interval_upper).

Trade only when the credible interval for edge is ENTIRELY above zero.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from scipy import optimize, stats
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class BayesianCalibrationModel:
    """
    Bayesian calibration layer for binary prediction models.

    Combines two classical calibration methods (Platt scaling and isotonic
    regression) with a Beta-Binomial posterior that updates as new
    predictions resolve.

    Usage
    -----
    >>> cal = BayesianCalibrationModel()
    >>> cal.fit_calibration(val_predictions, val_actuals)
    >>> mean, lo, hi = cal.get_credible_interval(raw_prob=0.65)
    >>> if cal.has_edge(raw_prob=0.65, market_price=0.55):
    ...     # take the trade
    """

    def __init__(
        self,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        use_isotonic: bool = True,
        use_platt: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        alpha_prior : Beta prior alpha (uniform = 1.0).
        beta_prior : Beta prior beta (uniform = 1.0).
        use_isotonic : whether to apply isotonic regression calibration.
        use_platt : whether to apply Platt scaling calibration.
        """
        # Platt scaling parameters: P(y=1|f) = 1 / (1 + exp(a*f + b))
        self.platt_a: float = 0.0
        self.platt_b: float = 0.0
        self._platt_fitted: bool = False

        # Isotonic regression
        self.isotonic: IsotonicRegression | None = None
        self._isotonic_fitted: bool = False

        # Beta prior
        self.alpha_prior: float = alpha_prior
        self.beta_prior: float = beta_prior

        # Online observations for Bayesian updating
        # Each entry: (predicted_probability, actual_outcome)
        self.observations: list[tuple[float, int]] = []

        # Binned posterior accumulators: for each probability bucket,
        # track alpha_posterior, beta_posterior
        self._n_bins: int = 20
        self._bin_alphas: np.ndarray = np.full(self._n_bins, alpha_prior)
        self._bin_betas: np.ndarray = np.full(self._n_bins, beta_prior)

        self._use_isotonic = use_isotonic
        self._use_platt = use_platt

    # ------------------------------------------------------------------
    # Calibration fitting
    # ------------------------------------------------------------------

    def fit_calibration(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> None:
        """
        Fit Platt scaling and isotonic regression on validation data.

        Parameters
        ----------
        predictions : (n,) raw model outputs (probabilities or scores).
        actuals : (n,) binary labels (0 or 1).
        """
        predictions = np.asarray(predictions, dtype=np.float64).ravel()
        actuals = np.asarray(actuals, dtype=np.float64).ravel()

        if len(predictions) != len(actuals):
            raise ValueError("predictions and actuals must have the same length")

        # 1. Platt scaling: find a, b that minimise NLL of sigmoid(a*f + b)
        if self._use_platt:
            self._fit_platt(predictions, actuals)

        # After Platt, apply Platt before isotonic
        platt_preds = self._apply_platt(predictions) if self._platt_fitted else predictions

        # 2. Isotonic regression on Platt-scaled outputs
        if self._use_isotonic:
            self.isotonic = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip",
            )
            self.isotonic.fit(platt_preds, actuals)
            self._isotonic_fitted = True

        # 3. Initialise binned Beta posterior from calibration data
        calibrated = self.calibrate_array(predictions)
        self._initialise_bins(calibrated, actuals)

        n_pos = int(actuals.sum())
        n_neg = int(len(actuals) - n_pos)
        logger.info(
            "Calibration fitted on %d samples (%d pos, %d neg). "
            "Platt: a=%.4f b=%.4f",
            len(predictions), n_pos, n_neg, self.platt_a, self.platt_b,
        )

    def _fit_platt(self, predictions: np.ndarray, actuals: np.ndarray) -> None:
        """
        Fit Platt scaling: P(y=1|f) = sigmoid(a*f + b).

        Uses the regularised MLE formulation from Platt (1999).
        """
        # Target values with Bayesian correction (Platt 1999, Eq. 3)
        n_pos = actuals.sum()
        n_neg = len(actuals) - n_pos
        t_pos = (n_pos + 1) / (n_pos + 2)
        t_neg = 1 / (n_neg + 2)
        targets = np.where(actuals > 0.5, t_pos, t_neg)

        def neg_log_likelihood(params: np.ndarray) -> float:
            a, b = params
            p = 1.0 / (1.0 + np.exp(a * predictions + b))
            p = np.clip(p, 1e-12, 1 - 1e-12)
            nll = -np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p))
            return float(nll)

        result = optimize.minimize(
            neg_log_likelihood,
            x0=np.array([0.0, 0.0]),
            method="Nelder-Mead",
            options={"maxiter": 1000},
        )
        self.platt_a, self.platt_b = result.x
        self._platt_fitted = True

    def _apply_platt(self, scores: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to raw scores."""
        return 1.0 / (1.0 + np.exp(self.platt_a * scores + self.platt_b))

    def _initialise_bins(
        self, calibrated: np.ndarray, actuals: np.ndarray,
    ) -> None:
        """Populate binned Beta posteriors from calibration data."""
        self._bin_alphas = np.full(self._n_bins, self.alpha_prior)
        self._bin_betas = np.full(self._n_bins, self.beta_prior)

        bins = np.clip((calibrated * self._n_bins).astype(int), 0, self._n_bins - 1)
        for b, y in zip(bins, actuals):
            if y > 0.5:
                self._bin_alphas[b] += 1
            else:
                self._bin_betas[b] += 1

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, raw_probability: float) -> float:
        """
        Apply the full calibration pipeline to a single raw model output.

        Pipeline: raw -> Platt -> isotonic -> clipped to [0, 1].
        """
        p = float(raw_probability)

        if self._platt_fitted:
            p = float(self._apply_platt(np.array([p]))[0])

        if self._isotonic_fitted and self.isotonic is not None:
            p = float(self.isotonic.predict(np.array([p]))[0])

        return float(np.clip(p, 0.0, 1.0))

    def calibrate_array(self, raw_probabilities: np.ndarray) -> np.ndarray:
        """Calibrate an array of raw probabilities."""
        p = np.asarray(raw_probabilities, dtype=np.float64)

        if self._platt_fitted:
            p = self._apply_platt(p)

        if self._isotonic_fitted and self.isotonic is not None:
            p = self.isotonic.predict(p)

        return np.clip(p, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Bayesian updating
    # ------------------------------------------------------------------

    def update_posterior(self, predicted: float, actual: int) -> None:
        """
        Bayesian update with a single new observation.

        Parameters
        ----------
        predicted : the calibrated probability that was predicted.
        actual : 1 if event occurred, 0 otherwise.
        """
        self.observations.append((float(predicted), int(actual)))

        bin_idx = min(int(predicted * self._n_bins), self._n_bins - 1)
        bin_idx = max(bin_idx, 0)

        if actual > 0:
            self._bin_alphas[bin_idx] += 1
        else:
            self._bin_betas[bin_idx] += 1

    def batch_update_posterior(
        self, predicted: np.ndarray, actual: np.ndarray,
    ) -> None:
        """Update posterior with a batch of observations."""
        for p, a in zip(predicted.ravel(), actual.ravel()):
            self.update_posterior(float(p), int(a))

    # ------------------------------------------------------------------
    # Credible intervals
    # ------------------------------------------------------------------

    def get_credible_interval(
        self,
        raw_probability: float,
        alpha: float = 0.05,
    ) -> tuple[float, float, float]:
        """
        Compute Bayesian credible interval for the true probability.

        Parameters
        ----------
        raw_probability : raw model output (pre-calibration).
        alpha : significance level. 0.05 -> 95% credible interval.

        Returns
        -------
        (mean, lower, upper) -- posterior mean and alpha/2 credible bounds.
        """
        calibrated = self.calibrate(raw_probability)

        bin_idx = min(int(calibrated * self._n_bins), self._n_bins - 1)
        bin_idx = max(bin_idx, 0)

        a = self._bin_alphas[bin_idx]
        b = self._bin_betas[bin_idx]

        posterior = stats.beta(a, b)
        mean = float(posterior.mean())
        lower = float(posterior.ppf(alpha / 2))
        upper = float(posterior.ppf(1 - alpha / 2))

        return mean, lower, upper

    def get_edge_interval(
        self,
        raw_probability: float,
        market_price: float,
        alpha: float = 0.05,
    ) -> tuple[float, float, float]:
        """
        Compute credible interval for the EDGE (= true_prob - market_price).

        Returns
        -------
        (mean_edge, lower_edge, upper_edge).
        """
        mean, lower, upper = self.get_credible_interval(raw_probability, alpha)
        return (
            mean - market_price,
            lower - market_price,
            upper - market_price,
        )

    # ------------------------------------------------------------------
    # Decision functions
    # ------------------------------------------------------------------

    def has_edge(
        self,
        raw_probability: float,
        market_price: float,
        alpha: float = 0.05,
    ) -> bool:
        """
        True if the ENTIRE credible interval implies positive edge.

        This is the conservative gate: we trade only when we are (1-alpha)
        confident that our probability exceeds the market price.
        """
        _, lower_edge, _ = self.get_edge_interval(
            raw_probability, market_price, alpha,
        )
        return lower_edge > 0.0

    def kelly_fraction(
        self,
        raw_probability: float,
        market_price: float,
        max_fraction: float = 0.25,
    ) -> float:
        """
        Compute Kelly-optimal bet fraction using posterior mean.

        f* = (p * b - q) / b   where b = odds, p = true prob, q = 1-p.
        Capped at *max_fraction* and floored at 0.

        Returns
        -------
        float in [0, max_fraction].
        """
        mean, _, _ = self.get_credible_interval(raw_probability)
        p = mean
        q = 1.0 - p

        if market_price <= 0 or market_price >= 1:
            return 0.0

        # Decimal odds from market price
        b = (1.0 / market_price) - 1.0
        if b <= 0:
            return 0.0

        f = (p * b - q) / b
        return float(np.clip(f, 0.0, max_fraction))

    # ------------------------------------------------------------------
    # Calibration diagnostics
    # ------------------------------------------------------------------

    def calibration_error(self, n_bins: int = 10) -> dict[str, float]:
        """
        Compute ECE (Expected Calibration Error) and MCE (Maximum Calibration
        Error) from the observation history.

        Returns
        -------
        Dict with keys: ``ece``, ``mce``, ``n_observations``.
        """
        if len(self.observations) < 10:
            return {"ece": float("nan"), "mce": float("nan"), "n_observations": len(self.observations)}

        preds = np.array([o[0] for o in self.observations])
        actuals = np.array([o[1] for o in self.observations])

        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (preds >= lo) & (preds < hi)
            if mask.sum() == 0:
                continue
            bin_acc = actuals[mask].mean()
            bin_conf = preds[mask].mean()
            bin_weight = mask.sum() / len(preds)
            err = abs(bin_acc - bin_conf)
            ece += bin_weight * err
            mce = max(mce, err)

        return {
            "ece": float(ece),
            "mce": float(mce),
            "n_observations": len(self.observations),
        }

    def reliability_curve(
        self, n_bins: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute reliability (calibration) curve data.

        Returns
        -------
        (bin_centers, empirical_fractions, bin_counts).
        """
        if len(self.observations) < 2:
            empty = np.array([])
            return empty, empty, empty

        preds = np.array([o[0] for o in self.observations])
        actuals = np.array([o[1] for o in self.observations])

        bin_edges = np.linspace(0, 1, n_bins + 1)
        centers = []
        fractions = []
        counts = []

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (preds >= lo) & (preds < hi)
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            centers.append((lo + hi) / 2)
            fractions.append(float(actuals[mask].mean()))
            counts.append(cnt)

        return np.array(centers), np.array(fractions), np.array(counts)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Return serialisable state dict."""
        return {
            "platt_a": self.platt_a,
            "platt_b": self.platt_b,
            "platt_fitted": self._platt_fitted,
            "isotonic": self.isotonic,
            "isotonic_fitted": self._isotonic_fitted,
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
            "observations": self.observations,
            "bin_alphas": self._bin_alphas.tolist(),
            "bin_betas": self._bin_betas.tolist(),
            "n_bins": self._n_bins,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore from a state dict."""
        self.platt_a = state["platt_a"]
        self.platt_b = state["platt_b"]
        self._platt_fitted = state["platt_fitted"]
        self.isotonic = state["isotonic"]
        self._isotonic_fitted = state["isotonic_fitted"]
        self.alpha_prior = state["alpha_prior"]
        self.beta_prior = state["beta_prior"]
        self.observations = state["observations"]
        self._bin_alphas = np.array(state["bin_alphas"])
        self._bin_betas = np.array(state["bin_betas"])
        self._n_bins = state["n_bins"]

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_obs = len(self.observations)
        platt = f"a={self.platt_a:.3f},b={self.platt_b:.3f}" if self._platt_fitted else "unfitted"
        return (
            f"<BayesianCalibrationModel [platt={platt}, "
            f"isotonic={'fitted' if self._isotonic_fitted else 'unfitted'}, "
            f"observations={n_obs}]>"
        )
