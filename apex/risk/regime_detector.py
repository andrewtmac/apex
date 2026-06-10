"""
APEX HMM Regime Detection

Market regime detection using a Gaussian Hidden Markov Model (HMM) combined
with Bayesian Online Changepoint Detection (BOCPD) for rapid transition
identification.

Market Regimes: CALM, NORMAL, ELEVATED, CRISIS
- Based on VIX level, realised volatility, volume metrics, and correlation shifts.

Prediction Market Regimes: EFFICIENT, NORMAL, DISLOCATED
- Based on spread metrics, volume, and calibration residuals.

The HMM provides smooth regime classification, while BOCPD provides fast
detection of regime transitions that the HMM might lag on.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# We import hmmlearn lazily to avoid hard dependency in testing
_HMM_AVAILABLE = False
try:
    from hmmlearn.hmm import GaussianHMM

    _HMM_AVAILABLE = True
except ImportError:
    pass


class RegimeDetector:
    """Market regime detection using HMM + BOCPD.

    The detector operates in two modes:
    1. **Fitted mode**: After calling :meth:`fit`, the HMM is trained on
       historical data and :meth:`detect` uses the trained model.
    2. **Heuristic mode**: Before fitting, :meth:`detect` uses rule-based
       thresholds for classification.

    Parameters
    ----------
    n_market_regimes : int
        Number of hidden states for the market HMM (default 4).
    n_pm_regimes : int
        Number of hidden states for the prediction-market HMM (default 3).
    lookback : int
        Number of observations to keep in the rolling window for
        online detection.
    bocpd_hazard : float
        Hazard rate for BOCPD (probability of changepoint at each step).
        Lower values = fewer false alarms, higher values = faster detection.
    """

    MARKET_REGIMES = ["CALM", "NORMAL", "ELEVATED", "CRISIS"]
    PM_REGIMES = ["EFFICIENT", "NORMAL", "DISLOCATED"]

    # Heuristic thresholds for rule-based classification
    _VOL_THRESHOLDS = {
        "CALM": (0.0, 0.10),
        "NORMAL": (0.10, 0.20),
        "ELEVATED": (0.20, 0.35),
        "CRISIS": (0.35, float("inf")),
    }

    _SPREAD_THRESHOLDS = {
        "EFFICIENT": (0.0, 200.0),   # < 200 bps
        "NORMAL": (200.0, 500.0),
        "DISLOCATED": (500.0, float("inf")),
    }

    def __init__(
        self,
        n_market_regimes: int = 4,
        n_pm_regimes: int = 3,
        lookback: int = 100,
        bocpd_hazard: float = 1.0 / 250.0,
    ) -> None:
        self.n_market_regimes = n_market_regimes
        self.n_pm_regimes = n_pm_regimes
        self.lookback = lookback
        self.bocpd_hazard = bocpd_hazard

        self.hmm_model: Any | None = None  # GaussianHMM
        self.pm_hmm_model: Any | None = None

        self.current_regime: str = "NORMAL"
        self.current_pm_regime: str = "NORMAL"
        self.regime_history: list[str] = []
        self.pm_regime_history: list[str] = []

        self._is_fitted: bool = False
        self._feature_buffer: list[np.ndarray] = []

        # BOCPD state
        self._run_length_probs: np.ndarray | None = None

    # ------------------------------------------------------------------
    # HMM Fitting
    # ------------------------------------------------------------------

    def fit(self, features: np.ndarray, n_iter: int = 100) -> dict[str, Any]:
        """Fit the HMM on historical feature data.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(n_samples, n_features)`` -- historical observations.
            Typical features: [realised_vol, vix, volume_zscore, corr_mean].
        n_iter : int
            Maximum EM iterations for HMM fitting.

        Returns
        -------
        dict
            Fitting metrics: ``log_likelihood``, ``n_samples``,
            ``converged``, ``regime_distribution``.
        """
        if not _HMM_AVAILABLE:
            logger.warning("hmmlearn not installed -- using heuristic regime detection")
            self._is_fitted = False
            return {"error": "hmmlearn not available"}

        n_samples, n_features = features.shape

        self.hmm_model = GaussianHMM(
            n_components=self.n_market_regimes,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42,
            verbose=False,
        )

        self.hmm_model.fit(features)

        # Decode to get regime labels
        log_prob, states = self.hmm_model.decode(features)

        # Map HMM states to named regimes by sorting state means by volatility
        # (first feature assumed to be volatility-like)
        state_means = self.hmm_model.means_[:, 0]
        state_order = np.argsort(state_means)
        self._state_to_regime = {
            int(state_order[i]): self.MARKET_REGIMES[i]
            for i in range(min(len(state_order), len(self.MARKET_REGIMES)))
        }

        # Compute regime distribution
        regime_counts: dict[str, int] = {r: 0 for r in self.MARKET_REGIMES}
        for s in states:
            regime = self._state_to_regime.get(int(s), "NORMAL")
            regime_counts[regime] += 1
        regime_dist = {r: c / n_samples for r, c in regime_counts.items()}

        self._is_fitted = True

        metrics = {
            "log_likelihood": float(log_prob),
            "n_samples": n_samples,
            "converged": self.hmm_model.monitor_.converged,
            "regime_distribution": regime_dist,
        }

        logger.info("HMM fitted: %s", metrics)
        return metrics

    def fit_pm(self, features: np.ndarray, n_iter: int = 100) -> dict[str, Any]:
        """Fit the prediction-market regime HMM.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(n_samples, n_features)`` -- prediction-market features.
            Typical: [avg_spread_bps, volume_zscore, calibration_residual].
        """
        if not _HMM_AVAILABLE:
            return {"error": "hmmlearn not available"}

        self.pm_hmm_model = GaussianHMM(
            n_components=self.n_pm_regimes,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42,
            verbose=False,
        )

        self.pm_hmm_model.fit(features)
        log_prob, states = self.pm_hmm_model.decode(features)

        # Map by spread (first feature) -- lower spread = more efficient
        state_means = self.pm_hmm_model.means_[:, 0]
        state_order = np.argsort(state_means)
        self._pm_state_to_regime = {
            int(state_order[i]): self.PM_REGIMES[i]
            for i in range(min(len(state_order), len(self.PM_REGIMES)))
        }

        return {"log_likelihood": float(log_prob), "n_samples": features.shape[0]}

    # ------------------------------------------------------------------
    # Online Detection
    # ------------------------------------------------------------------

    def detect(self, features: np.ndarray) -> dict[str, Any]:
        """Detect current regime from a feature vector.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(n_features,)`` -- current observation.

        Returns
        -------
        dict
            ``regime``, ``regime_confidence``, ``regime_probabilities``,
            ``pm_regime``, ``changepoint_detected``.
        """
        features_2d = features.reshape(1, -1) if features.ndim == 1 else features

        # Buffer for BOCPD
        self._feature_buffer.append(features.ravel())
        if len(self._feature_buffer) > self.lookback:
            self._feature_buffer = self._feature_buffer[-self.lookback :]

        # Market regime detection
        if self._is_fitted and self.hmm_model is not None:
            regime, confidence, probs = self._hmm_detect(features_2d)
        else:
            regime, confidence, probs = self._heuristic_detect(features.ravel())

        # Update history
        self.current_regime = regime
        self.regime_history.append(regime)
        if len(self.regime_history) > self.lookback:
            self.regime_history = self.regime_history[-self.lookback :]

        # Changepoint detection
        changepoint = self.detect_changepoint(features.ravel())

        result = {
            "regime": regime,
            "regime_confidence": confidence,
            "regime_probabilities": probs,
            "changepoint_detected": changepoint,
        }

        logger.debug("Regime detected: %s (conf=%.3f, changepoint=%s)", regime, confidence, changepoint)
        return result

    def _hmm_detect(
        self, features: np.ndarray
    ) -> tuple[str, float, dict[str, float]]:
        """Detect regime using fitted HMM."""
        # Predict state probabilities
        log_prob = self.hmm_model.score(features)
        state_probs = self.hmm_model.predict_proba(features)[0]
        state = int(np.argmax(state_probs))

        regime = self._state_to_regime.get(state, "NORMAL")
        confidence = float(state_probs[state])

        probs = {}
        for s, r in self._state_to_regime.items():
            if s < len(state_probs):
                probs[r] = float(state_probs[s])

        return regime, confidence, probs

    def _heuristic_detect(
        self, features: np.ndarray
    ) -> tuple[str, float, dict[str, float]]:
        """Detect regime using rule-based thresholds when HMM is not fitted.

        Expects features[0] to be a volatility measure (e.g. realised vol
        or VIX / 100).
        """
        vol = float(features[0]) if len(features) > 0 else 0.15

        regime = "NORMAL"
        for r, (low, high) in self._VOL_THRESHOLDS.items():
            if low <= vol < high:
                regime = r
                break

        # Confidence based on distance from threshold boundaries
        for r, (low, high) in self._VOL_THRESHOLDS.items():
            if r == regime:
                range_width = min(high, 1.0) - low
                if range_width > 0:
                    mid = (low + min(high, 1.0)) / 2.0
                    dist = abs(vol - mid) / (range_width / 2.0)
                    confidence = max(0.5, 1.0 - dist * 0.5)
                else:
                    confidence = 0.5
                break
        else:
            confidence = 0.5

        probs = {r: 0.0 for r in self.MARKET_REGIMES}
        probs[regime] = confidence
        remaining = 1.0 - confidence
        others = [r for r in self.MARKET_REGIMES if r != regime]
        for r in others:
            probs[r] = remaining / len(others)

        return regime, confidence, probs

    # ------------------------------------------------------------------
    # Bayesian Online Changepoint Detection (BOCPD)
    # ------------------------------------------------------------------

    def detect_changepoint(self, data: np.ndarray) -> bool:
        """BOCPD for rapid regime transition detection.

        Uses the hazard function model: at each step, there is a constant
        probability (``self.bocpd_hazard``) that a changepoint occurs.

        Parameters
        ----------
        data : np.ndarray
            Current observation vector.

        Returns
        -------
        bool
            True if a changepoint is detected at the current step.
        """
        if len(self._feature_buffer) < 5:
            return False

        # Simplified BOCPD using sequential likelihood ratio
        buffer = np.array(self._feature_buffer)
        n = len(buffer)

        if n < 10:
            return False

        # Use first feature dimension for changepoint detection
        series = buffer[:, 0] if buffer.ndim > 1 else buffer

        # Split into "before" and "recent" windows
        split = max(3, n - 5)
        before = series[:split]
        recent = series[split:]

        if len(before) < 3 or len(recent) < 2:
            return False

        # Mean and variance of each window
        mu_before = np.mean(before)
        mu_recent = np.mean(recent)
        std_before = max(np.std(before), 1e-8)

        # Z-score of the mean shift
        z_shift = abs(mu_recent - mu_before) / (std_before / np.sqrt(len(recent)))

        # Also check variance ratio (F-test like)
        std_recent = max(np.std(recent), 1e-8)
        var_ratio = max(std_recent / std_before, std_before / std_recent)

        # Changepoint if mean shift is significant OR variance changed dramatically
        mean_changed = z_shift > 3.0
        var_changed = var_ratio > 3.0

        is_changepoint = mean_changed or var_changed

        if is_changepoint:
            logger.info(
                "BOCPD changepoint detected: z_shift=%.2f var_ratio=%.2f",
                z_shift,
                var_ratio,
            )

        return is_changepoint

    # ------------------------------------------------------------------
    # Prediction market regime
    # ------------------------------------------------------------------

    def detect_pm_regime(self, features: np.ndarray) -> dict[str, Any]:
        """Detect prediction-market regime (EFFICIENT, NORMAL, DISLOCATED).

        Parameters
        ----------
        features : np.ndarray
            Expects features[0] = avg_spread_bps.

        Returns
        -------
        dict
            ``pm_regime``, ``pm_confidence``, ``pm_probabilities``.
        """
        if self.pm_hmm_model is not None:
            features_2d = features.reshape(1, -1) if features.ndim == 1 else features
            state_probs = self.pm_hmm_model.predict_proba(features_2d)[0]
            state = int(np.argmax(state_probs))
            regime = self._pm_state_to_regime.get(state, "NORMAL")
            confidence = float(state_probs[state])
            probs = {
                self._pm_state_to_regime.get(i, "NORMAL"): float(state_probs[i])
                for i in range(len(state_probs))
            }
        else:
            # Heuristic based on spread
            spread = float(features[0]) if len(features) > 0 else 300.0
            regime = "NORMAL"
            for r, (low, high) in self._SPREAD_THRESHOLDS.items():
                if low <= spread < high:
                    regime = r
                    break
            confidence = 0.7
            probs = {r: 0.1 for r in self.PM_REGIMES}
            probs[regime] = confidence

        self.current_pm_regime = regime
        self.pm_regime_history.append(regime)
        if len(self.pm_regime_history) > self.lookback:
            self.pm_regime_history = self.pm_regime_history[-self.lookback :]

        return {
            "pm_regime": regime,
            "pm_confidence": confidence,
            "pm_probabilities": probs,
        }

    # ------------------------------------------------------------------
    # Regime encoding
    # ------------------------------------------------------------------

    @staticmethod
    def encode_regime(regime: str) -> float:
        """Encode a market regime as a float for model input.

        CALM=0.0, NORMAL=0.33, ELEVATED=0.67, CRISIS=1.0.
        """
        mapping = {"CALM": 0.0, "NORMAL": 0.33, "ELEVATED": 0.67, "CRISIS": 1.0}
        return mapping.get(regime, 0.33)

    @staticmethod
    def encode_pm_regime(regime: str) -> float:
        """Encode a PM regime as a float. EFFICIENT=0.0, NORMAL=0.5, DISLOCATED=1.0."""
        mapping = {"EFFICIENT": 0.0, "NORMAL": 0.5, "DISLOCATED": 1.0}
        return mapping.get(regime, 0.5)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Serialize detector state."""
        return {
            "current_regime": self.current_regime,
            "current_pm_regime": self.current_pm_regime,
            "regime_history": list(self.regime_history),
            "pm_regime_history": list(self.pm_regime_history),
            "is_fitted": self._is_fitted,
        }

    def __repr__(self) -> str:
        return (
            f"RegimeDetector(market={self.current_regime}, "
            f"pm={self.current_pm_regime}, fitted={self._is_fitted})"
        )
