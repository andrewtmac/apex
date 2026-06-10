"""
APEX S12: Regime-Adaptive Strategy Selector

Meta-strategy that dynamically shifts capital allocation across sub-strategies
based on HMM regime detection and Thompson Sampling.  Rather than running all
strategies at full capacity all the time, this selector adjusts each strategy's
allocation weight based on which strategies perform best in the current regime.

Regime-strategy affinities:
    CALM    -> CalibrationExploit, SmartMM (tight spreads, mean reversion)
    NORMAL  -> BayesianForecaster, Convergence, InfoArb (balanced)
    ELEVATED -> Microstructure, Convergence (volatility plays)
    CRISIS  -> Reduce all, favor hedging strategies
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from apex.ensemble.signal import ApexSignal
from apex.ensemble.thompson_sampling import ThompsonSampler
from apex.risk.regime_detector import RegimeDetector
from apex.strategies.apex_strategy import ApexStrategy, ApexStrategyConfig

logger = logging.getLogger(__name__)

# Default regime-strategy affinity matrix
# Rows: strategies, Columns: CALM, NORMAL, ELEVATED, CRISIS
_AFFINITY_MATRIX: dict[str, dict[str, float]] = {
    "calibration_exploit": {"CALM": 1.0, "NORMAL": 0.8, "ELEVATED": 0.4, "CRISIS": 0.1},
    "bayesian_forecaster": {"CALM": 0.6, "NORMAL": 1.0, "ELEVATED": 0.7, "CRISIS": 0.3},
    "microstructure":      {"CALM": 0.3, "NORMAL": 0.6, "ELEVATED": 1.0, "CRISIS": 0.5},
    "info_arb":            {"CALM": 0.5, "NORMAL": 0.9, "ELEVATED": 0.8, "CRISIS": 0.2},
    "convergence":         {"CALM": 0.7, "NORMAL": 0.8, "ELEVATED": 0.9, "CRISIS": 0.4},
    "smart_mm":            {"CALM": 1.0, "NORMAL": 0.7, "ELEVATED": 0.3, "CRISIS": 0.0},
    "vol_surface":         {"CALM": 0.4, "NORMAL": 0.6, "ELEVATED": 1.0, "CRISIS": 0.6},
    "earnings":            {"CALM": 0.5, "NORMAL": 0.8, "ELEVATED": 0.5, "CRISIS": 0.2},
}


class RegimeSelectorConfig(ApexStrategyConfig, frozen=True):
    """Configuration for the Regime-Adaptive Selector."""

    strategy_name: str = "regime_selector"
    min_edge: float = 0.02
    min_ensemble_score: float = 0.50

    # Selector-specific parameters
    rebalance_interval_bars: int = 60     # Rebalance every 60 bars
    thompson_decay: float = 0.995         # Thompson sampling decay rate
    min_strategy_weight: float = 0.05     # Minimum weight for any strategy
    max_strategy_weight: float = 0.40     # Maximum weight for any strategy


class RegimeSelectorStrategy(ApexStrategy):
    """Meta-strategy that dynamically shifts capital allocation.

    Does not generate its own trading signals.  Instead, it determines
    the optimal allocation of capital across sub-strategies based on
    the current regime and each strategy's historical performance.
    """

    def __init__(self, config: RegimeSelectorConfig) -> None:
        super().__init__(config)

        self._rebalance_interval = config.rebalance_interval_bars
        self._min_weight = config.min_strategy_weight
        self._max_weight = config.max_strategy_weight

        # Strategy names
        self._strategy_names = list(_AFFINITY_MATRIX.keys())

        # Thompson sampler for each strategy
        self._strategy_sampler = ThompsonSampler(
            model_names=self._strategy_names,
            decay=config.thompson_decay,
            min_weight=config.min_strategy_weight,
        )

        # Current weights
        self._strategy_weights: dict[str, float] = {
            name: 1.0 / len(self._strategy_names)
            for name in self._strategy_names
        }

        # Performance tracking per strategy per regime
        self._regime_performance: dict[str, dict[str, list[float]]] = {
            name: {regime: [] for regime in RegimeDetector.MARKET_REGIMES}
            for name in self._strategy_names
        }

        # Rebalance tracking
        self._bars_since_rebalance: int = 0
        self._allocation_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def compute_weights(self, regime: str) -> dict[str, float]:
        """Compute strategy weights for the current regime.

        Combines the static affinity matrix with Thompson Sampling
        exploration to determine dynamic weights.

        Parameters
        ----------
        regime : str
            Current market regime.

        Returns
        -------
        dict[str, float]
            Strategy name -> weight (sums to 1.0).
        """
        # 1. Static affinity weights
        affinity_weights: dict[str, float] = {}
        for name in self._strategy_names:
            affinity = _AFFINITY_MATRIX.get(name, {}).get(regime, 0.5)
            affinity_weights[name] = affinity

        # 2. Thompson sampling weights (exploration/exploitation)
        thompson_weights = self._strategy_sampler.sample_weights()

        # 3. Blend: 60% affinity, 40% Thompson
        blended: dict[str, float] = {}
        for name in self._strategy_names:
            w = 0.6 * affinity_weights[name] + 0.4 * thompson_weights[name]
            blended[name] = w

        # 4. Constrain and normalise
        constrained = self._constrain_weights(blended)

        return constrained

    def _constrain_weights(
        self,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Apply min/max constraints and normalise weights."""
        clamped = {
            name: max(self._min_weight, min(self._max_weight, w))
            for name, w in weights.items()
        }

        total = sum(clamped.values())
        if total <= 0:
            equal = 1.0 / len(self._strategy_names)
            return {name: equal for name in self._strategy_names}

        normalised = {name: w / total for name, w in clamped.items()}
        return normalised

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def should_rebalance(self) -> bool:
        """Check if enough bars have passed since last rebalance."""
        return self._bars_since_rebalance >= self._rebalance_interval

    def rebalance(self, regime: str) -> dict[str, float]:
        """Rebalance strategy weights.

        Parameters
        ----------
        regime : str
            Current market regime.

        Returns
        -------
        dict[str, float]
            Updated strategy weights.
        """
        old_weights = self._strategy_weights.copy()
        new_weights = self.compute_weights(regime)

        self._strategy_weights = new_weights
        self._bars_since_rebalance = 0

        # Apply Thompson decay
        self._strategy_sampler.decay_params()

        # Record
        self._allocation_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "regime": regime,
            "old_weights": old_weights,
            "new_weights": new_weights,
        })

        logger.info(
            "Regime selector rebalanced (regime=%s): %s",
            regime,
            {k: f"{v:.3f}" for k, v in new_weights.items()},
        )

        return new_weights

    # ------------------------------------------------------------------
    # Performance recording
    # ------------------------------------------------------------------

    def record_strategy_result(
        self,
        strategy_name: str,
        regime: str,
        pnl: float,
    ) -> None:
        """Record a trade result for a strategy in a given regime.

        Parameters
        ----------
        strategy_name : str
            Name of the strategy.
        regime : str
            Market regime when the trade was taken.
        pnl : float
            PnL of the trade.
        """
        if strategy_name in self._regime_performance:
            if regime in self._regime_performance[strategy_name]:
                self._regime_performance[strategy_name][regime].append(pnl)

                # Keep rolling window
                max_history = 200
                if len(self._regime_performance[strategy_name][regime]) > max_history:
                    self._regime_performance[strategy_name][regime] = (
                        self._regime_performance[strategy_name][regime][-max_history:]
                    )

        # Update Thompson sampler
        if strategy_name in self._strategy_names:
            self._strategy_sampler.update_continuous(strategy_name, pnl, threshold=0.0)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _generate_signal(self, features: np.ndarray) -> ApexSignal | None:
        """The regime selector does not generate trading signals.

        Instead, it updates the regime and rebalances weights.
        The output is a metadata signal with current weights.
        """
        self._bars_since_rebalance += 1

        # Detect regime
        regime = "NORMAL"
        regime_confidence = 0.7
        if self.regime_detector is not None:
            regime_info = self.regime_detector.detect(features[:4] if len(features) >= 4 else features)
            regime = regime_info.get("regime", "NORMAL")
            regime_confidence = regime_info.get("regime_confidence", 0.7)

        # Rebalance if needed
        if self.should_rebalance():
            self.rebalance(regime)

        # The selector returns None -- it doesn't trade directly
        return None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def current_weights(self) -> dict[str, float]:
        """Current strategy allocation weights."""
        return self._strategy_weights.copy()

    def get_weight(self, strategy_name: str) -> float:
        """Get the current weight for a specific strategy."""
        return self._strategy_weights.get(strategy_name, 0.0)

    def strategy_performance_summary(self) -> dict[str, dict[str, Any]]:
        """Summary of strategy performance per regime."""
        summary: dict[str, dict[str, Any]] = {}

        for name in self._strategy_names:
            summary[name] = {
                "current_weight": self._strategy_weights.get(name, 0.0),
                "thompson_reliability": self._strategy_sampler.model_reliability(name),
            }

            for regime in RegimeDetector.MARKET_REGIMES:
                pnls = self._regime_performance.get(name, {}).get(regime, [])
                if pnls:
                    summary[name][f"regime_{regime}_mean_pnl"] = float(np.mean(pnls))
                    summary[name][f"regime_{regime}_n_trades"] = len(pnls)
                    summary[name][f"regime_{regime}_win_rate"] = (
                        float(np.mean([1 if p > 0 else 0 for p in pnls]))
                    )

        return summary
