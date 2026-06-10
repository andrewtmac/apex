"""
APEX Bayesian Adaptive Kelly Position Sizer

Dynamic position sizing that replaces static quarter-Kelly with a
Bayesian-shrunk Kelly criterion adjusted for:

1. Sample uncertainty (Bayesian shrinkage when n < 50 trades)
2. Market regime (CALM -> full, CRISIS -> minimal)
3. Strategy graduation (capped sizing for immature strategies)
4. Edge confidence interval width
5. Circuit breaker multiplier

The sizing pipeline:
    Full Kelly f* = (p*b - q) / b
    -> Bayesian shrinkage toward prior
    -> Regime scaling
    -> Strategy graduation cap
    -> Confidence scaling
    -> Circuit breaker multiplier
    -> Final position size in USD
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from apex.config import RiskConfig, Regime

logger = logging.getLogger(__name__)

# Regime scaling factors: how much of the Kelly fraction to use per regime
_REGIME_SCALES: dict[str, float] = {
    "CALM": 1.0,
    "NORMAL": 0.8,
    "ELEVATED": 0.5,
    "CRISIS": 0.2,
}

# Strategy graduation: max Kelly fraction by number of completed trades
_GRADUATION_CAPS: list[tuple[int, float]] = [
    (10, 0.05),    # < 10 trades: max 5% Kelly
    (25, 0.10),    # 10-25 trades: max 10% Kelly
    (50, 0.20),    # 25-50 trades: max 20% Kelly
    (100, 0.30),   # 50-100 trades: max 30% Kelly
    # > 100 trades: governed by config
]

# Prior for Bayesian shrinkage
_PRIOR_WIN_RATE: float = 0.50  # Uninformative prior
_PRIOR_PAYOFF: float = 1.0    # Even-money prior
_PRIOR_STRENGTH: float = 10.0  # Equivalent to 10 prior observations


class PositionSizer:
    """Dynamic position sizing using Bayesian-shrunk Kelly.

    Parameters
    ----------
    config : RiskConfig
        Risk configuration with per-regime parameters.
    max_position_pct : float
        Hard maximum position size as a fraction of bankroll (default 15%).
    min_position_usd : float
        Minimum position size in USD -- below this, the trade is not worth
        the execution cost.
    """

    def __init__(
        self,
        config: RiskConfig,
        max_position_pct: float = 0.15,
        min_position_usd: float = 5.0,
    ) -> None:
        self.config = config
        self.max_position_pct = max_position_pct
        self.min_position_usd = min_position_usd

    # ------------------------------------------------------------------
    # Main sizing
    # ------------------------------------------------------------------

    def compute_size(
        self,
        signal: Any,
        strategy_stats: dict[str, float],
        regime: str,
        bankroll: float,
        circuit_breaker_multiplier: float = 1.0,
    ) -> float:
        """Compute position size in USD.

        Parameters
        ----------
        signal : ApexSignal
            The ensemble signal (uses ``calibrated_edge``,
            ``edge_ci_lower``, ``edge_ci_upper``, ``ensemble_score``,
            ``recommended_size``).
        strategy_stats : dict
            Strategy performance statistics:

            - ``win_rate`` : float -- historical win rate (0-1)
            - ``avg_win`` : float -- average winning trade PnL (absolute)
            - ``avg_loss`` : float -- average losing trade PnL (absolute, positive)
            - ``n_trades`` : int -- total completed trades
        regime : str
            Current market regime (``"CALM"``, ``"NORMAL"``, etc.).
        bankroll : float
            Total available capital.
        circuit_breaker_multiplier : float
            Multiplier from the circuit breaker (0.0 to 1.0).

        Returns
        -------
        float
            Position size in USD.  Zero means do not trade.
        """
        if bankroll <= 0:
            return 0.0

        if circuit_breaker_multiplier <= 0:
            return 0.0

        win_rate = strategy_stats.get("win_rate", 0.5)
        avg_win = strategy_stats.get("avg_win", 0.0)
        avg_loss = strategy_stats.get("avg_loss", 0.0)
        n_trades = int(strategy_stats.get("n_trades", 0))

        # Step 1: Compute full Kelly fraction
        f_kelly = self._full_kelly(win_rate, avg_win, avg_loss)

        # Step 2: Bayesian shrinkage
        f_bayes = self._bayesian_shrink(f_kelly, win_rate, avg_win, avg_loss, n_trades)

        # Step 3: Regime scaling
        regime_scale = _REGIME_SCALES.get(regime, 0.5)
        f_regime = f_bayes * regime_scale

        # Step 4: Strategy graduation cap
        grad_cap = self._graduation_cap(n_trades, regime)
        f_graduated = min(f_regime, grad_cap)

        # Step 5: Confidence scaling (use edge CI width and ensemble score)
        confidence = self._confidence_scale(signal)
        f_confident = f_graduated * confidence

        # Step 6: Circuit breaker multiplier
        f_final = f_confident * circuit_breaker_multiplier

        # Step 7: Hard clamp
        f_final = max(0.0, min(f_final, self.max_position_pct))

        # Convert to USD
        position_usd = bankroll * f_final

        # Floor check
        if position_usd < self.min_position_usd:
            return 0.0

        logger.debug(
            "PositionSizer: kelly=%.4f bayes=%.4f regime=%.4f grad=%.4f "
            "conf=%.4f cb=%.4f final=%.4f -> $%.2f",
            f_kelly,
            f_bayes,
            f_regime,
            f_graduated,
            f_confident,
            circuit_breaker_multiplier,
            f_final,
            position_usd,
        )

        return round(position_usd, 2)

    # ------------------------------------------------------------------
    # Kelly criterion
    # ------------------------------------------------------------------

    @staticmethod
    def _full_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Compute the full Kelly fraction.

        Kelly formula: f* = (p * b - q) / b
        where:
            p = win probability
            q = loss probability = 1 - p
            b = odds = avg_win / avg_loss (win/loss ratio)

        Returns 0.0 if the Kelly fraction is negative (negative edge).
        """
        if avg_loss <= 0 or avg_win <= 0:
            return 0.0

        p = max(0.0, min(1.0, win_rate))
        q = 1.0 - p
        b = avg_win / avg_loss

        kelly = (p * b - q) / b
        return max(0.0, kelly)

    # ------------------------------------------------------------------
    # Bayesian shrinkage
    # ------------------------------------------------------------------

    @staticmethod
    def _bayesian_shrink(
        f_kelly: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        n_trades: int,
    ) -> float:
        """Shrink Kelly estimate toward uninformative prior.

        With few trades, the observed win rate and payoff ratio are noisy.
        We blend toward a prior of 50% win rate and even payoffs using
        the effective sample size.

        Shrinkage weight: w = n / (n + prior_strength)
        Shrunk Kelly = w * f_kelly + (1 - w) * f_prior
        """
        # Prior Kelly fraction (50% win, 1:1 payoff -> f* = 0)
        f_prior = PositionSizer._full_kelly(
            _PRIOR_WIN_RATE, _PRIOR_PAYOFF, _PRIOR_PAYOFF
        )

        w = n_trades / (n_trades + _PRIOR_STRENGTH)
        return w * f_kelly + (1.0 - w) * f_prior

    # ------------------------------------------------------------------
    # Strategy graduation
    # ------------------------------------------------------------------

    def _graduation_cap(self, n_trades: int, regime: str) -> float:
        """Return the maximum Kelly fraction allowed for a strategy.

        Young strategies (few trades) are capped more aggressively.
        After graduating (> 100 trades), the config-based Kelly fraction
        for the current regime applies.
        """
        for threshold, cap in _GRADUATION_CAPS:
            if n_trades < threshold:
                return cap

        # Graduated: use config regime parameter
        try:
            regime_enum = Regime(regime)
            params = self.config.params_for(regime_enum)
            return params.kelly_fraction
        except (ValueError, KeyError):
            return 0.20  # Safe default

    # ------------------------------------------------------------------
    # Confidence scaling
    # ------------------------------------------------------------------

    @staticmethod
    def _confidence_scale(signal: Any) -> float:
        """Scale position size by signal confidence.

        Uses ensemble score and edge confidence interval width.
        Wider CI -> less confidence -> smaller size.
        """
        # Ensemble score: 0-1, directly usable as a scaling factor
        ensemble_score = getattr(signal, "ensemble_score", 0.5)

        # Edge CI width penalty: narrower CI = more confidence
        edge_ci_lower = getattr(signal, "edge_ci_lower", 0.0)
        edge_ci_upper = getattr(signal, "edge_ci_upper", 0.0)
        ci_width = max(0.0, edge_ci_upper - edge_ci_lower)

        # Transform CI width to a 0-1 scaling factor
        # Wider CI = lower confidence. CI of 0 -> 1.0, CI of 0.2 -> ~0.37
        ci_factor = math.exp(-5.0 * ci_width)

        # PPO recommended size (0-1)
        ppo_size = getattr(signal, "recommended_size", 0.5)

        # Blend: 40% ensemble, 30% CI confidence, 30% PPO
        confidence = 0.4 * ensemble_score + 0.3 * ci_factor + 0.3 * ppo_size

        return max(0.0, min(1.0, confidence))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def half_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Convenience: compute half-Kelly fraction."""
        return PositionSizer._full_kelly(win_rate, avg_win, avg_loss) * 0.5

    @staticmethod
    def quarter_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Convenience: compute quarter-Kelly fraction."""
        return PositionSizer._full_kelly(win_rate, avg_win, avg_loss) * 0.25
