"""
APEX Dynamic Capital Allocation

Dynamic allocation across Polymarket, Kalshi, and TastyTrade venues.
Rebalances hourly based on trailing performance metrics (Sharpe ratio,
hit rate, drawdown).

Initial allocation: 50% Polymarket / 30% Kalshi / 20% TastyTrade.
Rebalancing is constrained by minimum/maximum allocation bounds and
a smoothing factor to prevent excessive churn.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default initial allocations
_DEFAULT_ALLOCATIONS: dict[str, float] = {
    "polymarket": 0.50,
    "kalshi": 0.30,
    "tastytrade": 0.20,
}


class CapitalAllocator:
    """Dynamic allocation across trading venues.

    Parameters
    ----------
    total_capital : float
        Total available capital in USD.
    initial_allocations : dict[str, float] or None
        Initial allocation weights.  If None, uses defaults.
    min_allocation : float
        Minimum allocation per venue (prevents complete withdrawal).
    max_allocation : float
        Maximum allocation per venue (prevents over-concentration).
    smoothing : float
        Exponential smoothing factor for rebalancing (0-1).
        Higher values = more reactive to recent performance.
        Lower values = more stable allocations.
    rebalance_cooldown_hours : float
        Minimum hours between rebalances.
    """

    VENUES = ["polymarket", "kalshi", "tastytrade"]

    def __init__(
        self,
        total_capital: float = 5000.0,
        initial_allocations: dict[str, float] | None = None,
        min_allocation: float = 0.10,
        max_allocation: float = 0.60,
        smoothing: float = 0.3,
        rebalance_cooldown_hours: float = 1.0,
    ) -> None:
        self.total_capital = total_capital
        self.allocations = dict(initial_allocations or _DEFAULT_ALLOCATIONS)
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation
        self.smoothing = smoothing
        self.rebalance_cooldown_hours = rebalance_cooldown_hours

        # Deployed capital tracking
        self._deployed: dict[str, float] = {v: 0.0 for v in self.VENUES}
        self._last_rebalance: datetime | None = None
        self._rebalance_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core rebalancing
    # ------------------------------------------------------------------

    def rebalance(
        self,
        venue_performance: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Rebalance allocations based on trailing venue performance.

        Parameters
        ----------
        venue_performance : dict[str, dict]
            Performance stats per venue.  Each inner dict should contain:

            - ``"sharpe"`` : float -- trailing 14-day Sharpe ratio
            - ``"hit_rate"`` : float -- win rate (0-1)
            - ``"total_pnl"`` : float -- total PnL in USD
            - ``"n_trades"`` : int -- number of trades
            - ``"max_drawdown"`` : float -- max drawdown fraction (0-1)

        Returns
        -------
        dict[str, float]
            Updated allocation weights (sum to 1.0).
        """
        now = datetime.now(timezone.utc)

        # Cooldown check
        if self._last_rebalance is not None:
            hours_since = (now - self._last_rebalance).total_seconds() / 3600.0
            if hours_since < self.rebalance_cooldown_hours:
                logger.debug(
                    "Rebalance skipped: %.1f hours since last (cooldown=%.1f)",
                    hours_since,
                    self.rebalance_cooldown_hours,
                )
                return self.allocations.copy()

        # Compute performance scores
        scores = self._compute_scores(venue_performance)

        # Convert scores to target weights
        target_weights = self._scores_to_weights(scores)

        # Smooth toward target (exponential moving average)
        old_allocations = self.allocations.copy()
        new_allocations: dict[str, float] = {}
        for venue in self.VENUES:
            old_w = self.allocations.get(venue, 1.0 / len(self.VENUES))
            target_w = target_weights.get(venue, 1.0 / len(self.VENUES))
            new_w = old_w + self.smoothing * (target_w - old_w)
            new_allocations[venue] = new_w

        # Apply constraints and normalise
        self.allocations = self._constrain_and_normalise(new_allocations)

        # Record
        self._last_rebalance = now
        self._rebalance_history.append({
            "timestamp": now.isoformat(),
            "old": old_allocations,
            "new": self.allocations.copy(),
            "scores": scores,
            "target_weights": target_weights,
        })

        logger.info(
            "Capital rebalanced: %s (scores: %s)",
            {v: f"{w:.3f}" for v, w in self.allocations.items()},
            {v: f"{s:.3f}" for v, s in scores.items()},
        )

        return self.allocations.copy()

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def _compute_scores(
        self,
        venue_performance: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Compute a composite performance score per venue.

        Score = 0.5 * normalised_sharpe + 0.3 * hit_rate + 0.2 * (1 - drawdown)
        """
        scores: dict[str, float] = {}

        # Extract raw metrics
        sharpes: dict[str, float] = {}
        hit_rates: dict[str, float] = {}
        drawdowns: dict[str, float] = {}

        for venue in self.VENUES:
            perf = venue_performance.get(venue, {})
            sharpes[venue] = perf.get("sharpe", 0.0)
            hit_rates[venue] = perf.get("hit_rate", 0.5)
            drawdowns[venue] = perf.get("max_drawdown", 0.0)

        # Normalise Sharpe to [0, 1] range
        sharpe_values = list(sharpes.values())
        sharpe_min = min(sharpe_values)
        sharpe_max = max(sharpe_values)
        sharpe_range = sharpe_max - sharpe_min

        for venue in self.VENUES:
            # Normalised Sharpe
            if sharpe_range > 0:
                norm_sharpe = (sharpes[venue] - sharpe_min) / sharpe_range
            else:
                norm_sharpe = 0.5

            # Hit rate already in [0, 1]
            hr = hit_rates[venue]

            # Drawdown penalty (lower drawdown = higher score)
            dd_score = max(0.0, 1.0 - drawdowns[venue])

            # Composite score
            n_trades = venue_performance.get(venue, {}).get("n_trades", 0)
            if n_trades < 5:
                # Too few trades: use prior (equal allocation)
                scores[venue] = 0.5
            else:
                scores[venue] = 0.5 * norm_sharpe + 0.3 * hr + 0.2 * dd_score

        return scores

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def _scores_to_weights(
        self,
        scores: dict[str, float],
    ) -> dict[str, float]:
        """Convert performance scores to target allocation weights.

        Uses softmax-like transformation to avoid extreme allocations.
        """
        # Temperature-scaled softmax
        temperature = 2.0  # Higher = more uniform, lower = more extreme
        score_arr = np.array([scores.get(v, 0.5) for v in self.VENUES])

        # Softmax
        exp_scores = np.exp(score_arr / temperature)
        weights = exp_scores / exp_scores.sum()

        return {venue: float(w) for venue, w in zip(self.VENUES, weights)}

    # ------------------------------------------------------------------
    # Constraint enforcement
    # ------------------------------------------------------------------

    def _constrain_and_normalise(
        self,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Apply min/max constraints and renormalise to sum to 1.0."""
        # Clamp
        clamped = {
            v: max(self.min_allocation, min(self.max_allocation, w))
            for v, w in weights.items()
        }

        # Normalise
        total = sum(clamped.values())
        if total <= 0:
            equal = 1.0 / len(self.VENUES)
            return {v: equal for v in self.VENUES}

        normalised = {v: w / total for v, w in clamped.items()}

        # Second pass: ensure constraints still hold after normalisation
        # (normalising can push values below min or above max)
        for _ in range(5):
            violated = False
            for v in self.VENUES:
                if normalised[v] < self.min_allocation:
                    normalised[v] = self.min_allocation
                    violated = True
                if normalised[v] > self.max_allocation:
                    normalised[v] = self.max_allocation
                    violated = True
            if not violated:
                break
            total = sum(normalised.values())
            normalised = {v: w / total for v, w in normalised.items()}

        return normalised

    # ------------------------------------------------------------------
    # Capital queries
    # ------------------------------------------------------------------

    def get_available(self, venue: str) -> float:
        """Available capital for a venue (allocation - deployed).

        Parameters
        ----------
        venue : str
            Venue name (lowercase).

        Returns
        -------
        float
            Available USD capital.
        """
        venue_lower = venue.lower()
        alloc_pct = self.allocations.get(venue_lower, 0.0)
        allocated = self.total_capital * alloc_pct
        deployed = self._deployed.get(venue_lower, 0.0)
        return max(0.0, allocated - deployed)

    def get_allocated(self, venue: str) -> float:
        """Total allocated capital for a venue (before deployment)."""
        return self.total_capital * self.allocations.get(venue.lower(), 0.0)

    def deploy(self, venue: str, amount: float) -> bool:
        """Record capital deployment for a venue.

        Returns
        -------
        bool
            True if deployment succeeded (enough available capital).
        """
        venue_lower = venue.lower()
        available = self.get_available(venue_lower)
        if amount > available:
            logger.warning(
                "Cannot deploy $%.2f to %s: only $%.2f available",
                amount,
                venue_lower,
                available,
            )
            return False

        self._deployed[venue_lower] = self._deployed.get(venue_lower, 0.0) + amount
        return True

    def release(self, venue: str, amount: float) -> None:
        """Release deployed capital (position closed)."""
        venue_lower = venue.lower()
        self._deployed[venue_lower] = max(
            0.0, self._deployed.get(venue_lower, 0.0) - amount
        )

    def update_total_capital(self, new_total: float) -> None:
        """Update total capital (e.g. after PnL change)."""
        self.total_capital = new_total

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Current allocation summary."""
        result: dict[str, Any] = {
            "total_capital": self.total_capital,
            "allocations": self.allocations.copy(),
            "venues": {},
        }

        for venue in self.VENUES:
            allocated = self.get_allocated(venue)
            deployed = self._deployed.get(venue, 0.0)
            available = self.get_available(venue)
            result["venues"][venue] = {
                "allocation_pct": self.allocations.get(venue, 0.0),
                "allocated_usd": allocated,
                "deployed_usd": deployed,
                "available_usd": available,
                "utilisation_pct": deployed / allocated if allocated > 0 else 0.0,
            }

        return result

    @property
    def rebalance_history(self) -> list[dict[str, Any]]:
        """List of past rebalance events."""
        return list(self._rebalance_history)
