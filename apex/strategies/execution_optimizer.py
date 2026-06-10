"""
APEX S13: Multi-Venue Execution Optimizer

Contextual bandit for optimal order routing across venues.  Not a
trading strategy per se -- used by all strategies for execution
decisions.

Learns which venue provides the best execution quality (fill rate,
slippage, latency) for different types of orders, adjusting dynamically
via an upper confidence bound (UCB) bandit algorithm.

Features considered:
- Order size (small/medium/large)
- Market liquidity (spread, depth)
- Time of day
- Current venue load/latency
- Historical fill quality per venue
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

from apex.ensemble.signal import ApexSignal

logger = logging.getLogger(__name__)


class ExecutionOptimizer:
    """Contextual bandit for optimal order routing across venues.

    Maintains execution quality statistics per venue and uses
    UCB1 (Upper Confidence Bound) to balance exploration of new
    venues with exploitation of known-good venues.

    Parameters
    ----------
    venues : list[str]
        Available venues for routing.
    exploration_factor : float
        UCB exploration parameter (higher = more exploration).
    min_observations : int
        Minimum executions per venue before UCB kicks in.
    decay : float
        Exponential decay for historical execution quality (0-1).
        Lower = forgets faster, stays responsive.
    """

    VENUES = ["POLYMARKET", "KALSHI", "TASTYTRADE"]

    def __init__(
        self,
        venues: list[str] | None = None,
        exploration_factor: float = 1.5,
        min_observations: int = 10,
        decay: float = 0.995,
    ) -> None:
        self.venues = venues or self.VENUES
        self.exploration_factor = exploration_factor
        self.min_observations = min_observations
        self.decay = decay

        # Per-venue execution statistics
        self._fill_rates: dict[str, list[float]] = defaultdict(list)
        self._slippage_bps: dict[str, list[float]] = defaultdict(list)
        self._latencies_ms: dict[str, list[float]] = defaultdict(list)
        self._costs_bps: dict[str, list[float]] = defaultdict(list)

        # UCB state
        self._total_executions: int = 0
        self._venue_executions: dict[str, int] = defaultdict(int)
        self._venue_rewards: dict[str, float] = defaultdict(float)

        # Context-dependent stats: (venue, context_key) -> quality
        self._contextual_quality: dict[tuple[str, str], list[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Order routing
    # ------------------------------------------------------------------

    def route_order(
        self,
        signal: ApexSignal,
        venue_stats: dict[str, dict[str, float]] | None = None,
    ) -> str:
        """Choose optimal venue for order execution.

        Parameters
        ----------
        signal : ApexSignal
            The signal to route (contains venue hint, size, market_id).
        venue_stats : dict or None
            Real-time venue stats per venue:

            - ``"spread_bps"`` : float -- current spread
            - ``"depth_usd"`` : float -- available depth at best price
            - ``"latency_ms"`` : float -- current API latency
            - ``"is_available"`` : float -- 1.0 if venue is up, 0.0 if down

        Returns
        -------
        str
            Selected venue name.
        """
        venue_stats = venue_stats or {}

        # Filter to available venues
        available = [
            v for v in self.venues
            if venue_stats.get(v, {}).get("is_available", 1.0) > 0.5
        ]

        if not available:
            logger.warning("No venues available, defaulting to signal venue")
            return signal.venue

        # If signal specifies a venue and it's available, prefer it
        # (some strategies are venue-specific)
        if signal.venue in available:
            # Check if there's a better option via UCB
            scores = self._compute_ucb_scores(available, signal, venue_stats)
            best_venue = max(scores, key=scores.get)  # type: ignore[arg-type]

            # Only override if significantly better
            if best_venue != signal.venue:
                score_diff = scores[best_venue] - scores[signal.venue]
                if score_diff < 0.1:
                    return signal.venue  # Stick with signal venue

            return best_venue

        # No venue preference: use UCB
        scores = self._compute_ucb_scores(available, signal, venue_stats)
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def _compute_ucb_scores(
        self,
        venues: list[str],
        signal: ApexSignal,
        venue_stats: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Compute UCB1 scores for each venue.

        Score = mean_reward + exploration_factor * sqrt(ln(N) / n_i)

        Where reward is a composite of fill rate, slippage, and cost.
        """
        scores: dict[str, float] = {}
        N = max(1, self._total_executions)

        for venue in venues:
            n_i = max(1, self._venue_executions[venue])

            if n_i < self.min_observations:
                # Not enough data: give a high exploration bonus
                scores[venue] = 1.0 + self.exploration_factor * 2.0
                continue

            # Mean reward
            mean_reward = self._venue_rewards[venue] / n_i

            # Exploration bonus (UCB1)
            exploration = self.exploration_factor * math.sqrt(math.log(N) / n_i)

            # Context bonus: adjust for current conditions
            context_bonus = self._context_adjustment(venue, signal, venue_stats)

            scores[venue] = mean_reward + exploration + context_bonus

        return scores

    def _context_adjustment(
        self,
        venue: str,
        signal: ApexSignal,
        venue_stats: dict[str, dict[str, float]],
    ) -> float:
        """Adjust score based on current market context."""
        stats = venue_stats.get(venue, {})
        adjustment = 0.0

        # Prefer lower spread
        spread = stats.get("spread_bps", 200.0)
        adjustment -= spread / 10000.0  # Small penalty for wide spreads

        # Prefer deeper markets for large orders
        depth = stats.get("depth_usd", 1000.0)
        if signal.position_size_usd > 0 and depth > 0:
            size_ratio = signal.position_size_usd / depth
            if size_ratio > 0.5:
                adjustment -= 0.1  # Penalty for oversized orders

        # Prefer lower latency
        latency = stats.get("latency_ms", 100.0)
        adjustment -= latency / 10000.0  # Small penalty for high latency

        return adjustment

    # ------------------------------------------------------------------
    # Execution feedback
    # ------------------------------------------------------------------

    def record_execution(
        self,
        venue: str,
        fill_rate: float,
        slippage_bps: float,
        latency_ms: float,
        cost_bps: float = 0.0,
        context_key: str = "default",
    ) -> None:
        """Record execution quality after an order is filled.

        Parameters
        ----------
        venue : str
            Venue where the order was executed.
        fill_rate : float
            Fraction of the order that was filled (0-1).
        slippage_bps : float
            Slippage in basis points (positive = adverse).
        latency_ms : float
            Round-trip latency in milliseconds.
        cost_bps : float
            Total execution cost in basis points.
        context_key : str
            Context identifier for contextual learning.
        """
        self._fill_rates[venue].append(fill_rate)
        self._slippage_bps[venue].append(slippage_bps)
        self._latencies_ms[venue].append(latency_ms)
        self._costs_bps[venue].append(cost_bps)

        # Trim history
        max_history = 500
        for data in [self._fill_rates, self._slippage_bps, self._latencies_ms, self._costs_bps]:
            if len(data[venue]) > max_history:
                data[venue] = data[venue][-max_history:]

        # Compute reward: higher is better
        # Reward = fill_rate - normalised_slippage - normalised_cost
        reward = fill_rate - slippage_bps / 1000.0 - cost_bps / 1000.0
        reward = max(0.0, min(1.0, reward))

        # Update UCB state
        self._total_executions += 1
        self._venue_executions[venue] += 1
        self._venue_rewards[venue] += reward

        # Contextual quality
        self._contextual_quality[(venue, context_key)].append(reward)

        logger.debug(
            "Execution recorded: venue=%s fill=%.2f slippage=%.1fbps "
            "latency=%.0fms reward=%.3f",
            venue,
            fill_rate,
            slippage_bps,
            latency_ms,
            reward,
        )

    def decay_stats(self) -> None:
        """Apply decay to historical execution quality.

        Call periodically (e.g. daily) to keep the optimizer responsive.
        """
        for venue in self.venues:
            self._venue_rewards[venue] *= self.decay
            self._venue_executions[venue] = max(
                1, int(self._venue_executions[venue] * self.decay)
            )

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def venue_quality_summary(self) -> dict[str, dict[str, float]]:
        """Summary of execution quality per venue."""
        summary: dict[str, dict[str, float]] = {}

        for venue in self.venues:
            fills = self._fill_rates.get(venue, [])
            slips = self._slippage_bps.get(venue, [])
            lats = self._latencies_ms.get(venue, [])
            costs = self._costs_bps.get(venue, [])

            summary[venue] = {
                "n_executions": self._venue_executions[venue],
                "avg_fill_rate": float(np.mean(fills)) if fills else 0.0,
                "avg_slippage_bps": float(np.mean(slips)) if slips else 0.0,
                "avg_latency_ms": float(np.mean(lats)) if lats else 0.0,
                "avg_cost_bps": float(np.mean(costs)) if costs else 0.0,
                "ucb_reward": (
                    self._venue_rewards[venue] / max(1, self._venue_executions[venue])
                ),
            }

        return summary

    def best_venue_for_context(self, context_key: str) -> str | None:
        """Return the best venue for a specific context based on history."""
        best_venue = None
        best_quality = -float("inf")

        for venue in self.venues:
            quality_list = self._contextual_quality.get((venue, context_key), [])
            if len(quality_list) >= 5:
                avg = float(np.mean(quality_list[-20:]))  # Recent quality
                if avg > best_quality:
                    best_quality = avg
                    best_venue = venue

        return best_venue
