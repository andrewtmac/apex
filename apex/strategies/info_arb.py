"""
APEX S3: Cross-Market Information Arbitrage Strategy

Detects information cascades across correlated markets and trades
secondary markets that haven't adjusted yet.

Key idea: when new information arrives, it moves the primary market
first.  Secondary markets (related contracts, same event on different
venues, correlated events) adjust with a lag.  This strategy detects
the primary move and trades the lagging secondaries.

Examples:
- Election winner market moves, VP market lags
- Polymarket "Will X happen?" moves, Kalshi equivalent lags
- Weather event contract moves, related economic impact contract lags
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

from apex.ensemble.signal import ApexSignal
from apex.strategies.apex_strategy import ApexStrategy, ApexStrategyConfig

logger = logging.getLogger(__name__)


class InfoArbConfig(ApexStrategyConfig, frozen=True):
    """Configuration for the Information Arbitrage strategy."""

    strategy_name: str = "info_arb"
    min_edge: float = 0.02
    min_ensemble_score: float = 0.55

    # Info arb parameters
    price_change_threshold: float = 0.03  # Min move to detect info cascade
    lag_window_bars: int = 10             # Window to detect lagging markets
    max_lag_bars: int = 30                # Max lag before signal expires
    correlation_threshold: float = 0.6    # Min correlation to consider related
    cascade_decay: float = 0.9           # Signal strength decay per bar of lag


class InfoArbStrategy(ApexStrategy):
    """Detects information cascades across correlated markets.

    Monitors a universe of correlated markets and detects when one
    market moves significantly (primary) while related markets
    (secondaries) haven't adjusted.
    """

    def __init__(self, config: InfoArbConfig) -> None:
        super().__init__(config)

        self._price_change_threshold = config.price_change_threshold
        self._lag_window = config.lag_window_bars
        self._max_lag = config.max_lag_bars
        self._corr_threshold = config.correlation_threshold
        self._cascade_decay = config.cascade_decay

        # Market universe: market_id -> price history
        self._market_prices: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=200)
        )

        # Market relationships: (primary, secondary) -> correlation
        self._relationships: dict[tuple[str, str], float] = {}

        # Active cascades: primary_id -> {timestamp, direction, magnitude}
        self._active_cascades: dict[str, dict[str, Any]] = {}

        # Cross-venue mappings: market_id -> list of equivalent market_ids
        self._venue_equivalents: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Market universe management
    # ------------------------------------------------------------------

    def register_relationship(
        self,
        primary: str,
        secondary: str,
        correlation: float,
    ) -> None:
        """Register a relationship between two markets.

        Parameters
        ----------
        primary : str
            Primary market ID (moves first).
        secondary : str
            Secondary market ID (lags).
        correlation : float
            Expected correlation (-1 to 1).
        """
        self._relationships[(primary, secondary)] = correlation
        logger.debug(
            "Registered relationship: %s -> %s (corr=%.3f)",
            primary,
            secondary,
            correlation,
        )

    def register_venue_equivalent(
        self,
        market_id: str,
        equivalents: list[str],
    ) -> None:
        """Register cross-venue equivalent markets.

        These are the same event traded on different venues.
        """
        self._venue_equivalents[market_id] = equivalents
        for eq in equivalents:
            if eq not in self._venue_equivalents:
                self._venue_equivalents[eq] = []
            if market_id not in self._venue_equivalents[eq]:
                self._venue_equivalents[eq].append(market_id)

    def update_price(self, market_id: str, price: float) -> None:
        """Record a price observation for a market."""
        self._market_prices[market_id].append(price)

    # ------------------------------------------------------------------
    # Cascade detection
    # ------------------------------------------------------------------

    def detect_cascade(
        self,
        market_id: str,
        current_price: float,
    ) -> list[dict[str, Any]]:
        """Detect if a price move on market_id creates cascade opportunities.

        Parameters
        ----------
        market_id : str
            The market that just moved.
        current_price : float
            Its current price.

        Returns
        -------
        list[dict]
            List of cascade opportunities:
            ``{secondary, expected_move, correlation, lag_bars, signal_strength}``.
        """
        prices = self._market_prices.get(market_id)
        if prices is None or len(prices) < 5:
            return []

        # Compute recent price change
        lookback = min(self._lag_window, len(prices) - 1)
        old_price = prices[-lookback - 1] if lookback < len(prices) else prices[0]
        price_change = current_price - old_price

        if abs(price_change) < self._price_change_threshold:
            return []

        # Record cascade
        self._active_cascades[market_id] = {
            "timestamp": datetime.now(timezone.utc),
            "direction": 1 if price_change > 0 else -1,
            "magnitude": abs(price_change),
            "bar_count": 0,
        }

        # Find related markets that haven't adjusted
        opportunities: list[dict[str, Any]] = []

        # Check registered relationships
        for (primary, secondary), correlation in self._relationships.items():
            if primary != market_id:
                continue
            if abs(correlation) < self._corr_threshold:
                continue

            secondary_prices = self._market_prices.get(secondary)
            if secondary_prices is None or len(secondary_prices) < 3:
                continue

            # Check if secondary has moved proportionally
            sec_lookback = min(self._lag_window, len(secondary_prices) - 1)
            sec_old = secondary_prices[-sec_lookback - 1] if sec_lookback < len(secondary_prices) else secondary_prices[0]
            sec_current = secondary_prices[-1]
            sec_change = sec_current - sec_old

            # Expected secondary move based on correlation
            expected_move = price_change * correlation
            actual_move = sec_change

            # Gap = expected - actual (positive = secondary is lagging)
            gap = expected_move - actual_move

            if abs(gap) < self._price_change_threshold * 0.5:
                continue  # Secondary has already adjusted

            signal_strength = min(1.0, abs(gap) / abs(expected_move)) if expected_move != 0 else 0.0

            opportunities.append({
                "secondary": secondary,
                "expected_move": expected_move,
                "actual_move": actual_move,
                "gap": gap,
                "correlation": correlation,
                "lag_bars": 0,
                "signal_strength": signal_strength,
            })

        # Check cross-venue equivalents
        for equiv_id in self._venue_equivalents.get(market_id, []):
            equiv_prices = self._market_prices.get(equiv_id)
            if equiv_prices is None or len(equiv_prices) < 3:
                continue

            equiv_current = equiv_prices[-1]
            price_diff = current_price - equiv_current

            if abs(price_diff) > self._price_change_threshold:
                opportunities.append({
                    "secondary": equiv_id,
                    "expected_move": price_change,
                    "actual_move": 0.0,
                    "gap": price_diff,
                    "correlation": 1.0,
                    "lag_bars": 0,
                    "signal_strength": min(1.0, abs(price_diff) * 10.0),
                    "type": "cross_venue",
                })

        return opportunities

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _generate_signal(self, features: np.ndarray) -> ApexSignal | None:
        """Generate an information arbitrage signal.

        Pipeline:
        1. Update market prices
        2. Check for cascade opportunities
        3. If cascade detected, generate signal on the lagging market
        """
        if len(features) < 5:
            return None

        market_price = float(features[3])
        if market_price <= 0.01 or market_price >= 0.99:
            return None

        market_id = f"primary_{self._bar_count}"
        self.update_price(market_id, market_price)

        # Detect cascades
        opportunities = self.detect_cascade(market_id, market_price)

        if not opportunities:
            # Age out old cascades
            self._age_cascades()
            return None

        # Take the strongest opportunity
        best = max(opportunities, key=lambda x: x["signal_strength"])

        gap = best["gap"]
        edge = abs(gap)
        direction = 1 if gap > 0 else -1

        if edge < self._min_edge:
            return None

        action = "BUY" if direction > 0 else "SELL"
        ensemble_score = min(1.0, 0.5 + best["signal_strength"] * 0.5)
        ci_half = 0.02 + (1.0 - best["signal_strength"]) * 0.03

        # Regime
        regime_info = {"regime": "NORMAL", "regime_confidence": 0.7}
        if self.regime_detector is not None:
            regime_info = self.regime_detector.detect(features[:4])

        return ApexSignal(
            market_id=best["secondary"],
            venue=self.venue,
            timestamp=datetime.now(timezone.utc),
            strategy=self.strategy_name,
            xgb_probability=0.5 + direction * edge,
            xgb_edge=edge * direction,
            lgbm_predicted_return=edge * 0.8,
            tft_quantiles={0.1: edge - ci_half, 0.5: edge, 0.9: edge + ci_half},
            regime=regime_info.get("regime", "NORMAL"),
            regime_confidence=regime_info.get("regime_confidence", 0.7),
            sentiment_score=0.0,
            calibrated_edge=edge,
            edge_ci_lower=edge - ci_half,
            edge_ci_upper=edge + ci_half,
            ensemble_score=ensemble_score,
            recommended_action=action,
            recommended_size=min(1.0, best["signal_strength"]),
            market_price=market_price,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _age_cascades(self) -> None:
        """Age and remove expired cascades."""
        expired = []
        for market_id, cascade in self._active_cascades.items():
            cascade["bar_count"] += 1
            cascade["magnitude"] *= self._cascade_decay
            if cascade["bar_count"] > self._max_lag:
                expired.append(market_id)

        for market_id in expired:
            del self._active_cascades[market_id]
