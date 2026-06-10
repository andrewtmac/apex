"""
APEX S6: Smart Market Making Strategy

Avellaneda-Stoikov market making with adverse selection defense.  Provides
liquidity to prediction markets while managing inventory risk and detecting
toxic (informed) order flow.

Key features:
- Optimal bid/ask quotes based on Avellaneda-Stoikov model
- VPIN-based toxicity detection to widen or withdraw quotes
- Inventory management with skew adjustment
- Spread widening during high-volatility regimes

Best suited for: liquid Polymarket and Kalshi markets in CALM/NORMAL regimes.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

from apex.ensemble.signal import ApexSignal
from apex.strategies.apex_strategy import ApexStrategy, ApexStrategyConfig

logger = logging.getLogger(__name__)


class SmartMMConfig(ApexStrategyConfig, frozen=True):
    """Configuration for the Smart Market Making strategy."""

    strategy_name: str = "smart_mm"
    min_edge: float = 0.01
    min_ensemble_score: float = 0.40

    # Market making parameters
    risk_aversion: float = 0.1         # Avellaneda-Stoikov gamma
    target_spread_bps: float = 200.0   # Target half-spread in bps
    max_inventory: float = 500.0       # Max inventory in USD
    inventory_skew_factor: float = 0.5  # How much to skew quotes
    volatility_window: int = 50        # Bars for volatility estimation
    toxicity_withdraw_threshold: float = 0.8  # Withdraw above this toxicity
    toxicity_widen_threshold: float = 0.5     # Widen spread above this


class SmartMarketMakingStrategy(ApexStrategy):
    """Avellaneda-Stoikov market making with adverse selection defense.

    Provides two-sided quotes (bid and ask) to prediction markets.
    The spread and skew are adjusted dynamically based on:
    1. Estimated volatility (wider spread in volatile markets)
    2. Current inventory (skew toward reducing inventory)
    3. Detected toxicity (widen or withdraw when informed flow detected)
    """

    def __init__(self, config: SmartMMConfig) -> None:
        super().__init__(config)

        self._gamma = config.risk_aversion
        self._target_half_spread = config.target_spread_bps / 10000.0
        self._max_inventory = config.max_inventory
        self._skew_factor = config.inventory_skew_factor
        self._vol_window = config.volatility_window
        self._toxicity_withdraw = config.toxicity_withdraw_threshold
        self._toxicity_widen = config.toxicity_widen_threshold

        # State
        self._inventory: float = 0.0  # Positive = long, negative = short
        self._mid_price_history: deque[float] = deque(maxlen=config.volatility_window)
        self._realized_vol: float = 0.01
        self._current_toxicity: float = 0.0

        # PnL tracking
        self._mm_pnl: float = 0.0
        self._total_quoted: int = 0
        self._total_fills: int = 0

    # ------------------------------------------------------------------
    # Avellaneda-Stoikov optimal quotes
    # ------------------------------------------------------------------

    def compute_optimal_quotes(
        self,
        mid_price: float,
        inventory: float,
        volatility: float,
        time_horizon: float = 1.0,
    ) -> tuple[float, float]:
        """Compute optimal bid and ask prices using Avellaneda-Stoikov.

        The model computes the reservation price (adjusted mid based on
        inventory) and the optimal spread.

        Parameters
        ----------
        mid_price : float
            Current mid-market price.
        inventory : float
            Current inventory (positive = long).
        volatility : float
            Estimated annualised volatility.
        time_horizon : float
            Time horizon in days.

        Returns
        -------
        tuple[float, float]
            (bid_price, ask_price).
        """
        # Daily volatility
        daily_vol = volatility / math.sqrt(252)
        sigma_sq = daily_vol ** 2

        # Reservation price: mid - gamma * sigma^2 * inventory * time_horizon
        # This skews quotes to reduce inventory
        reservation = mid_price - self._gamma * sigma_sq * inventory * time_horizon

        # Optimal spread: gamma * sigma^2 * time_horizon + 2/gamma * ln(1 + gamma/k)
        # Simplified: we use a base spread plus volatility adjustment
        base_spread = self._target_half_spread
        vol_spread = self._gamma * sigma_sq * time_horizon
        optimal_half_spread = max(base_spread, base_spread + vol_spread)

        # Apply inventory skew
        inventory_skew = self._skew_factor * inventory * sigma_sq
        bid_price = reservation - optimal_half_spread + inventory_skew
        ask_price = reservation + optimal_half_spread + inventory_skew

        # Clamp to valid range
        bid_price = max(0.01, min(mid_price - 0.001, bid_price))
        ask_price = min(0.99, max(mid_price + 0.001, ask_price))

        return bid_price, ask_price

    # ------------------------------------------------------------------
    # Volatility estimation
    # ------------------------------------------------------------------

    def _update_volatility(self, mid_price: float) -> float:
        """Update realized volatility estimate from price history."""
        self._mid_price_history.append(mid_price)

        if len(self._mid_price_history) < 5:
            return self._realized_vol

        prices = np.array(self._mid_price_history)
        returns = np.diff(np.log(np.maximum(prices, 1e-8)))

        if len(returns) < 2:
            return self._realized_vol

        # Annualised volatility from per-bar returns
        # Assuming 1-minute bars, ~390 bars/day for equities
        # For prediction markets, roughly 1440 bars/day
        bars_per_year = 1440 * 252
        self._realized_vol = float(np.std(returns) * math.sqrt(bars_per_year))

        # Floor volatility to prevent zero spreads
        self._realized_vol = max(0.01, self._realized_vol)

        return self._realized_vol

    # ------------------------------------------------------------------
    # Toxicity detection
    # ------------------------------------------------------------------

    def _assess_toxicity(self, features: np.ndarray) -> float:
        """Quick toxicity assessment from bar features.

        Uses price impact and volume patterns as proxies for informed flow.
        """
        if len(features) < 7:
            return 0.0

        price_change = abs(float(features[5]))  # Normalised price change
        volume = float(features[4])
        price_range = float(features[6])  # (high - low) / open

        # High price change + high volume = potential informed flow
        change_score = min(1.0, price_change * 20.0)  # Normalise
        vol_score = min(1.0, volume / (self._max_inventory * 2))
        range_score = min(1.0, price_range * 10.0)

        toxicity = 0.5 * change_score + 0.3 * vol_score + 0.2 * range_score
        self._current_toxicity = float(min(1.0, max(0.0, toxicity)))

        return self._current_toxicity

    # ------------------------------------------------------------------
    # Inventory management
    # ------------------------------------------------------------------

    def update_inventory(self, fill_side: str, fill_size: float) -> None:
        """Update inventory after a fill.

        Parameters
        ----------
        fill_side : str
            "bid" (we bought) or "ask" (we sold).
        fill_size : float
            Size of the fill in USD.
        """
        if fill_side == "bid":
            self._inventory += fill_size
        else:
            self._inventory -= fill_size

        self._total_fills += 1

    @property
    def inventory_utilization(self) -> float:
        """Current inventory as a fraction of max."""
        if self._max_inventory <= 0:
            return 0.0
        return abs(self._inventory) / self._max_inventory

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _generate_signal(self, features: np.ndarray) -> ApexSignal | None:
        """Generate market making quotes.

        The MM strategy generates a signal that represents the optimal
        bid/ask quotes.  The action is BUY at the bid or SELL at the ask,
        depending on which side has more opportunity.
        """
        if len(features) < 5:
            return None

        mid_price = float(features[3])
        if mid_price <= 0.01 or mid_price >= 0.99:
            return None

        # Update volatility
        vol = self._update_volatility(mid_price)

        # Assess toxicity
        toxicity = self._assess_toxicity(features)

        # Check toxicity thresholds
        if toxicity > self._toxicity_withdraw:
            logger.debug("Smart MM: withdrawing quotes (toxicity=%.3f)", toxicity)
            return None

        # Compute optimal quotes
        bid, ask = self.compute_optimal_quotes(
            mid_price=mid_price,
            inventory=self._inventory,
            volatility=vol,
        )

        # Widen spread if toxicity is elevated
        if toxicity > self._toxicity_widen:
            widen_factor = 1.0 + (toxicity - self._toxicity_widen) * 2.0
            half_spread = (ask - bid) / 2.0 * widen_factor
            center = (bid + ask) / 2.0
            bid = max(0.01, center - half_spread)
            ask = min(0.99, center + half_spread)

        # Determine preferred side based on inventory
        spread = ask - bid
        edge = spread / 2.0  # Expected profit per round trip

        if abs(self._inventory) > self._max_inventory * 0.8:
            # Need to reduce inventory
            if self._inventory > 0:
                action = "SELL"  # Unload long inventory
            else:
                action = "BUY"  # Cover short inventory
            recommended_size = min(1.0, abs(self._inventory) / self._max_inventory)
        else:
            # Normal MM: quote both sides
            action = "BUY" if self._inventory <= 0 else "SELL"
            recommended_size = 0.3  # Conservative size for MM

        # Check regime
        regime_info = {"regime": "NORMAL", "regime_confidence": 0.7}
        if self.regime_detector is not None:
            regime_info = self.regime_detector.detect(features[:4])

        # Don't market make in crisis
        if regime_info.get("regime") == "CRISIS":
            return None

        self._total_quoted += 1

        ensemble_score = min(1.0, 0.5 + edge * 15.0 - toxicity * 0.3)
        ci_half = 0.01 + toxicity * 0.02

        return ApexSignal(
            market_id=f"mm_{self._bar_count}",
            venue=self.venue,
            timestamp=datetime.now(timezone.utc),
            strategy=self.strategy_name,
            xgb_probability=mid_price,
            xgb_edge=edge,
            lgbm_predicted_return=edge * 0.5,
            tft_quantiles={0.1: -ci_half, 0.5: edge, 0.9: edge + ci_half},
            regime=regime_info.get("regime", "NORMAL"),
            regime_confidence=regime_info.get("regime_confidence", 0.7),
            sentiment_score=0.0,
            calibrated_edge=edge,
            edge_ci_lower=edge - ci_half,
            edge_ci_upper=edge + ci_half,
            ensemble_score=ensemble_score,
            recommended_action=action,
            recommended_size=recommended_size,
            spread_bps=spread * 10000.0,
            market_price=mid_price,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def mm_summary(self) -> dict[str, Any]:
        """Market making performance summary."""
        return {
            "inventory": self._inventory,
            "inventory_utilization": self.inventory_utilization,
            "realized_vol": self._realized_vol,
            "current_toxicity": self._current_toxicity,
            "total_quoted": self._total_quoted,
            "total_fills": self._total_fills,
            "mm_pnl": self._mm_pnl,
        }
