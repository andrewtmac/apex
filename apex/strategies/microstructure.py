"""
APEX S2: Market Microstructure Alpha Strategy

Detects informed order flow via VPIN (Volume-Synchronized Probability of
Informed Trading) and toxicity classification.  Trades in the direction
of detected informed flow before the price fully adjusts.

Key signals:
- VPIN spikes: sudden increase in informed trading probability
- Buy/sell volume imbalance
- Toxicity classification: distinguishes noise from informed flow
- Order flow acceleration

Works best on Polymarket and Kalshi where order flow data is available.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

from apex.ensemble.signal import ApexSignal
from apex.strategies.apex_strategy import ApexStrategy, ApexStrategyConfig

logger = logging.getLogger(__name__)


class MicrostructureConfig(ApexStrategyConfig, frozen=True):
    """Configuration for the Microstructure Alpha strategy."""

    strategy_name: str = "microstructure"
    min_edge: float = 0.015
    min_ensemble_score: float = 0.55

    # VPIN parameters
    vpin_bucket_size: float = 1000.0  # Volume per VPIN bucket (USD)
    vpin_n_buckets: int = 50          # Number of buckets for VPIN calculation
    vpin_alert_threshold: float = 0.7  # VPIN level that triggers alert

    # Toxicity parameters
    toxicity_threshold: float = 0.6    # Above this = informed flow
    min_volume_for_signal: float = 500.0  # Minimum volume to generate signal


class MicrostructureStrategy(ApexStrategy):
    """Detects informed order flow via VPIN and toxicity classification.

    Maintains a rolling VPIN calculation from trade data and classifies
    order flow toxicity.  When a VPIN spike coincides with directional
    flow, generates a signal in the direction of informed trading.
    """

    def __init__(self, config: MicrostructureConfig) -> None:
        super().__init__(config)

        # VPIN parameters
        self._bucket_size = config.vpin_bucket_size
        self._n_buckets = config.vpin_n_buckets
        self._vpin_threshold = config.vpin_alert_threshold
        self._toxicity_threshold = config.toxicity_threshold
        self._min_volume = config.min_volume_for_signal

        # VPIN state
        self._buy_volumes: deque[float] = deque(maxlen=config.vpin_n_buckets)
        self._sell_volumes: deque[float] = deque(maxlen=config.vpin_n_buckets)
        self._current_bucket_volume: float = 0.0
        self._current_bucket_buy: float = 0.0
        self._current_bucket_sell: float = 0.0

        # Trade flow tracking
        self._trade_buffer: deque[dict[str, Any]] = deque(maxlen=1000)
        self._vpin_history: deque[float] = deque(maxlen=200)
        self._last_price: float = 0.0

    # ------------------------------------------------------------------
    # VPIN Computation
    # ------------------------------------------------------------------

    def compute_vpin(
        self,
        trades: list[dict[str, Any]],
        bucket_size: float | None = None,
    ) -> float:
        """Volume-synchronized probability of informed trading.

        VPIN groups trades into volume buckets and measures the imbalance
        between buy and sell volume in each bucket.  High VPIN indicates
        a high probability of informed trading.

        Parameters
        ----------
        trades : list[dict]
            List of trade dicts, each with:

            - ``"price"`` : float
            - ``"volume"`` : float
            - ``"side"`` : str ("buy" or "sell"), or None (classified by tick rule)
        bucket_size : float or None
            Override for volume per bucket.

        Returns
        -------
        float
            VPIN value between 0 (no informed trading) and 1 (fully informed).
        """
        bucket_sz = bucket_size or self._bucket_size

        for trade in trades:
            price = trade.get("price", 0.0)
            volume = trade.get("volume", 0.0)
            side = trade.get("side", None)

            # Classify by tick rule if side not provided
            if side is None:
                if self._last_price > 0:
                    side = "buy" if price >= self._last_price else "sell"
                else:
                    side = "buy"  # Default for first trade
            self._last_price = price

            # Accumulate into current bucket
            remaining = volume
            while remaining > 0:
                space = bucket_sz - self._current_bucket_volume
                fill = min(remaining, space)

                if side == "buy":
                    self._current_bucket_buy += fill
                else:
                    self._current_bucket_sell += fill
                self._current_bucket_volume += fill
                remaining -= fill

                # Bucket complete
                if self._current_bucket_volume >= bucket_sz:
                    self._buy_volumes.append(self._current_bucket_buy)
                    self._sell_volumes.append(self._current_bucket_sell)
                    self._current_bucket_buy = 0.0
                    self._current_bucket_sell = 0.0
                    self._current_bucket_volume = 0.0

        # Compute VPIN
        if len(self._buy_volumes) < 5:
            return 0.0

        n = len(self._buy_volumes)
        total_volume = sum(
            self._buy_volumes[i] + self._sell_volumes[i] for i in range(n)
        )

        if total_volume <= 0:
            return 0.0

        order_imbalance = sum(
            abs(self._buy_volumes[i] - self._sell_volumes[i]) for i in range(n)
        )

        vpin = order_imbalance / total_volume
        self._vpin_history.append(vpin)

        return float(vpin)

    # ------------------------------------------------------------------
    # Toxicity Classification
    # ------------------------------------------------------------------

    def classify_toxicity(self, features: dict[str, float]) -> float:
        """Classify order flow toxicity.

        Combines multiple microstructure metrics into a single toxicity
        score.  Higher values indicate more informed (toxic) flow.

        Parameters
        ----------
        features : dict
            Microstructure features:

            - ``"vpin"`` : float -- current VPIN
            - ``"volume_imbalance"`` : float -- buy/sell volume ratio
            - ``"spread_bps"`` : float -- current spread
            - ``"trade_intensity"`` : float -- trades per time unit
            - ``"price_impact"`` : float -- price change per unit volume

        Returns
        -------
        float
            Toxicity score between 0 (noise trading) and 1 (fully informed).
        """
        vpin = features.get("vpin", 0.0)
        vol_imbalance = features.get("volume_imbalance", 0.5)
        spread = features.get("spread_bps", 0.0)
        intensity = features.get("trade_intensity", 0.0)
        impact = features.get("price_impact", 0.0)

        # VPIN component (most important)
        vpin_score = min(1.0, vpin / 0.8)

        # Volume imbalance component
        # Imbalance far from 0.5 = more informed
        imb_score = abs(vol_imbalance - 0.5) * 2.0

        # Spread widening = market makers detecting informed flow
        spread_score = min(1.0, spread / 500.0)

        # High trade intensity = urgency (informed traders want to trade fast)
        intensity_score = min(1.0, intensity / 10.0)

        # Price impact = trades moving the price = informed
        impact_score = min(1.0, abs(impact) * 20.0)

        # Weighted combination
        toxicity = (
            0.35 * vpin_score
            + 0.25 * imb_score
            + 0.15 * spread_score
            + 0.15 * intensity_score
            + 0.10 * impact_score
        )

        return float(min(1.0, max(0.0, toxicity)))

    # ------------------------------------------------------------------
    # Flow direction detection
    # ------------------------------------------------------------------

    def _detect_flow_direction(self) -> tuple[int, float]:
        """Detect the net direction of informed flow.

        Returns
        -------
        tuple[int, float]
            (direction, confidence) where direction is +1 (buy pressure)
            or -1 (sell pressure), and confidence is 0-1.
        """
        if len(self._buy_volumes) < 5:
            return 0, 0.0

        # Recent bucket imbalance
        recent_n = min(10, len(self._buy_volumes))
        recent_buys = [self._buy_volumes[-i] for i in range(1, recent_n + 1)]
        recent_sells = [self._sell_volumes[-i] for i in range(1, recent_n + 1)]

        total_buy = sum(recent_buys)
        total_sell = sum(recent_sells)
        total = total_buy + total_sell

        if total <= 0:
            return 0, 0.0

        buy_pct = total_buy / total
        direction = 1 if buy_pct > 0.5 else -1
        confidence = abs(buy_pct - 0.5) * 2.0  # 0 at balanced, 1 at fully one-sided

        return direction, confidence

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _generate_signal(self, features: np.ndarray) -> ApexSignal | None:
        """Generate a microstructure signal.

        Pipeline:
        1. Compute VPIN from recent trades
        2. Classify flow toxicity
        3. Detect flow direction
        4. If VPIN > threshold and toxicity > threshold, generate signal
        """
        if len(features) < 5:
            return None

        market_price = float(features[3])
        if market_price <= 0.01 or market_price >= 0.99:
            return None

        volume = float(features[4]) if len(features) > 4 else 0.0
        price_change = float(features[5]) if len(features) > 5 else 0.0

        # Simulate trade data from bar (in production, use actual trade ticks)
        trades = [{
            "price": market_price,
            "volume": volume,
            "side": "buy" if price_change > 0 else "sell",
        }]

        # Step 1: VPIN
        vpin = self.compute_vpin(trades)

        # Step 2: Flow direction
        direction, dir_confidence = self._detect_flow_direction()
        if direction == 0:
            return None

        # Step 3: Toxicity
        micro_features = {
            "vpin": vpin,
            "volume_imbalance": 0.5 + direction * dir_confidence * 0.5,
            "spread_bps": 100.0,  # Placeholder
            "trade_intensity": volume / max(self._bucket_size, 1.0),
            "price_impact": abs(price_change),
        }
        toxicity = self.classify_toxicity(micro_features)

        # Step 4: Signal conditions
        if vpin < self._vpin_threshold:
            return None
        if toxicity < self._toxicity_threshold:
            return None
        if volume < self._min_volume:
            return None

        # Edge estimate: proportional to VPIN and direction confidence
        edge = direction * vpin * dir_confidence * 0.05  # ~5% max edge
        edge_ci_half = 0.02 + (1.0 - dir_confidence) * 0.03

        action = "BUY" if direction > 0 else "SELL"
        ensemble_score = min(1.0, 0.4 + toxicity * 0.4 + dir_confidence * 0.2)
        recommended_size = min(1.0, toxicity * dir_confidence)

        # Regime
        regime_info = {"regime": "NORMAL", "regime_confidence": 0.7}
        if self.regime_detector is not None:
            regime_info = self.regime_detector.detect(features[:4])

        return ApexSignal(
            market_id=f"micro_{self._bar_count}",
            venue=self.venue,
            timestamp=datetime.now(timezone.utc),
            strategy=self.strategy_name,
            xgb_probability=0.5 + direction * 0.1,
            xgb_edge=edge,
            lgbm_predicted_return=edge * 0.7,
            tft_quantiles={
                0.1: edge - edge_ci_half,
                0.5: edge,
                0.9: edge + edge_ci_half,
            },
            regime=regime_info.get("regime", "NORMAL"),
            regime_confidence=regime_info.get("regime_confidence", 0.7),
            sentiment_score=0.0,
            calibrated_edge=abs(edge),
            edge_ci_lower=abs(edge) - edge_ci_half,
            edge_ci_upper=abs(edge) + edge_ci_half,
            ensemble_score=ensemble_score,
            recommended_action=action,
            recommended_size=recommended_size,
            market_price=market_price,
        )
