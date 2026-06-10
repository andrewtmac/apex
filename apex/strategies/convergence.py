"""
APEX S5: Resolution Convergence Strategy

Models how prediction market prices converge to 0 or 1 as the event
resolution date approaches.  Trades markets that converge too slowly
(underpricing) or too quickly (overreaction).

The key insight: prices in prediction markets should follow a specific
convergence curve based on time-to-resolution and the true probability.
Deviations from this curve create trading opportunities.

Convergence model:
    Expected price at time t:
        p_t = p_true + (p_market - p_true) * exp(-lambda * (T - t))

    Where lambda depends on information arrival rate and market efficiency.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

from apex.ensemble.signal import ApexSignal
from apex.strategies.apex_strategy import ApexStrategy, ApexStrategyConfig

logger = logging.getLogger(__name__)


class ConvergenceConfig(ApexStrategyConfig, frozen=True):
    """Configuration for the Resolution Convergence strategy."""

    strategy_name: str = "convergence"
    min_edge: float = 0.025
    min_ensemble_score: float = 0.55

    # Convergence-specific parameters
    convergence_rate: float = 0.1       # Base lambda (information arrival rate)
    min_hours_to_resolution: float = 2.0  # Don't trade within 2 hours of resolution
    max_days_to_resolution: float = 30.0  # Don't trade markets > 30 days out
    convergence_tolerance: float = 0.03   # Minimum deviation from expected path


class ConvergenceStrategy(ApexStrategy):
    """Trades markets that converge too slowly or too quickly.

    Maintains a model of expected convergence paths for each market
    and generates signals when the actual price deviates significantly.
    """

    def __init__(self, config: ConvergenceConfig) -> None:
        super().__init__(config)

        self._convergence_rate = config.convergence_rate
        self._min_hours = config.min_hours_to_resolution
        self._max_days = config.max_days_to_resolution
        self._tolerance = config.convergence_tolerance

        # Per-market convergence tracking
        # market_id -> {true_prob, resolution_time, price_history}
        self._market_models: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Convergence model
    # ------------------------------------------------------------------

    def expected_price(
        self,
        true_prob: float,
        current_price: float,
        hours_to_resolution: float,
        convergence_rate: float | None = None,
    ) -> float:
        """Compute the expected price at current time given convergence model.

        The price should converge from current_price toward true_prob
        as the resolution approaches.

        Parameters
        ----------
        true_prob : float
            Estimated true probability (0-1).
        current_price : float
            Current market price (0-1).
        hours_to_resolution : float
            Hours until the event resolves.
        convergence_rate : float or None
            Information arrival rate (lambda). Higher = faster convergence.

        Returns
        -------
        float
            Expected market price.
        """
        lam = convergence_rate or self._convergence_rate

        # Time decay factor: exp(-lambda * time_remaining)
        # As time_remaining -> 0, decay -> 0, price -> true_prob
        # As time_remaining -> inf, decay -> 1, price stays at current
        decay = math.exp(-lam * hours_to_resolution)

        expected = true_prob + (current_price - true_prob) * decay
        return max(0.01, min(0.99, expected))

    def convergence_speed(
        self,
        price_history: list[float],
        hours_between: float = 1.0,
    ) -> float:
        """Estimate the convergence speed from price history.

        Fits an exponential decay to the price trajectory to estimate
        how fast the market is converging.

        Parameters
        ----------
        price_history : list[float]
            Chronological price observations.
        hours_between : float
            Time between observations in hours.

        Returns
        -------
        float
            Estimated convergence rate (lambda).
        """
        if len(price_history) < 5:
            return self._convergence_rate

        prices = np.array(price_history)
        n = len(prices)

        # Estimate convergence target (where prices are heading)
        target = prices[-1]  # Simple: last price as target
        if prices[-1] > 0.9:
            target = 1.0
        elif prices[-1] < 0.1:
            target = 0.0

        # Distance from target over time
        distances = np.abs(prices - target) + 1e-8
        log_distances = np.log(distances)

        # Linear fit to log(distance) = -lambda * t + const
        times = np.arange(n) * hours_between
        if n < 3:
            return self._convergence_rate

        # Simple OLS for slope
        t_mean = np.mean(times)
        ld_mean = np.mean(log_distances)
        numerator = np.sum((times - t_mean) * (log_distances - ld_mean))
        denominator = np.sum((times - t_mean) ** 2)

        if abs(denominator) < 1e-12:
            return self._convergence_rate

        slope = numerator / denominator
        estimated_lambda = -slope  # lambda = -slope (decay rate)

        # Clamp to reasonable range
        return float(max(0.01, min(1.0, estimated_lambda)))

    def convergence_deviation(
        self,
        true_prob: float,
        current_price: float,
        hours_to_resolution: float,
    ) -> float:
        """How much the current price deviates from expected convergence path.

        Returns
        -------
        float
            Deviation (positive = price is too high, negative = too low).
        """
        expected = self.expected_price(true_prob, current_price, hours_to_resolution)

        # The deviation is how far the market is from where it "should" be
        # based on the convergence model
        deviation = current_price - expected

        return deviation

    # ------------------------------------------------------------------
    # Market tracking
    # ------------------------------------------------------------------

    def register_market(
        self,
        market_id: str,
        true_prob: float,
        resolution_time: datetime,
    ) -> None:
        """Register a market for convergence tracking.

        Parameters
        ----------
        market_id : str
            Unique market identifier.
        true_prob : float
            Estimated true probability.
        resolution_time : datetime
            When the event resolves (UTC).
        """
        self._market_models[market_id] = {
            "true_prob": true_prob,
            "resolution_time": resolution_time,
            "price_history": [],
            "estimated_lambda": self._convergence_rate,
        }

    def update_market(self, market_id: str, price: float) -> None:
        """Record a new price observation for a tracked market."""
        if market_id not in self._market_models:
            return

        self._market_models[market_id]["price_history"].append(price)

        # Re-estimate convergence speed periodically
        history = self._market_models[market_id]["price_history"]
        if len(history) % 10 == 0 and len(history) >= 10:
            self._market_models[market_id]["estimated_lambda"] = (
                self.convergence_speed(history)
            )

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _generate_signal(self, features: np.ndarray) -> ApexSignal | None:
        """Generate a convergence signal.

        Pipeline:
        1. Estimate true probability (from models)
        2. Compute expected convergence path
        3. Measure deviation from expected path
        4. If deviation > tolerance, generate signal
        """
        if len(features) < 5:
            return None

        market_price = float(features[3])
        if market_price <= 0.01 or market_price >= 0.99:
            return None

        # Estimate true probability from models
        true_prob = self._estimate_true_prob(features)

        # Simulate time-to-resolution (in production, comes from market metadata)
        # Use a proxy: higher volume = closer to resolution
        volume = float(features[4]) if len(features) > 4 else 100.0
        hours_to_resolution = max(2.0, 168.0 - volume * 0.1)  # Proxy

        # Check time bounds
        if hours_to_resolution < self._min_hours:
            return None
        if hours_to_resolution > self._max_days * 24:
            return None

        # Compute deviation
        deviation = self.convergence_deviation(
            true_prob, market_price, hours_to_resolution
        )

        # Edge: if the market is deviating from the convergence path,
        # we expect it to revert
        edge = -deviation  # Fade the deviation

        if abs(edge) < self._tolerance:
            return None

        # CI based on time to resolution (more uncertain further out)
        time_factor = min(1.0, hours_to_resolution / 100.0)
        ci_half = 0.02 + time_factor * 0.03

        action = "BUY" if edge > 0 else "SELL"
        ensemble_score = min(1.0, 0.5 + abs(edge) * 8.0)
        recommended_size = min(1.0, abs(edge) * 10.0 * (1.0 - time_factor * 0.5))

        # Regime
        regime_info = {"regime": "NORMAL", "regime_confidence": 0.7}
        if self.regime_detector is not None:
            regime_info = self.regime_detector.detect(features[:4])

        return ApexSignal(
            market_id=f"conv_{self._bar_count}",
            venue=self.venue,
            timestamp=datetime.now(timezone.utc),
            strategy=self.strategy_name,
            xgb_probability=true_prob,
            xgb_edge=true_prob - market_price,
            lgbm_predicted_return=edge * 0.8,
            tft_quantiles={0.1: edge - ci_half, 0.5: edge, 0.9: edge + ci_half},
            regime=regime_info.get("regime", "NORMAL"),
            regime_confidence=regime_info.get("regime_confidence", 0.7),
            sentiment_score=0.0,
            calibrated_edge=abs(edge),
            edge_ci_lower=abs(edge) - ci_half,
            edge_ci_upper=abs(edge) + ci_half,
            ensemble_score=ensemble_score,
            recommended_action=action,
            recommended_size=recommended_size,
            market_price=market_price,
        )

    def _estimate_true_prob(self, features: np.ndarray) -> float:
        """Estimate true probability from model outputs."""
        if "xgboost" in self.models and self.models["xgboost"] is not None:
            pred = self.models["xgboost"].predict(features.reshape(1, -1))
            return float(np.clip(pred[0], 0.01, 0.99))

        # Fallback: slight mean reversion from current price
        close = float(features[3]) if len(features) > 3 else 0.5
        return max(0.01, min(0.99, close * 0.9 + 0.05))
