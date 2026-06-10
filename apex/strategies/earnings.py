"""
APEX S8: ML Earnings Surprise + Options Strategy

Gradient-boosted earnings surprise predictor with options overlay.
TastyTrade only.

Pipeline:
1. Predict earnings surprise direction and magnitude using XGBoost
   trained on fundamental, sentiment, and options flow features.
2. Estimate expected move from IV and historical patterns.
3. If predicted surprise is larger than expected move, trade options
   (straddles, strangles, or directional spreads).
4. If predicted surprise is smaller, sell premium (iron condors).

Features used for prediction:
- Analyst estimate dispersion
- Pre-earnings IV vs historical vol
- Insider trading patterns
- Sector earnings momentum
- Options flow imbalance (put/call ratio)
- Social sentiment divergence
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from apex.ensemble.signal import ApexSignal
from apex.strategies.apex_strategy import ApexStrategy, ApexStrategyConfig

logger = logging.getLogger(__name__)


class EarningsConfig(ApexStrategyConfig, frozen=True):
    """Configuration for the Earnings Strategy."""

    strategy_name: str = "earnings"
    venue: str = "TASTYTRADE"
    min_edge: float = 0.03
    min_ensemble_score: float = 0.60

    # Earnings-specific parameters
    min_surprise_magnitude: float = 0.05  # Min |surprise| to trade (as fraction)
    max_days_before_earnings: int = 5      # Max days before earnings to enter
    min_days_before_earnings: int = 1      # Min days before earnings to enter
    expected_move_ratio_threshold: float = 1.2  # Predicted/expected ratio to trade
    max_premium_risk_pct: float = 0.05     # Max premium at risk per trade


class EarningsStrategy(ApexStrategy):
    """Gradient-boosted earnings surprise predictor with options overlay.

    Predicts whether the upcoming earnings release will produce a
    surprise larger or smaller than the market-implied move, and
    constructs appropriate options positions.
    """

    def __init__(self, config: EarningsConfig) -> None:
        super().__init__(config)

        self._min_surprise = config.min_surprise_magnitude
        self._max_days_before = config.max_days_before_earnings
        self._min_days_before = config.min_days_before_earnings
        self._move_ratio_threshold = config.expected_move_ratio_threshold
        self._max_premium_risk = config.max_premium_risk_pct

        # Earnings calendar: symbol -> earnings_date
        self._earnings_calendar: dict[str, datetime] = {}

        # Historical earnings data for model training
        self._earnings_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Earnings prediction
    # ------------------------------------------------------------------

    def predict_surprise(
        self,
        features: dict[str, float],
    ) -> dict[str, float]:
        """Predict earnings surprise direction and magnitude.

        Parameters
        ----------
        features : dict
            Earnings prediction features:

            - ``"analyst_dispersion"`` : float -- std dev of analyst estimates / mean
            - ``"pre_earnings_iv"`` : float -- pre-earnings implied volatility
            - ``"historical_vol"`` : float -- trailing realized vol
            - ``"iv_vol_ratio"`` : float -- IV / HV ratio
            - ``"insider_flow"`` : float -- net insider buying (-1 to 1)
            - ``"sector_momentum"`` : float -- sector earnings surprise trend
            - ``"put_call_ratio"`` : float -- options put/call ratio
            - ``"sentiment_divergence"`` : float -- news vs options sentiment gap

        Returns
        -------
        dict
            ``surprise_direction`` (+1 beat, -1 miss),
            ``surprise_magnitude`` (0-1),
            ``confidence`` (0-1),
            ``predicted_move`` (expected % move).
        """
        if "xgboost" in self.models and self.models["xgboost"] is not None:
            X = np.array(list(features.values())).reshape(1, -1)
            pred = self.models["xgboost"].predict(X)
            direction = 1 if pred[0] > 0 else -1
            magnitude = float(abs(pred[0]))
        else:
            # Heuristic fallback
            direction = self._heuristic_direction(features)
            magnitude = self._heuristic_magnitude(features)

        # Confidence based on feature quality
        dispersion = features.get("analyst_dispersion", 0.5)
        confidence = max(0.3, 1.0 - dispersion)

        # Predicted price move
        iv = features.get("pre_earnings_iv", 0.3)
        predicted_move = magnitude * iv * 0.1  # Rough scaling

        return {
            "surprise_direction": float(direction),
            "surprise_magnitude": magnitude,
            "confidence": confidence,
            "predicted_move": predicted_move,
        }

    @staticmethod
    def _heuristic_direction(features: dict[str, float]) -> int:
        """Heuristic earnings direction prediction."""
        score = 0.0
        score += features.get("insider_flow", 0.0) * 2.0
        score += features.get("sector_momentum", 0.0) * 1.5
        score -= (features.get("put_call_ratio", 1.0) - 1.0) * 1.0
        score += features.get("sentiment_divergence", 0.0) * 1.0
        return 1 if score > 0 else -1

    @staticmethod
    def _heuristic_magnitude(features: dict[str, float]) -> float:
        """Heuristic earnings magnitude prediction."""
        iv = features.get("pre_earnings_iv", 0.3)
        hv = features.get("historical_vol", 0.2)
        dispersion = features.get("analyst_dispersion", 0.5)

        # Higher IV/HV ratio = market expects bigger move
        iv_hv = iv / max(hv, 0.01)

        # Higher dispersion = more uncertainty = potentially bigger surprise
        magnitude = 0.3 * min(1.0, iv_hv / 2.0) + 0.3 * dispersion + 0.4 * 0.5
        return max(0.0, min(1.0, magnitude))

    # ------------------------------------------------------------------
    # Expected move computation
    # ------------------------------------------------------------------

    @staticmethod
    def expected_move_from_iv(
        stock_price: float,
        iv: float,
        dte: int,
    ) -> float:
        """Compute the expected move from implied volatility.

        Expected move = Stock Price * IV * sqrt(DTE / 365) * (1/sqrt(2*pi))

        This is the market-implied one-standard-deviation move.
        """
        T = max(dte / 365.0, 1e-6)
        one_sd_move = stock_price * iv * np.sqrt(T)
        # Expected absolute move (for normal distribution) = sigma * sqrt(2/pi)
        expected = one_sd_move * np.sqrt(2.0 / np.pi)
        return float(expected)

    @staticmethod
    def historical_earnings_move(
        past_moves: list[float],
    ) -> dict[str, float]:
        """Compute statistics from historical earnings moves.

        Parameters
        ----------
        past_moves : list[float]
            Historical post-earnings percentage moves.

        Returns
        -------
        dict
            ``mean_abs_move``, ``std_move``, ``beat_rate``.
        """
        if not past_moves:
            return {"mean_abs_move": 0.05, "std_move": 0.03, "beat_rate": 0.5}

        moves = np.array(past_moves)
        return {
            "mean_abs_move": float(np.mean(np.abs(moves))),
            "std_move": float(np.std(moves)),
            "beat_rate": float(np.mean(moves > 0)),
        }

    # ------------------------------------------------------------------
    # Trade structure selection
    # ------------------------------------------------------------------

    def select_trade_structure(
        self,
        prediction: dict[str, float],
        expected_move: float,
        stock_price: float,
    ) -> dict[str, Any]:
        """Select the optimal options structure for the earnings trade.

        Parameters
        ----------
        prediction : dict
            Output from :meth:`predict_surprise`.
        expected_move : float
            Market-implied expected move in dollars.
        stock_price : float
            Current stock price.

        Returns
        -------
        dict
            Trade structure: ``type``, ``direction``, ``strikes``,
            ``max_risk``, ``max_reward``.
        """
        predicted_move = prediction["predicted_move"] * stock_price
        direction = int(prediction["surprise_direction"])
        confidence = prediction["confidence"]
        ratio = predicted_move / max(expected_move, 1e-6)

        if ratio > self._move_ratio_threshold:
            # Predicted move > expected: buy premium
            if confidence > 0.7 and direction != 0:
                # High confidence directional: vertical spread
                if direction > 0:
                    return {
                        "type": "CALL_DEBIT_SPREAD",
                        "direction": "BULLISH",
                        "strikes": {
                            "long": stock_price,
                            "short": stock_price * 1.05,
                        },
                        "max_risk": expected_move * 0.5,
                        "max_reward": expected_move * 1.5,
                    }
                else:
                    return {
                        "type": "PUT_DEBIT_SPREAD",
                        "direction": "BEARISH",
                        "strikes": {
                            "long": stock_price,
                            "short": stock_price * 0.95,
                        },
                        "max_risk": expected_move * 0.5,
                        "max_reward": expected_move * 1.5,
                    }
            else:
                # Low confidence: straddle (non-directional)
                return {
                    "type": "LONG_STRADDLE",
                    "direction": "NEUTRAL",
                    "strikes": {"center": stock_price},
                    "max_risk": expected_move * 0.8,
                    "max_reward": predicted_move - expected_move,
                }
        else:
            # Predicted move < expected: sell premium
            return {
                "type": "IRON_CONDOR",
                "direction": "NEUTRAL",
                "strikes": {
                    "put_short": stock_price * 0.95,
                    "put_long": stock_price * 0.90,
                    "call_short": stock_price * 1.05,
                    "call_long": stock_price * 1.10,
                },
                "max_risk": expected_move * 0.3,
                "max_reward": expected_move * 0.2,
            }

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _generate_signal(self, features: np.ndarray) -> ApexSignal | None:
        """Generate an earnings signal.

        In production, would check the earnings calendar and only
        generate signals for stocks with upcoming earnings.
        """
        if len(features) < 5:
            return None

        price = float(features[3])
        if price <= 0:
            return None

        # In production, this would be driven by the earnings calendar
        # and real fundamental/options data. For now, the strategy
        # remains dormant unless manually triggered with features.

        # Regime check
        regime_info = {"regime": "NORMAL", "regime_confidence": 0.7}
        if self.regime_detector is not None:
            regime_info = self.regime_detector.detect(features[:4])

        # Only trade in calm/normal regimes (earnings are stock-specific)
        if regime_info.get("regime") in ("ELEVATED", "CRISIS"):
            return None

        # No signal without earnings data
        return None

    # ------------------------------------------------------------------
    # Earnings calendar
    # ------------------------------------------------------------------

    def register_earnings(
        self,
        symbol: str,
        earnings_date: datetime,
    ) -> None:
        """Register an upcoming earnings date."""
        self._earnings_calendar[symbol] = earnings_date

    def upcoming_earnings(
        self,
        within_days: int = 7,
    ) -> list[tuple[str, datetime, int]]:
        """List symbols with earnings within the specified window.

        Returns list of (symbol, earnings_date, days_until).
        """
        now = datetime.now(timezone.utc)
        upcoming: list[tuple[str, datetime, int]] = []

        for symbol, date in self._earnings_calendar.items():
            days_until = (date - now).days
            if 0 <= days_until <= within_days:
                upcoming.append((symbol, date, days_until))

        upcoming.sort(key=lambda x: x[2])
        return upcoming
