"""
APEX Ensemble Signal

Immutable dataclass representing a complete signal from the ensemble pipeline.
Carries model outputs, ensemble scores, recommended actions, and risk metrics
through the entire decision pipeline: models -> ensemble -> gate -> execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class ApexSignal:
    """Complete signal from the ensemble pipeline.

    A single ``ApexSignal`` encapsulates every piece of information needed
    to decide whether to trade, in what direction, and at what size.  It is
    produced by the ensemble layer and consumed by the :class:`TradeGate`
    and the execution engine.

    Attributes
    ----------
    market_id : str
        Unique identifier for the market / contract (e.g. a CLOB condition-id
        on Polymarket, a ticker-id on Kalshi, or an OCC symbol on TastyTrade).
    venue : str
        Trading venue -- one of ``"POLYMARKET"``, ``"KALSHI"``, ``"TASTYTRADE"``.
    timestamp : datetime
        UTC timestamp when the signal was generated.
    strategy : str
        Name of the strategy that produced this signal (e.g.
        ``"calibration_exploit"``, ``"bayesian_forecaster"``).

    xgb_probability : float
        Raw probability output from the XGBoost classifier (0-1).
    xgb_edge : float
        Predicted edge = ``xgb_probability - market_price``.
    lgbm_predicted_return : float
        Expected return predicted by the LightGBM regression model.
    tft_quantiles : dict[float, float]
        Quantile predictions from the Temporal Fusion Transformer.
        Keys are quantile levels (e.g. 0.1, 0.5, 0.9), values are the
        predicted return at that quantile.
    regime : str
        Current market regime from the HMM detector
        (``"CALM"``, ``"NORMAL"``, ``"ELEVATED"``, ``"CRISIS"``).
    regime_confidence : float
        Confidence of the regime classification (0-1).
    sentiment_score : float
        Aggregated sentiment score from NLP pipeline (-1 to +1).
    calibrated_edge : float
        Edge after isotonic calibration and bias correction.
    edge_ci_lower : float
        Lower bound of the edge confidence interval (e.g. 5th percentile).
    edge_ci_upper : float
        Upper bound of the edge confidence interval (e.g. 95th percentile).

    ensemble_score : float
        Combined score from the meta-learner (0-1).
    recommended_action : str
        Recommended trade action: ``"HOLD"``, ``"BUY"``, ``"SELL"``, or
        ``"CLOSE"``.
    recommended_size : float
        Fractional position size recommended by the PPO position manager (0-1).

    marginal_cvar : float
        Marginal contribution to portfolio CVaR if this trade is added.
    position_size_usd : float
        Dollar-denominated position size after all risk adjustments.
    spread_bps : float
        Current bid-ask spread in basis points at signal generation time.
    market_price : float
        Current mid-price of the market at signal generation time.
    """

    # -- Identifiers --------------------------------------------------------
    market_id: str
    venue: str  # POLYMARKET, KALSHI, TASTYTRADE
    timestamp: datetime
    strategy: str

    # -- Model outputs ------------------------------------------------------
    xgb_probability: float
    xgb_edge: float
    lgbm_predicted_return: float
    tft_quantiles: dict[float, float]
    regime: str
    regime_confidence: float
    sentiment_score: float
    calibrated_edge: float
    edge_ci_lower: float
    edge_ci_upper: float

    # -- Ensemble -----------------------------------------------------------
    ensemble_score: float
    recommended_action: str  # HOLD, BUY, SELL, CLOSE
    recommended_size: float  # From PPO position manager (0-1 fraction)

    # -- Risk ---------------------------------------------------------------
    marginal_cvar: float = 0.0
    position_size_usd: float = 0.0
    spread_bps: float = 0.0
    market_price: float = 0.0

    # -- Convenience --------------------------------------------------------

    @property
    def is_actionable(self) -> bool:
        """True if the signal recommends a non-HOLD action."""
        return self.recommended_action != "HOLD"

    @property
    def direction(self) -> int:
        """Returns +1 for BUY, -1 for SELL/CLOSE, 0 for HOLD."""
        if self.recommended_action == "BUY":
            return 1
        elif self.recommended_action in ("SELL", "CLOSE"):
            return -1
        return 0

    @property
    def edge_ci_width(self) -> float:
        """Width of the edge confidence interval."""
        return self.edge_ci_upper - self.edge_ci_lower

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON/logging."""
        return {
            "market_id": self.market_id,
            "venue": self.venue,
            "timestamp": self.timestamp.isoformat(),
            "strategy": self.strategy,
            "xgb_probability": self.xgb_probability,
            "xgb_edge": self.xgb_edge,
            "lgbm_predicted_return": self.lgbm_predicted_return,
            "tft_quantiles": self.tft_quantiles,
            "regime": self.regime,
            "regime_confidence": self.regime_confidence,
            "sentiment_score": self.sentiment_score,
            "calibrated_edge": self.calibrated_edge,
            "edge_ci_lower": self.edge_ci_lower,
            "edge_ci_upper": self.edge_ci_upper,
            "ensemble_score": self.ensemble_score,
            "recommended_action": self.recommended_action,
            "recommended_size": self.recommended_size,
            "marginal_cvar": self.marginal_cvar,
            "position_size_usd": self.position_size_usd,
            "spread_bps": self.spread_bps,
            "market_price": self.market_price,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ApexSignal:
        """Deserialize from a plain dict."""
        d = d.copy()
        if isinstance(d.get("timestamp"), str):
            d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        return cls(**d)

    @classmethod
    def hold(cls, market_id: str, venue: str, strategy: str) -> ApexSignal:
        """Factory for a no-action HOLD signal with zeroed model outputs."""
        return cls(
            market_id=market_id,
            venue=venue,
            timestamp=datetime.now(timezone.utc),
            strategy=strategy,
            xgb_probability=0.5,
            xgb_edge=0.0,
            lgbm_predicted_return=0.0,
            tft_quantiles={0.1: 0.0, 0.5: 0.0, 0.9: 0.0},
            regime="NORMAL",
            regime_confidence=0.5,
            sentiment_score=0.0,
            calibrated_edge=0.0,
            edge_ci_lower=0.0,
            edge_ci_upper=0.0,
            ensemble_score=0.0,
            recommended_action="HOLD",
            recommended_size=0.0,
        )
