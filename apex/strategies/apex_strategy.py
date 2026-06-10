"""
APEX Base Strategy

Base strategy class extending NautilusTrader's Strategy.  Runs the full
ensemble pipeline on each bar: features -> models -> meta-learner ->
trade gate -> position sizer -> execution.

Subclasses implement specific signal generation logic by overriding
:meth:`_generate_signal`.  The base class handles the common pipeline:
model loading, feature building, risk checking, and order submission.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, TradeTick
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

from apex.config import ApexConfig, load_config
from apex.ensemble.meta_learner import MetaLearner
from apex.ensemble.signal import ApexSignal
from apex.ensemble.thompson_sampling import ThompsonSampler
from apex.ensemble.trade_gate import TradeGate
from apex.risk.circuit_breaker import CircuitBreaker
from apex.risk.position_sizer import PositionSizer
from apex.risk.regime_detector import RegimeDetector

logger = logging.getLogger(__name__)


class ApexStrategyConfig(StrategyConfig, frozen=True):
    """Configuration for an ApexStrategy instance.

    Parameters
    ----------
    strategy_name : str
        Human-readable strategy name (used in signal tagging).
    instrument_id : str
        NautilusTrader instrument identifier string.
    venue : str
        Venue name: POLYMARKET, KALSHI, or TASTYTRADE.
    bar_type : str
        Bar type string for subscription (e.g. "1-MINUTE-LAST").
    min_edge : float
        Minimum calibrated edge for the trade gate.
    min_ensemble_score : float
        Minimum ensemble score for the trade gate.
    max_spread_bps : float
        Maximum spread in basis points.
    """

    strategy_name: str = "apex_base"
    instrument_id: str = ""
    venue: str = "POLYMARKET"
    bar_type: str = "1-MINUTE-LAST"
    min_edge: float = 0.02
    min_ensemble_score: float = 0.6
    max_spread_bps: float = 500.0


class ApexStrategy(Strategy):
    """Base strategy class for APEX.

    Runs the full ensemble pipeline on each bar/tick.  Subclasses
    implement specific signal generation logic by overriding
    :meth:`_generate_signal`.

    Lifecycle:
        on_start -> load models, initialize pipeline components
        on_bar   -> features -> models -> ensemble -> gate -> order
        on_stop  -> persist state, cleanup
    """

    def __init__(self, config: ApexStrategyConfig) -> None:
        super().__init__(config)

        self.strategy_name = config.strategy_name
        self.venue = config.venue

        # Pipeline components (initialised in on_start)
        self.feature_builder: Any | None = None
        self.models: dict[str, Any] = {}
        self.meta_learner: MetaLearner | None = None
        self.thompson_sampler: ThompsonSampler | None = None
        self.trade_gate: TradeGate | None = None
        self.position_sizer: PositionSizer | None = None
        self.regime_detector: RegimeDetector | None = None
        self.circuit_breaker: CircuitBreaker | None = None

        # Configuration
        self._apex_config: ApexConfig | None = None
        self._min_edge = config.min_edge
        self._min_ensemble_score = config.min_ensemble_score
        self._max_spread_bps = config.max_spread_bps

        # State tracking
        self._signal_count: int = 0
        self._trade_count: int = 0
        self._bar_count: int = 0
        self._last_signal: ApexSignal | None = None
        self._strategy_stats: dict[str, float] = {
            "win_rate": 0.5,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "n_trades": 0,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        """Load models, initialize feature pipeline and risk components."""
        logger.info("Starting %s strategy on %s", self.strategy_name, self.venue)

        # Load configuration
        try:
            self._apex_config = load_config()
        except Exception:
            logger.warning("Could not load .env config, using defaults")
            self._apex_config = ApexConfig()

        # Initialize risk components
        self.regime_detector = RegimeDetector()
        self.circuit_breaker = CircuitBreaker()
        self.position_sizer = PositionSizer(self._apex_config.risk)
        self.trade_gate = TradeGate(
            config=self._apex_config,
            min_edge=self._min_edge,
            min_ensemble_score=self._min_ensemble_score,
            max_spread_bps=self._max_spread_bps,
        )

        # Initialize ensemble components
        self.thompson_sampler = ThompsonSampler(
            model_names=["xgboost", "lgbm", "tft", "sentiment"],
            decay=0.995,
        )

        # Subclass-specific initialisation
        self._on_start_strategy()

        logger.info(
            "%s strategy started: venue=%s, min_edge=%.3f, min_ensemble=%.3f",
            self.strategy_name,
            self.venue,
            self._min_edge,
            self._min_ensemble_score,
        )

    def _on_start_strategy(self) -> None:
        """Override in subclasses for strategy-specific initialisation."""
        pass

    def on_stop(self) -> None:
        """Persist state, cleanup resources."""
        logger.info(
            "%s stopping: %d signals, %d trades",
            self.strategy_name,
            self._signal_count,
            self._trade_count,
        )

    # ------------------------------------------------------------------
    # Data handlers
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        """Main loop: features -> models -> ensemble -> gate -> order.

        Parameters
        ----------
        bar : Bar
            NautilusTrader bar data.
        """
        self._bar_count += 1

        try:
            # Extract features from bar
            features = self._extract_features(bar)
            if features is None:
                return

            # Generate signal via the ensemble pipeline
            signal = self._generate_signal(features)
            if signal is None:
                return

            self._signal_count += 1
            self._last_signal = signal

            # Execute if approved by trade gate
            if signal.is_actionable:
                self._execute_signal(signal)

        except Exception:
            logger.exception(
                "%s error processing bar #%d",
                self.strategy_name,
                self._bar_count,
            )

    def on_trade_tick(self, tick: TradeTick) -> None:
        """React to individual trades for microstructure analysis.

        Override in subclasses that need tick-level data (e.g.
        MicrostructureStrategy, SmartMarketMakingStrategy).
        """
        pass

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, bar: Bar) -> np.ndarray | None:
        """Extract features from a bar.

        Default implementation creates a basic feature vector from bar
        OHLCV data.  Override in subclasses for richer feature sets.

        Returns None if feature extraction fails.
        """
        try:
            features = np.array([
                float(bar.open),
                float(bar.high),
                float(bar.low),
                float(bar.close),
                float(bar.volume),
                # Price change
                (float(bar.close) - float(bar.open)) / max(float(bar.open), 1e-8),
                # Range
                (float(bar.high) - float(bar.low)) / max(float(bar.open), 1e-8),
            ], dtype=np.float64)

            # Replace any NaN/Inf with 0
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            return features

        except Exception:
            logger.debug("Feature extraction failed for bar")
            return None

    # ------------------------------------------------------------------
    # Signal generation (override in subclasses)
    # ------------------------------------------------------------------

    def _generate_signal(self, features: np.ndarray) -> ApexSignal | None:
        """Run all models and ensemble pipeline.

        Override in subclasses for custom signal generation logic.
        The base implementation returns a HOLD signal.

        Parameters
        ----------
        features : np.ndarray
            Feature vector from :meth:`_extract_features`.

        Returns
        -------
        ApexSignal or None
            The generated signal, or None if no signal should be produced.
        """
        # Default: no signal (subclasses must override)
        return ApexSignal.hold(
            market_id="",
            venue=self.venue,
            strategy=self.strategy_name,
        )

    # ------------------------------------------------------------------
    # Signal execution
    # ------------------------------------------------------------------

    def _execute_signal(self, signal: ApexSignal) -> None:
        """Submit order to NautilusTrader based on the signal.

        Parameters
        ----------
        signal : ApexSignal
            An actionable signal (recommended_action != HOLD).
        """
        # Build portfolio state for gate evaluation
        portfolio_state = self._build_portfolio_state()

        # Gate check
        assert self.trade_gate is not None
        approved, reason = self.trade_gate.evaluate(signal, portfolio_state)

        if not approved:
            logger.debug(
                "%s signal rejected: %s",
                self.strategy_name,
                reason,
            )
            return

        # Compute final position size
        assert self.position_sizer is not None
        assert self.circuit_breaker is not None

        bankroll = portfolio_state.get("venue_capital_available", 0.0)
        cb_multiplier = self.circuit_breaker.sizing_multiplier()

        position_size = self.position_sizer.compute_size(
            signal=signal,
            strategy_stats=self._strategy_stats,
            regime=signal.regime,
            bankroll=bankroll,
            circuit_breaker_multiplier=cb_multiplier,
        )

        if position_size <= 0:
            logger.debug(
                "%s position size too small: $%.2f",
                self.strategy_name,
                position_size,
            )
            return

        # Log the trade decision
        logger.info(
            "%s EXECUTE: action=%s market=%s size=$%.2f edge=%.4f ensemble=%.4f",
            self.strategy_name,
            signal.recommended_action,
            signal.market_id,
            position_size,
            signal.calibrated_edge,
            signal.ensemble_score,
        )

        # Submit the order via NautilusTrader
        self._submit_order(signal, position_size)
        self._trade_count += 1

    def _submit_order(self, signal: ApexSignal, position_size: float) -> None:
        """Submit an order to the NautilusTrader execution engine.

        This is a placeholder that subclasses should extend with
        venue-specific order construction.
        """
        # In production, this would construct a NautilusTrader Order object
        # and call self.submit_order(order).
        # For now, log the intended order.
        logger.info(
            "ORDER: %s %s @ market %s, size=$%.2f, venue=%s",
            signal.recommended_action,
            signal.market_id,
            signal.market_price,
            position_size,
            signal.venue,
        )

    # ------------------------------------------------------------------
    # Portfolio state
    # ------------------------------------------------------------------

    def _build_portfolio_state(self) -> dict[str, Any]:
        """Build the portfolio state dict for gate evaluation."""
        cb_level = "GREEN"
        if self.circuit_breaker is not None:
            cb_level = self.circuit_breaker.level.value

        return {
            "circuit_breaker_level": cb_level,
            "portfolio_cvar": 0.0,
            "total_capital": 5000.0,
            "open_positions": len(self.cache.positions()) if hasattr(self, 'cache') and self.cache else 0,
            "max_positions": 50,
            "venue_capital_available": 1000.0,
            "correlated_exposure": 0.0,
        }

    # ------------------------------------------------------------------
    # Performance tracking
    # ------------------------------------------------------------------

    def record_trade_result(self, pnl: float) -> None:
        """Record a trade result for strategy stats and circuit breaker.

        Parameters
        ----------
        pnl : float
            PnL of the completed trade.
        """
        n = self._strategy_stats["n_trades"]
        if pnl > 0:
            old_avg = self._strategy_stats["avg_win"]
            wins = self._strategy_stats["win_rate"] * n if n > 0 else 0
            wins += 1
            self._strategy_stats["avg_win"] = (
                (old_avg * (wins - 1) + pnl) / wins if wins > 0 else pnl
            )
        else:
            old_avg = self._strategy_stats["avg_loss"]
            losses = (1 - self._strategy_stats["win_rate"]) * n if n > 0 else 0
            losses += 1
            self._strategy_stats["avg_loss"] = (
                (old_avg * (losses - 1) + abs(pnl)) / losses if losses > 0 else abs(pnl)
            )

        n += 1
        self._strategy_stats["n_trades"] = n

        # Recalculate win rate
        if n > 0:
            wins_total = self._strategy_stats["win_rate"] * (n - 1)
            if pnl > 0:
                wins_total += 1
            self._strategy_stats["win_rate"] = wins_total / n

        # Update circuit breaker
        if self.circuit_breaker is not None:
            self.circuit_breaker.update(
                equity=self._build_portfolio_state()["total_capital"],
                trade_result=pnl,
            )

        # Update Thompson sampler
        if self.thompson_sampler is not None:
            self.thompson_sampler.update_continuous(
                "xgboost", pnl, threshold=0.0
            )
