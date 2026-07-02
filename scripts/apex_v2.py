#!/usr/bin/env python3
"""APEX V2: Multi-Agent Kalshi Trading System.

Mission: $1,000 -> $1,000,000 in 365 days.

Starts:
1. Weather strategy agent (NWS forecasts vs Kalshi weather markets)
2. Crypto strategy agent (BTC/ETH signals for Kalshi crypto)
3. Learner agent (strategy weight adjustment, regime detection)
4. Telegram reporter (hourly updates)
5. Dashboard (port 8080)
6. Paper trading loop with real Kalshi market scanning
"""

import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

from trade_narrator import narrate_close, narrate_entry_thesis, narrate_improvement

load_dotenv(Path(__file__).parent.parent / ".env")

import httpx
import numpy as np
import structlog

from weather_agent import WeatherAgent
from crypto_agent import CryptoAgent, CRYPTO_SERIES
from learner import LearnerAgent
from reporter import TelegramReporter
from telegram_commander import TelegramCommander

# New strategy agents (loaded lazily)
try:
    from macro_agent import MacroAgent
except ImportError:
    MacroAgent = None
try:
    from sports_agent import SportsAgent
except ImportError:
    SportsAgent = None
try:
    from events_agent import EventsAgent
except ImportError:
    EventsAgent = None

# Import existing components
sys.path.insert(0, str(Path(__file__).parent.parent))
from apex.risk.circuit_breaker import CircuitBreaker

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
)
logger = structlog.get_logger()

STATE_FILE = Path(__file__).parent.parent / "paper_state_v2.json"
DASHBOARD_HTML = Path(__file__).parent / "dashboard_v2.html"


# ---------------------------------------------------------------------------
# Paper Position
# ---------------------------------------------------------------------------

@dataclass
class PaperPosition:
    position_id: str
    market_id: str
    venue: str
    question: str
    direction: str  # BUY or SELL
    entry_price: float
    cost_basis: float
    shares: float
    entry_time: str
    strategy: str  # weather, crypto, macro, sports
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    status: str = "OPEN"
    exit_price: float | None = None
    exit_time: str | None = None
    realized_pnl: float = 0.0
    resolution: str | None = None
    edge_at_entry: float = 0.0
    peak_pnl_pct: float = 0.0
    stop_loss: float = -0.25
    take_profit: float = 0.20
    expires_at: str | None = None  # ISO timestamp when market closes/trading stops


# ---------------------------------------------------------------------------
# Kalshi Series Config
# ---------------------------------------------------------------------------

WEATHER_SERIES = [
    "KXHIGHNY", "KXHIGHCHI", "KXHIGHLA", "KXHIGHMIA", "KXHIGHDC",
    "KXHIGHHOU", "KXHIGHDAL", "KXHIGHDEN", "KXHIGHPHX", "KXHIGHATL",
    "KXHIGHSF", "KXHIGHBOS", "KXHIGHSEA", "KXHIGHAUS",
]

MACRO_SERIES = ["KXFED", "KXCPI", "KXJOBLESS", "KXGDP"]
SPORTS_SERIES = ["KXNBA", "KXMLB", "KXNHL", "KXNFL", "KXSOCCER", "KXMLSGAME", "KXNBAGAME"]
EVENTS_SERIES = ["KXTSLA", "KXAMZN", "KXMETA", "KXHURRICANE"]

ALL_SERIES = WEATHER_SERIES + CRYPTO_SERIES + MACRO_SERIES + SPORTS_SERIES + EVENTS_SERIES


# ---------------------------------------------------------------------------
# APEX V2 Trader
# ---------------------------------------------------------------------------

class ApexV2Trader:
    """Multi-agent autonomous Kalshi trader."""

    MAX_POSITIONS = 15
    MAX_DEPLOYED_PCT = 0.70
    MIN_POSITION_USD = 10.0
    CYCLE_SECONDS = 60  # Faster cycles to catch forecast lag
    RESOLUTION_CHECK_EVERY = 3  # Check every 3 cycles (3 min)
    STALE_POSITION_HOURS = 48  # 2 days (weather markets are daily)
    EXPIRY_URGENCY_HOURS = 4  # Check every cycle when position is within 4h of expiry
    MAX_DAILY_LOSS_PCT = 0.15
    MAX_DAILY_TRADES = 20
    COOLDOWN_AFTER_CONSEC_LOSSES = 3
    COOLDOWN_SECONDS = 7200  # 2 hours
    POSITION_POLL_SECONDS = 1  # Poll open position prices every 1s (Kalshi basic: 10 req/s)

    def __init__(self, bankroll: float = 1000.0):
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.positions: dict[str, PaperPosition] = {}
        self.closed_trades: list[dict] = []
        self.equity_history: list[dict] = []
        self.signals_generated = 0
        self.trades_executed = 0
        self.total_realized_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self._cycle = 0
        self._start_time = time.time()
        self._running = False
        self._daily_trades = 0
        self._daily_loss = 0.0
        self._last_trade_date = ""
        self._consecutive_losses = 0
        self._cooldown_until = 0.0

        # Agents
        self.weather = WeatherAgent()
        self.crypto = CryptoAgent()
        self.macro = MacroAgent() if MacroAgent else None
        self.sports = SportsAgent() if SportsAgent else None
        self.events = EventsAgent() if EventsAgent else None
        self.learner = LearnerAgent()
        self.reporter = TelegramReporter()
        self.commander = TelegramCommander()
        self.breaker = CircuitBreaker(initial_equity=bankroll, auto_recovery=True)

        # Per-event cooldown after stop-losses
        # Tracks {event_key: cooldown_expires_timestamp}
        # Prevents immediately re-entering a losing city/threshold
        self._event_cooldowns: dict[str, float] = {}
        self.EVENT_COOLDOWN_SECONDS = 300  # 5 min cooldown after stop-loss

        # Signal counters (rolling 10-min window for dashboard)
        self._signal_counts: dict[str, list[float]] = {}

        # Regime
        self.regime = "NORMAL"

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def deployed_capital(self) -> float:
        return sum(p.cost_basis for p in self.positions.values())

    @property
    def equity(self) -> float:
        return self.bankroll + self.deployed_capital + self.total_unrealized_pnl

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def load_state(self):
        if not STATE_FILE.exists():
            logger.info("v2.no_saved_state", msg="Starting fresh")
            return

        try:
            with open(STATE_FILE) as f:
                state = json.load(f)

            self.bankroll = state["bankroll"]
            self.trades_executed = state.get("trades_executed", 0)
            self.total_realized_pnl = state.get("total_realized_pnl", 0.0)
            self.wins = state.get("wins", 0)
            self.losses = state.get("losses", 0)
            self.signals_generated = state.get("signals_generated", 0)
            self.closed_trades = state.get("closed_trades", [])
            self.equity_history = state.get("equity_history", [])
            self._cycle = state.get("cycle", 0)
            self._daily_trades = state.get("daily_trades", 0)
            self._daily_loss = state.get("daily_loss", 0.0)
            self._last_trade_date = state.get("last_trade_date", "")
            self._consecutive_losses = state.get("consecutive_losses", 0)

            for p in state.get("positions", []):
                pos = PaperPosition(**p)
                self.positions[pos.market_id] = pos

            if "breaker" in state:
                self.breaker = CircuitBreaker.from_state_dict(state["breaker"])

            if "event_cooldowns" in state:
                self._event_cooldowns = state["event_cooldowns"]

            logger.info(
                "v2.state_restored",
                bankroll=f"${self.bankroll:.0f}",
                positions=len(self.positions),
                trades=self.trades_executed,
                pnl=f"${self.total_realized_pnl:+.2f}",
            )
        except Exception as e:
            logger.warning("v2.state_restore_failed", error=str(e))

    def save_state(self):
        state = {
            "bankroll": round(self.bankroll, 2),
            "trades_executed": self.trades_executed,
            "total_realized_pnl": round(self.total_realized_pnl, 2),
            "wins": self.wins,
            "losses": self.losses,
            "signals_generated": self.signals_generated,
            "cycle": self._cycle,
            "daily_trades": self._daily_trades,
            "daily_loss": round(self._daily_loss, 2),
            "last_trade_date": self._last_trade_date,
            "consecutive_losses": self._consecutive_losses,
            "positions": [asdict(p) for p in self.positions.values()],
            "closed_trades": self.closed_trades[-500:],
            "equity_history": self.equity_history[-5000:],
            "breaker": self.breaker.state_dict(),
            "event_cooldowns": self._event_cooldowns,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = STATE_FILE.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, default=str)
        tmp.replace(STATE_FILE)

    # ------------------------------------------------------------------
    # Kalshi market scanning
    # ------------------------------------------------------------------

    async def scan_kalshi_markets(self) -> list[dict]:
        """Scan all configured Kalshi series for tradeable markets."""
        active = []
        seen = set()

        async with httpx.AsyncClient(timeout=30) as client:
            for series in ALL_SERIES:
                try:
                    resp = await client.get(
                        "https://api.elections.kalshi.com/trade-api/v2/markets",
                        params={
                            "limit": 50,
                            "series_ticker": series,
                            "status": "open",
                        },
                    )
                    if resp.status_code != 200:
                        continue

                    for m in resp.json().get("markets", []):
                        ticker = m.get("ticker", "")
                        if not ticker or ticker in seen:
                            continue
                        if "MVE" in ticker:
                            continue

                        seen.add(ticker)

                        price = float(m.get("last_price_dollars", 0) or 0)
                        bid = float(m.get("yes_bid_dollars", 0) or 0)
                        ask = float(m.get("yes_ask_dollars", 0) or 0)

                        if price == 0 and bid > 0 and ask > 0:
                            price = (bid + ask) / 2
                        elif price == 0 and ask > 0:
                            price = ask
                        elif price == 0:
                            continue

                        vol_24h = float(m.get("volume_24h_fp", 0) or 0)
                        spread = round(ask - bid, 4) if ask > 0 and bid > 0 else 0.0

                        # Categorize
                        category = "other"
                        if series in WEATHER_SERIES:
                            category = "weather"
                        elif series in CRYPTO_SERIES:
                            category = "crypto"
                        elif series in MACRO_SERIES:
                            category = "macro"
                        elif series in SPORTS_SERIES:
                            category = "sports"
                        elif series in EVENTS_SERIES:
                            category = "events"

                        active.append({
                            "market_id": ticker,
                            "question": m.get("title", "")[:100],
                            "category": category,
                            "series": series,
                            "current_price": price,
                            "volume_24h": vol_24h,
                            "spread": spread,
                            "end_date": m.get("close_time") or m.get("expiration_time", ""),
                            "close_time": m.get("close_time", ""),
                            "venue": "kalshi",
                        })
                except Exception:
                    continue

        logger.debug("v2.kalshi_scan", n_markets=len(active))
        return active

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def _check_daily_limits(self) -> bool:
        """Check if daily limits allow a new trade."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Reset daily counters on new day
        if today != self._last_trade_date:
            self._daily_trades = 0
            self._daily_loss = 0.0
            self._last_trade_date = today

        if self._daily_trades >= self.MAX_DAILY_TRADES:
            return False

        if self._daily_loss >= self.bankroll * self.MAX_DAILY_LOSS_PCT:
            return False

        return True

    def _check_cooldown(self) -> bool:
        """Check if we're in a cooldown period."""
        if self._consecutive_losses >= self.COOLDOWN_AFTER_CONSEC_LOSSES:
            if time.time() < self._cooldown_until:
                return False
            # Cooldown expired
            self._consecutive_losses = 0
        return True

    def execute_trade(self, signal: dict) -> PaperPosition | None:
        """Execute a paper trade from a validated signal."""
        market_id = signal["market_id"]

        if market_id in self.positions:
            return None

        # Extract strategy early for conflict check
        strategy = signal.get("strategy", "other")

        # CONFLICT CHECK: Allow multiple positions on the same event
        # (e.g. same city, same day), but block conflicting open
        # positions. Conflicts are:
        #   - Opposite directions (BUY vs SELL) on same event
        #   - Overlapping range buckets on same event
        # Extract event key from market_id (e.g. KXHIGHCHI-26JUL01-B95.5
        # -> KXHIGHCHI_26JUL01)
        parts = market_id.split("-")
        strategy = signal.get("strategy", "other")

        # Events (hurricanes): each threshold is an independent bet.
        # Don't group them — ">5 hurricanes" and ">8 hurricanes" are
        # not conflicting. Only block if same exact market_id.
        if strategy == "events":
            for pos in self.positions.values():
                if pos.status != "OPEN":
                    continue
                if pos.market_id == market_id:
                    return None  # Same exact market, skip
                # Opposite directions on same series = conflict
                if pos.market_id.split("-")[0] == parts[0]:
                    if pos.direction != signal.get("direction", "BUY"):
                        logger.info("v2.conflict_skip",
                                    market_id=market_id,
                                    existing=pos.market_id,
                                    reason="opposite_directions")
                        return None
        elif len(parts) >= 2:
            event_key = f"{parts[0]}_{parts[1]}"
            for pos in self.positions.values():
                if pos.status != "OPEN":
                    continue
                pos_parts = pos.market_id.split("-")
                if len(pos_parts) >= 2:
                    pos_event = f"{pos_parts[0]}_{pos_parts[1]}"
                    if pos_event == event_key:
                        # Same event. Check for conflict:
                        # 1. Opposite directions = conflict
                        if pos.direction != signal.get("direction", "BUY"):
                            logger.info("v2.conflict_skip",
                                        market_id=market_id,
                                        existing=pos.market_id,
                                        reason="opposite_directions")
                            return None
                        # 2. Same event, same direction, different bucket
                        #    on a range market = conflict (ranges are exclusive)
                        if pos.market_id != market_id:
                            logger.info("v2.conflict_skip",
                                        market_id=market_id,
                                        existing=pos.market_id,
                                        reason="exclusive_ranges")
                            return None

        # PER-EVENT COOLDOWN: after a stop-loss, don't re-enter same event
        # for EVENT_COOLDOWN_SECONDS. Prevents chasing losses on the same
        # city/threshold (like Chicago: 5 trades, -$47).
        now = time.time()
        if parts and len(parts) >= 2:
            event_key = f"{parts[0]}_{parts[1]}"
            cooldown_until = self._event_cooldowns.get(event_key, 0)
            if now < cooldown_until:
                remaining = int(cooldown_until - now)
                logger.info("v2.event_cooldown",
                           event_key=event_key,
                           remaining_s=remaining)
                return None

        if len(self.positions) >= self.MAX_POSITIONS:
            return None

        if not self.breaker.can_open_new_position():
            return None

        if not self._check_daily_limits():
            return None

        if not self._check_cooldown():
            return None

        # Calculate position size
        learner_config = self.learner.get_strategy_config(strategy)
        max_size_pct = learner_config.get("max_size_pct", 0.05)

        size_pct = min(signal.get("size_pct", 0.05), max_size_pct)
        # Calculate position size
        logger.debug("v2.execute_trade_check",
                    market_id=market_id,
                    strategy=strategy,
                    positions=len(self.positions),
                    max=self.MAX_POSITIONS)

        # Per-city multiplier from learner (hot cities get bigger bets)
        if strategy == "weather":
            city = signal.get("city", "unknown")
            city_mult = self.learner.get_city_multiplier(strategy, city)
            size_pct *= city_mult

        cost = self.bankroll * size_pct
        if cost < self.MIN_POSITION_USD:
            return None

        if cost > self.bankroll:
            return None

        deployed = self.deployed_capital
        if (deployed + cost) > self.equity * self.MAX_DEPLOYED_PCT:
            return None

        entry_price = signal["market_price"]

        if signal["direction"] == "BUY":
            shares = cost / max(entry_price, 0.01)
        else:
            shares = cost / max(1.0 - entry_price, 0.01)

        # Weather-specific exit tuning based on entry quality
        base_sl = learner_config.get("stop_loss", -0.25)
        base_tp = learner_config.get("take_profit", 0.20)

        if strategy == "weather":
            confidence = signal.get("confidence", 0.5)
            abs_edge = abs(signal.get("edge", 0))
            dist_sigma = signal.get("threshold_distance_sigma", 2.0)

            if signal["direction"] == "SELL":
                # SELL trades: prices should decay toward 0
                # Wider take profit — let winners run
                base_tp = 0.30
                # Tighter stop — cut losers faster
                base_sl = -0.20
                # High-confidence entries get even wider TP
                if confidence >= 0.8 and abs_edge >= 0.15:
                    base_tp = 0.40
                    base_sl = -0.18
            else:
                # BUY trades: prices should rise toward 1
                # Standard TP, but tighter stop
                base_tp = 0.25
                base_sl = -0.18
                if confidence >= 0.8 and abs_edge >= 0.15:
                    base_tp = 0.35

            # Edge quality adjustment — high-edge trades get more room
            if abs_edge >= 0.20:
                base_tp *= 1.2
            elif abs_edge <= 0.07:
                base_tp *= 0.8
                base_sl *= 0.8  # Tighter for marginal trades

        pos = PaperPosition(
            position_id=uuid.uuid4().hex[:8],
            market_id=market_id,
            venue="kalshi",
            question=signal["question"],
            direction=signal["direction"],
            entry_price=entry_price,
            cost_basis=cost,
            shares=round(shares, 2),
            entry_time=datetime.now(timezone.utc).isoformat(),
            strategy=strategy,
            current_price=entry_price,
            # Direction-aware edge (July 2026 fix): agents emit the BUY-side
            # edge (model_prob - price) and derive direction from its sign, so
            # a SELL's edge arrived negative even though the short side's edge
            # is positive. Store the edge FOR THE SIDE WE TOOK — dashboards and
            # the trade narrator read this field at face value.
            edge_at_entry=(signal["edge"] if signal["direction"] == "BUY" else -signal["edge"]),
            stop_loss=base_sl,
            take_profit=base_tp,
            expires_at=signal.get("end_date"),
        )

        self.bankroll -= cost
        self.positions[market_id] = pos
        self.trades_executed += 1
        self._daily_trades += 1
        self.signals_generated += 1

        logger.info(
            "v2.trade_executed",
            id=pos.position_id,
            strategy=strategy,
            direction=pos.direction,
            entry=f"{pos.entry_price:.3f}",
            cost=f"${pos.cost_basis:.0f}",
            edge=f"{pos.edge_at_entry:+.4f}",
            open_positions=len(self.positions),
            bankroll=f"${self.bankroll:.0f}",
            question=pos.question[:50],
        )

        return pos

    # ------------------------------------------------------------------
    # Mark-to-market & exits
    # ------------------------------------------------------------------

    def mark_to_market(self, price_map: dict[str, float]):
        for market_id, pos in self.positions.items():
            if market_id in price_map:
                pos.current_price = price_map[market_id]

            if pos.direction == "BUY":
                pos.unrealized_pnl = round(
                    (pos.current_price - pos.entry_price) * pos.shares, 2
                )
            else:
                pos.unrealized_pnl = round(
                    (pos.entry_price - pos.current_price) * pos.shares, 2
                )

            roi = pos.unrealized_pnl / pos.cost_basis if pos.cost_basis > 0 else 0
            if roi > pos.peak_pnl_pct:
                pos.peak_pnl_pct = roi

    def check_exits(self) -> int:
        """Check all positions for exit conditions."""
        exit_ids = []

        for market_id, pos in self.positions.items():
            roi = pos.unrealized_pnl / pos.cost_basis if pos.cost_basis > 0 else 0

            # Take profit (strategy-specific)
            if roi >= pos.take_profit:
                self._close_position(pos, "TAKE_PROFIT")
                exit_ids.append(market_id)
                continue

            # Stop loss (strategy-specific)
            if roi <= pos.stop_loss:
                self._close_position(pos, "STOP_LOSS")
                exit_ids.append(market_id)
                continue

            # Trailing stop (strategy-aware)
            # Weather SELLs: tighter trail after lower peak
            # Weather BUYs: standard trail
            # Other strategies: standard trail
            trail_peak = 0.15
            trail_giveback = 0.08
            if pos.strategy == "weather" and pos.direction == "SELL":
                # SELLs should decay toward 0 — trail tighter
                trail_peak = 0.10
                trail_giveback = 0.06
            elif pos.strategy == "weather" and pos.direction == "BUY":
                trail_peak = 0.12
                trail_giveback = 0.07

            if pos.peak_pnl_pct >= trail_peak:
                giveback = pos.peak_pnl_pct - roi
                if giveback >= trail_giveback:
                    self._close_position(pos, "TRAILING_STOP")
                    exit_ids.append(market_id)
                    continue

        for mid in exit_ids:
            del self.positions[mid]

        return len(exit_ids)

    def _spawn_close_narration(self, trade_record: dict):
        """v3 flywheel: generate the close summary off the trading path."""
        async def _run():
            summary = await narrate_close(trade_record)
            if summary:
                trade_record["closeSummary"] = summary
                self.save_state()
        try:
            asyncio.get_running_loop().create_task(_run())
        except RuntimeError:
            pass  # no loop (e.g. tests) — narration is best-effort

    def _close_position(self, pos: PaperPosition, reason: str):
        """Close a position at current MTM price."""
        if pos.direction == "BUY":
            payout = pos.current_price * pos.shares
        else:
            payout = (1.0 - pos.current_price) * pos.shares

        pnl = round(payout - pos.cost_basis, 2)

        pos.realized_pnl = pnl
        pos.exit_price = pos.current_price
        pos.exit_time = datetime.now(timezone.utc).isoformat()
        pos.status = reason

        self.bankroll += payout
        self.total_realized_pnl += pnl

        if pnl >= 0:
            self.wins += 1
            self._consecutive_losses = 0
        else:
            self.losses += 1
            self._consecutive_losses += 1
            self._daily_loss += abs(pnl)

            if self._consecutive_losses >= self.COOLDOWN_AFTER_CONSEC_LOSSES:
                self._cooldown_until = time.time() + self.COOLDOWN_SECONDS

            # PER-EVENT COOLDOWN: set cooldown for this event on stop-loss
            # Prevents immediately re-entering a losing city/threshold
            if reason == "STOP_LOSS" and pos.strategy == "weather":
                parts = pos.market_id.split("-")
                if len(parts) >= 2:
                    event_key = f"{parts[0]}_{parts[1]}"
                    self._event_cooldowns[event_key] = (
                        time.time() + self.EVENT_COOLDOWN_SECONDS
                    )
                    logger.info("v2.event_cooldown_set",
                               event_key=event_key,
                               cooldown_s=self.EVENT_COOLDOWN_SECONDS)

        # Prune old cooldowns (keep dict small)
        if len(self._event_cooldowns) > 50:
            now = time.time()
            self._event_cooldowns = {
                k: v for k, v in self._event_cooldowns.items() if v > now
            }

        self.breaker.update(equity=self.equity, trade_result=pnl)

        # Record with learner
        trade_record = asdict(pos)
        trade_record["strategy"] = pos.strategy
        self.learner.record_trade(trade_record)
        self.learner.generate_insight(trade_record)

        record = asdict(pos)
        self.closed_trades.append(record)
        self._spawn_close_narration(record)

        logger.info(
            "v2.position_closed",
            id=pos.position_id,
            reason=reason,
            strategy=pos.strategy,
            pnl=f"${pnl:+.2f}",
            question=pos.question[:50],
        )

    # ------------------------------------------------------------------
    # Resolution monitoring
    # ------------------------------------------------------------------

    async def check_resolutions(self):
        """Check if any positions' markets have settled or finalized."""
        resolved_ids = []
        terminal_statuses = {"settled", "finalized", "closed"}

        async with httpx.AsyncClient(timeout=15) as client:
            for market_id, pos in list(self.positions.items()):
                try:
                    resp = await client.get(
                        f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_id}",
                    )
                    if resp.status_code != 200:
                        continue

                    data = resp.json()
                    market = data.get("market", data)

                    status = (market.get("status") or "").lower()
                    if status not in terminal_statuses:
                        # Not settled yet — but update expires_at if we have it
                        close_time = market.get("close_time") or market.get("expiration_time")
                        if close_time and not pos.expires_at:
                            pos.expires_at = close_time
                        continue

                    result = (market.get("result") or "").lower()
                    if result in ("yes", "y"):
                        outcome = "YES"
                    elif result in ("no", "n"):
                        outcome = "NO"
                    else:
                        # Terminal but no result — force close if past expected
                        expected_exp = (
                            market.get("expected_expiration_time")
                            or market.get("expiration_time")
                        )
                        if expected_exp:
                            try:
                                exp_dt = datetime.fromisoformat(
                                    expected_exp.replace("Z", "+00:00")
                                )
                                if datetime.now(timezone.utc) > exp_dt + timedelta(hours=6):
                                    self._close_position(pos, "CLOSED_NO_RESULT")
                                    resolved_ids.append(market_id)
                                    logger.warning(
                                        "v2.force_closed_no_result",
                                        market=market_id,
                                        status=status,
                                    )
                            except (ValueError, TypeError):
                                pass
                        continue

                    self._resolve_position(pos, outcome)
                    resolved_ids.append(market_id)

                except Exception as e:
                    logger.debug("v2.resolution_error", market=market_id, error=str(e))

        for mid in resolved_ids:
            del self.positions[mid]

    def _resolve_position(self, pos: PaperPosition, outcome: str):
        """Resolve a position against market outcome."""
        if pos.direction == "BUY":
            payout = pos.shares * (1.0 if outcome == "YES" else 0.0)
        else:
            payout = pos.shares * (1.0 if outcome == "NO" else 0.0)

        pnl = round(payout - pos.cost_basis, 2)

        pos.realized_pnl = pnl
        pos.exit_price = 1.0 if outcome == "YES" else 0.0
        pos.exit_time = datetime.now(timezone.utc).isoformat()
        pos.resolution = outcome
        pos.status = "RESOLVED_WIN" if pnl >= 0 else "RESOLVED_LOSS"

        self.bankroll += payout
        self.total_realized_pnl += pnl

        if pnl >= 0:
            self.wins += 1
            self._consecutive_losses = 0
        else:
            self.losses += 1
            self._consecutive_losses += 1
            self._daily_loss += abs(pnl)

        self.breaker.update(equity=self.equity, trade_result=pnl)

        trade_record = asdict(pos)
        trade_record["strategy"] = pos.strategy
        self.learner.record_trade(trade_record)

        record = asdict(pos)
        self.closed_trades.append(record)
        self._spawn_close_narration(record)

        logger.info(
            "v2.resolved",
            id=pos.position_id,
            outcome=outcome,
            pnl=f"${pnl:+.2f}",
            question=pos.question[:50],
        )

    async def close_stale(self):
        """Close positions open longer than STALE_POSITION_HOURS."""
        now = datetime.now(timezone.utc)
        stale_ids = []

        for market_id, pos in self.positions.items():
            entry = datetime.fromisoformat(pos.entry_time)
            hours = (now - entry).total_seconds() / 3600
            if hours < self.STALE_POSITION_HOURS:
                continue

            self._close_position(pos, "CLOSED_STALE")
            stale_ids.append(market_id)

        for mid in stale_ids:
            del self.positions[mid]

    # Kalshi basic tier: 10 requests/second
    KALSHI_RATE_LIMIT_RPS = 10
    # Minimum delay between batches to stay under rate limit
    _BATCH_INTERVAL = 1.0 / KALSHI_RATE_LIMIT_RPS  # 0.1s between each request

    async def position_monitor_loop(self):
        """Fast loop: poll open position prices every POSITION_POLL_SECONDS.

        Throttled to Kalshi basic tier rate limit (10 req/s). Positions are
        fetched in batches; if there are more open positions than the rate
        limit allows per second, they roll across multiple seconds.
        Updates mark-to-market and checks exits after each batch.
        """
        import httpx as _httpx

        # Persistent client to avoid connection overhead
        client = _httpx.AsyncClient(timeout=10)
        try:
            while self._running:
                await asyncio.sleep(self.POSITION_POLL_SECONDS)

                if not self.positions:
                    continue

                try:
                    snapshot = list(self.positions.items())
                    resolutions = []

                    # Process in batches of KALSHI_RATE_LIMIT_RPS to stay under limit
                    batch_size = self.KALSHI_RATE_LIMIT_RPS
                    for i in range(0, len(snapshot), batch_size):
                        batch = snapshot[i:i + batch_size]

                        async def _poll_one(market_id: str, pos):
                            try:
                                resp = await client.get(
                                    f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_id}",
                                )
                                if resp.status_code == 429:
                                    logger.warning("v2.rate_limited", market=market_id)
                                    return None
                                if resp.status_code != 200:
                                    return None

                                data = resp.json()
                                market = data.get("market", data)

                                # Update price
                                price = float(market.get("last_price_dollars", 0) or 0)
                                bid = float(market.get("yes_bid_dollars", 0) or 0)
                                ask = float(market.get("yes_ask_dollars", 0) or 0)

                                if price == 0 and bid > 0 and ask > 0:
                                    price = (bid + ask) / 2
                                elif price == 0 and ask > 0:
                                    price = ask

                                if price > 0:
                                    pos.current_price = price
                                    if pos.direction == "BUY":
                                        pos.unrealized_pnl = round(
                                            (pos.current_price - pos.entry_price) * pos.shares, 2
                                        )
                                    else:
                                        pos.unrealized_pnl = round(
                                            (pos.entry_price - pos.current_price) * pos.shares, 2
                                        )
                                    roi = pos.unrealized_pnl / pos.cost_basis if pos.cost_basis > 0 else 0
                                    if roi > pos.peak_pnl_pct:
                                        pos.peak_pnl_pct = roi

                                # Check for resolution
                                status = (market.get("status") or "").lower()
                                if status in ("settled", "finalized", "closed"):
                                    result = (market.get("result") or "").lower()
                                    if result in ("yes", "y"):
                                        return (market_id, "YES")
                                    elif result in ("no", "n"):
                                        return (market_id, "NO")
                            except Exception as e:
                                logger.debug("v2.position_poll_error",
                                            market=market_id, error=str(e))
                            return None

                        # Fire batch concurrently
                        results = await asyncio.gather(
                            *[_poll_one(mid, pos) for mid, pos in batch],
                            return_exceptions=True,
                        )

                        # Collect resolutions
                        for r in results:
                            if isinstance(r, tuple):
                                resolutions.append(r)

                        # Throttle between batches (skip delay after last batch)
                        if i + batch_size < len(snapshot):
                            await asyncio.sleep(1.0)

                    # Handle resolutions
                    for mid, outcome in resolutions:
                        pos = self.positions.get(mid)
                        if pos:
                            self._resolve_position(pos, outcome)
                            if mid in self.positions:
                                del self.positions[mid]

                    # Check exits after all price updates
                    exits = self.check_exits()
                    if exits > 0 or resolutions:
                        self.save_state()
                        if exits > 0:
                            logger.info("v2.position_monitor_exit", exits=exits,
                                        open=len(self.positions))

                except Exception as e:
                    logger.debug("v2.position_monitor_error", error=str(e))

        finally:
            await client.aclose()

    def has_urgent_positions(self) -> bool:
        """Check if any position is within EXPIRY_URGENCY_HOURS of expiration.
        Used to increase resolution check frequency."""
        now = datetime.now(timezone.utc)
        for pos in self.positions.values():
            if not pos.expires_at:
                continue
            try:
                exp_dt = datetime.fromisoformat(pos.expires_at.replace("Z", "+00:00"))
                hours_left = (exp_dt - now).total_seconds() / 3600
                if hours_left < self.EXPIRY_URGENCY_HOURS:
                    return True
            except (ValueError, TypeError):
                continue
        return False

    # ------------------------------------------------------------------
    # Main trading loop
    # ------------------------------------------------------------------

    async def trading_loop(self):
        """Main loop: scan -> analyze -> trade -> manage -> repeat."""
        while self._running:
            self._cycle += 1
            t0 = time.time()

            try:
                # 1. Scan markets
                markets = await self.scan_kalshi_markets()

                # 2. Mark-to-market
                price_map = {m["market_id"]: m["current_price"] for m in markets}
                self.mark_to_market(price_map)
                self.breaker.update(equity=self.equity)

                # Smart recovery: de-escalate breaker when bot is stable
                from apex.risk.circuit_breaker import BreakerLevel
                if self._cycle % 5 == 0:
                    bl = self.breaker.level.value
                    # Recovery if P&L is positive (original logic)
                    if self.total_realized_pnl > 0 and bl in ("ORANGE", "RED", "YELLOW"):
                        self.breaker.level = BreakerLevel.GREEN
                        self.breaker.consecutive_losses = 0
                        self.breaker.peak_equity = self.equity
                    # Time-based recovery: if breaker is elevated but
                    # no active losses, de-escalate and reset peak
                    # (the first-hour peak trap: peak=$1,242 keeps
                    # drawdown at 23% even though bot is stable at $952)
                    elif bl in ("RED", "ORANGE") and self._consecutive_losses == 0:
                        if bl == "RED":
                            self.breaker.level = BreakerLevel.ORANGE
                        elif bl == "ORANGE":
                            self.breaker.level = BreakerLevel.YELLOW
                        # Reset peak to current so drawdown recalculates
                        self.breaker.peak_equity = self.equity
                        logger.info("v2.breaker_recovery",
                                   from_level=bl,
                                   to_level=self.breaker.level.value,
                                   peak_reset=True)

                # 3. Check exits
                exits = self.check_exits()

                # 4. Snapshot equity
                self.equity_history.append({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "equity": round(self.equity, 2),
                })

                # 5. Update regime
                self.regime = self.learner.detect_regime(
                    self.equity_history, self.closed_trades
                )

                # 6. Generate signals from each strategy agent
                weather_markets = [m for m in markets if m["category"] == "weather"]
                crypto_markets = [m for m in markets if m["category"] == "crypto"]
                macro_markets = [m for m in markets if m["category"] == "macro"]
                sports_markets = [m for m in markets if m["category"] == "sports"]
                events_markets = [m for m in markets if m["category"] == "events"]

                all_signals = []

                # Weather signals
                if weather_markets:
                    w_signals = await self.weather.generate_signals(weather_markets)
                    for s in w_signals:
                        s["size_usd"] = self.bankroll * s.get("size_pct", 0.05)
                    all_signals.extend(w_signals)
                    for _ in w_signals:
                        self.record_signal("weather")

                # Crypto signals
                if crypto_markets:
                    c_signals = await self.crypto.generate_signals(crypto_markets)
                    for s in c_signals:
                        s["size_usd"] = self.bankroll * s.get("size_pct", 0.05)
                    all_signals.extend(c_signals)
                    for _ in c_signals:
                        self.record_signal("crypto")

                # Macro signals (CPI, FED, GDP)
                if macro_markets and self.macro:
                    try:
                        m_signals = await self.macro.generate_signals(macro_markets)
                        for s in m_signals:
                            s["size_usd"] = self.bankroll * s.get("size_pct", 0.05)
                        all_signals.extend(m_signals)
                        for _ in m_signals:
                            self.record_signal("finance")
                    except Exception as e:
                        logger.warning("v2.macro_error", error=str(e))

                # Sports signals (NBA, MLB, NHL arb)
                if sports_markets and self.sports:
                    try:
                        sp_signals = await self.sports.generate_signals(sports_markets)
                        for s in sp_signals:
                            s["size_usd"] = self.bankroll * s.get("size_pct", 0.05)
                        all_signals.extend(sp_signals)
                        for _ in sp_signals:
                            self.record_signal("sports")
                    except Exception as e:
                        logger.warning("v2.sports_error", error=str(e))

                # Events signals (TSLA, hurricanes)
                if events_markets and self.events:
                    try:
                        e_signals = await self.events.generate_signals(events_markets)
                        for s in e_signals:
                            s["size_usd"] = self.bankroll * s.get("size_pct", 0.05)
                        all_signals.extend(e_signals)
                        for _ in e_signals:
                            self.record_signal("events")
                    except Exception as e:
                        logger.warning("v2.events_error", error=str(e))

                # 7. Filter by strategy minimum edge from learner
                filtered_signals = []
                for s in all_signals:
                    strategy = s.get("strategy", "other")
                    min_edge = self.learner.get_min_edge(strategy)
                    if abs(s["edge"]) >= min_edge:
                        filtered_signals.append(s)

                # 8. Sort by edge and execute
                filtered_signals.sort(key=lambda s: abs(s["edge"]), reverse=True)

                new_trades = 0
                for signal in filtered_signals:
                    pos = self.execute_trade(signal)
                    if pos is not None:
                        new_trades += 1

                # 9. Resolution check (periodic, or every cycle when near expiry)
                should_check_resolution = (
                    self._cycle % self.RESOLUTION_CHECK_EVERY == 0
                    or self.has_urgent_positions()
                )
                if should_check_resolution:
                    await self.check_resolutions()
                    await self.close_stale()

                # 10. Save state
                self.save_state()

                # 11. Cycle summary
                elapsed = time.time() - t0
                total_closed = self.wins + self.losses
                logger.info(
                    "v2.cycle",
                    cycle=self._cycle,
                    markets=len(markets),
                    signals=len(filtered_signals),
                    new_trades=new_trades,
                    exits=exits,
                    open=len(self.positions),
                    deployed=f"${self.deployed_capital:.0f}",
                    cash=f"${self.bankroll:.0f}",
                    equity=f"${self.equity:.0f}",
                    pnl=f"${self.total_realized_pnl:+.0f}",
                    record=f"{self.wins}W-{self.losses}L" if total_closed > 0 else "0-0",
                    breaker=self.breaker.level.value,
                    regime=self.regime,
                    elapsed=f"{elapsed:.1f}s",
                )

            except Exception as e:
                logger.error("v2.loop_error", error=str(e))

            await asyncio.sleep(self.CYCLE_SECONDS)

    # ------------------------------------------------------------------
    # Hourly reporting
    # ------------------------------------------------------------------

    async def reporting_loop(self):
        """Send hourly updates to Telegram."""
        while self._running:
            await asyncio.sleep(60)  # Check every minute

            if not self.reporter.should_send_hourly():
                continue

            try:
                state = {
                    "equity": round(self.equity, 2),
                    "initial_bankroll": self.initial_bankroll,
                    "bankroll": round(self.bankroll, 2),
                    "deployed": round(self.deployed_capital, 2),
                    "unrealized_pnl": round(self.total_unrealized_pnl, 2),
                    "realized_pnl": round(self.total_realized_pnl, 2),
                    "wins": self.wins,
                    "losses": self.losses,
                    "trades_executed": self.trades_executed,
                    "signals_generated": self.signals_generated,
                    "breaker": self.breaker.level.value,
                    "drawdown_pct": round(self.breaker.drawdown_pct, 2),
                    "cycle": self._cycle,
                    "regime": self.regime,
                    "positions": [asdict(p) for p in self.positions.values()],
                    "strategy_weights": self.learner.weights,
                    "learnings_summary": self.learner.build_daily_summary()[:150],
                }

                text = self.reporter.build_hourly_update(state)
                await self.reporter.send(text)
                logger.info("v2.hourly_report_sent")

            except Exception as e:
                logger.warning("v2.report_failed", error=str(e))

    # ------------------------------------------------------------------
    # Learning loop (daily)
    # ------------------------------------------------------------------

    async def learning_loop(self):
        """Run daily learning updates."""
        while self._running:
            # Wait until midnight UTC
            now = datetime.now(timezone.utc)
            tomorrow = (now + timedelta(days=1)).replace(
                hour=0, minute=5, second=0, microsecond=0
            )
            wait = (tomorrow - now).total_seconds()
            await asyncio.sleep(wait)

            if not self._running:
                break

            try:
                summary = await self.learner.daily_update()
                await self.reporter.send(f"*📝 Daily Learning Update*\n\n{summary}")
            except Exception as e:
                logger.warning("v2.learning_failed", error=str(e))

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    # Signal count rolling window
    SIGNAL_WINDOW_SECONDS = 600  # 10 minutes

    def record_signal(self, category: str):
        """Record a signal for the rolling signal counter."""
        now = time.time()
        if category not in self._signal_counts:
            self._signal_counts[category] = []
        self._signal_counts[category].append(now)
        # Prune old entries
        cutoff = now - self.SIGNAL_WINDOW_SECONDS
        self._signal_counts[category] = [
            t for t in self._signal_counts[category] if t > cutoff
        ]

    def get_signal_count(self, category: str) -> int:
        """Get rolling 10-min signal count for a category."""
        now = time.time()
        cutoff = now - self.SIGNAL_WINDOW_SECONDS
        entries = self._signal_counts.get(category, [])
        return sum(1 for t in entries if t > cutoff)

    async def run_dashboard(self):
        """Start the investor dashboard on port 8080."""
        try:
            import uvicorn
            from fastapi import FastAPI, Query
            from fastapi.responses import FileResponse, JSONResponse
            from fastapi.middleware.cors import CORSMiddleware

            app = FastAPI(title="APEX V2 Investor Dashboard")
            app.add_middleware(CORSMiddleware, allow_origins=["*"],
                             allow_methods=["*"], allow_headers=["*"])
            trader = self

            def _period_pnl(hours: int) -> dict:
                """Compute P&L for a rolling time window."""
                now = datetime.now(timezone.utc)
                cutoff = (now - timedelta(hours=hours)).isoformat()
                ref_eq = trader.initial_bankroll
                for entry in trader.equity_history:
                    if entry["ts"] <= cutoff:
                        ref_eq = entry["equity"]
                change = trader.equity - ref_eq
                pct = (change / ref_eq * 100) if ref_eq > 0 else 0
                return {"pnl": round(change, 2), "pct": round(pct, 1)}

            def _worst_dip(days: int = 30) -> float:
                """Compute worst peak-to-trough drawdown in equity history."""
                cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
                eqs = [e["equity"] for e in trader.equity_history if e["ts"] > cutoff]
                if len(eqs) < 2:
                    return 0.0
                peak = eqs[0]
                worst = 0.0
                for e in eqs:
                    if e > peak:
                        peak = e
                    dd = (peak - e) / peak * 100 if peak > 0 else 0
                    if dd > worst:
                        worst = dd
                return round(-worst, 1)

            def _category_for_strategy(strategy: str) -> str:
                """Map strategy name to category."""
                mapping = {
                    "weather": "weather", "crypto": "crypto",
                    "macro": "finance", "sports": "sports",
                    "events": "events",
                }
                return mapping.get(strategy, "other")

            # ---- Serve the dashboard HTML ----
            @app.get("/")
            async def dashboard():
                return FileResponse(DASHBOARD_HTML, media_type="text/html")

            # ---- /api/mode ----
            @app.get("/api/mode")
            async def api_mode():
                return {"mode": "PAPER"}

            # ---- /api/overview ----
            @app.get("/api/overview")
            async def api_overview():
                now = datetime.now(timezone.utc)
                eq = trader.equity
                deployed = trader.deployed_capital
                bankroll = trader.bankroll
                total = deployed + bankroll if (deployed + bankroll) > 0 else 1

                # Period P&L
                periods = {
                    "today": _period_pnl(24),
                    "week": _period_pnl(168),
                    "month": _period_pnl(720),
                    "worst_dip_pct": _worst_dip(30),
                    "last7": _period_pnl(168),
                    "last30": _period_pnl(720),
                }

                # Calendar week (Mon-Sun)
                weekday = now.weekday()
                mon = now - timedelta(days=weekday)
                mon = mon.replace(hour=0, minute=0, second=0, microsecond=0)
                week_hours = max((now - mon).total_seconds() / 3600, 1)
                periods["cal_week"] = _period_pnl(int(week_hours))

                return {
                    "equity": round(eq, 2),
                    "bankroll": round(bankroll, 2),
                    "deployed": round(deployed, 2),
                    "initial_bankroll": trader.initial_bankroll,
                    "realized_pnl": round(trader.total_realized_pnl, 2),
                    "unrealized_pnl": round(trader.total_unrealized_pnl, 2),
                    "trades_executed": trader.trades_executed,
                    "open_positions": len(trader.positions),
                    "wins": trader.wins,
                    "losses": trader.losses,
                    "win_rate": round(trader.win_rate, 4),
                    "breaker": trader.breaker.level.value,
                    "drawdown_pct": round(trader.breaker.drawdown_pct, 2),
                    "cycle": trader._cycle,
                    "regime": trader.regime,
                    "timestamp": now.isoformat(),
                    "periods": periods,
                }

            # ---- /api/positions/closed-full ----
            @app.get("/api/positions/closed-full")
            async def api_closed_full(hours: int = Query(default=168)):
                cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
                result = []
                for t in trader.closed_trades:
                    closed_at = t.get("exit_time", "")
                    if closed_at and closed_at < cutoff:
                        continue
                    entry = t.get("entry_time", "")
                    hold_hours = 0
                    if entry and closed_at:
                        try:
                            e = datetime.fromisoformat(entry.replace("Z", "+00:00"))
                            c = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
                            hold_hours = round((c - e).total_seconds() / 3600, 2)
                        except (ValueError, TypeError):
                            pass

                    pnl = t.get("realized_pnl", 0)
                    cost = t.get("cost_basis", 1)
                    roi = pnl / cost if cost > 0 else 0

                    result.append({
                        "ledgerId": t.get("position_id", ""),
                        "market": t.get("question", ""),
                        "market_id": t.get("market_id", ""),
                        "strategy": t.get("strategy", ""),
                        "direction": t.get("direction", ""),
                        "category": _category_for_strategy(t.get("strategy", "")),
                        "entry_price": t.get("entry_price", 0),
                        "exit_price": t.get("exit_price"),
                        "pnl": round(pnl, 2),
                        "roi": round(roi, 4),
                        "openedAt": entry,
                        "closedAt": closed_at,
                        "holdHours": hold_hours,
                        "closeReason": t.get("status", ""),
                        "closeSummary": t.get("closeSummary", ""),
                        "entryThesis": t.get("entryThesis", ""),
                        "improvementNote": t.get("improvementNote", ""),
                    })
                return result

            # ---- /api/equity/curve ----
            @app.get("/api/equity/curve")
            async def api_equity_curve(days: int = Query(default=7)):
                cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
                points = []
                for e in trader.equity_history:
                    if e["ts"] >= cutoff:
                        points.append({"date": e["ts"], "equity": e["equity"]})
                if not points:
                    points = [{"date": datetime.now(timezone.utc).isoformat(),
                              "equity": trader.equity}]
                return points

            # ---- /api/open-positions-live ----
            @app.get("/api/open-positions-live")
            async def api_open_live():
                result = []
                for pos in trader.positions.values():
                    roi = pos.unrealized_pnl / pos.cost_basis if pos.cost_basis > 0 else 0
                    result.append({
                        "position_id": pos.position_id,
                        "market_id": pos.market_id,
                        "question": pos.question,
                        "direction": pos.direction,
                        "strategy": pos.strategy,
                        "category": _category_for_strategy(pos.strategy),
                        "entry_price": pos.entry_price,
                        "current_price": pos.current_price,
                        "cost_basis": pos.cost_basis,
                        "shares": pos.shares,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "edge_at_entry": pos.edge_at_entry,
                        "entry_time": pos.entry_time,
                        "expires_at": pos.expires_at,
                        "roi": round(roi, 4),
                        "stop_loss": pos.stop_loss,
                        "take_profit": pos.take_profit,
                        "peak_pnl_pct": pos.peak_pnl_pct,
                        "mark_fresh": True,
                    })
                return result

            # ---- /api/category-breakdown-timed ----
            @app.get("/api/category-breakdown-timed")
            async def api_category_breakdown(hours: int = Query(default=168)):
                cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
                cats = {}
                for name in ["weather", "crypto", "finance", "sports", "events", "other"]:
                    cats[name] = {"totalPnl": 0, "wins": 0, "losses": 0,
                                  "trades": 0, "openPositions": 0, "deployed": 0,
                                  "winRate": 0}

                for t in trader.closed_trades:
                    closed_at = t.get("exit_time", "")
                    if closed_at and closed_at < cutoff:
                        continue
                    cat = _category_for_strategy(t.get("strategy", ""))
                    if cat not in cats:
                        cat = "other"
                    c = cats[cat]
                    c["trades"] += 1
                    c["totalPnl"] += t.get("realized_pnl", 0)
                    if t.get("realized_pnl", 0) >= 0:
                        c["wins"] += 1
                    else:
                        c["losses"] += 1

                for pos in trader.positions.values():
                    cat = _category_for_strategy(pos.strategy)
                    if cat not in cats:
                        cat = "other"
                    cats[cat]["openPositions"] += 1
                    cats[cat]["deployed"] += pos.cost_basis

                for c in cats.values():
                    if c["trades"] > 0:
                        c["winRate"] = round(c["wins"] / c["trades"], 4)
                    c["totalPnl"] = round(c["totalPnl"], 2)
                    c["deployed"] = round(c["deployed"], 2)

                return cats

            # ---- /api/category-status ----
            @app.get("/api/category-status")
            async def api_category_status():
                result = {}
                strategies = {"weather": "weather", "crypto": "crypto",
                             "macro": "finance", "sports": "sports",
                             "events": "events"}
                for strat, cat in strategies.items():
                    cfg = trader.learner.get_strategy_config(strat)
                    has_activity = cfg.get("total_trades", 0) > 0 or cat in trader._signal_counts
                    has_positions = any(p.strategy == strat for p in trader.positions.values())
                    if has_positions:
                        status = "LIVE"
                    elif has_activity:
                        status = "LIVE"
                    else:
                        status = "IDLE"
                    result[cat] = {
                        "status": status,
                        "signals10m": trader.get_signal_count(cat),
                    }
                return result

            # ---- /api/strategy-breakdown-timed ----
            @app.get("/api/strategy-breakdown-timed")
            async def api_strategy_breakdown(hours: int = Query(default=8760)):
                cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
                strats = {}
                for t in trader.closed_trades:
                    closed_at = t.get("exit_time", "")
                    if closed_at and closed_at < cutoff:
                        continue
                    s = t.get("strategy", "other")
                    if s not in strats:
                        strats[s] = {"trades": 0, "wins": 0, "losses": 0,
                                    "pnl": 0, "hold_minutes": []}
                    d = strats[s]
                    d["trades"] += 1
                    pnl = t.get("realized_pnl", 0)
                    d["pnl"] += pnl
                    if pnl >= 0:
                        d["wins"] += 1
                    else:
                        d["losses"] += 1
                    # Hold time
                    entry = t.get("entry_time", "")
                    exit_t = t.get("exit_time", "")
                    if entry and exit_t:
                        try:
                            e = datetime.fromisoformat(entry.replace("Z", "+00:00"))
                            x = datetime.fromisoformat(exit_t.replace("Z", "+00:00"))
                            d["hold_minutes"].append((x - e).total_seconds() / 60)
                        except (ValueError, TypeError):
                            pass

                result = []
                for s, d in strats.items():
                    avg_hold = round(sum(d["hold_minutes"]) / len(d["hold_minutes"])) if d["hold_minutes"] else 0
                    wr = d["wins"] / d["trades"] if d["trades"] > 0 else 0
                    result.append({
                        "strategy": s,
                        "trades": d["trades"],
                        "wins": d["wins"],
                        "losses": d["losses"],
                        "winRate": round(wr, 4),
                        "pnl": round(d["pnl"], 2),
                        "avgHoldMinutes": avg_hold,
                    })
                result.sort(key=lambda x: x["pnl"], reverse=True)
                return result

            # ---- /api/trade-detail/:id ----
            @app.get("/api/trade-detail/{trade_id}")
            async def api_trade_detail(trade_id: str, narrate: int = 0):
                # Search open positions first
                for pos in trader.positions.values():
                    if pos.position_id == trade_id:
                        roi = pos.unrealized_pnl / pos.cost_basis if pos.cost_basis > 0 else 0
                        return {
                            "ledgerId": pos.position_id,
                            "market": pos.question,
                            "market_id": pos.market_id,
                            "strategy": pos.strategy,
                            "direction": pos.direction,
                            "category": _category_for_strategy(pos.strategy),
                            "entry_price": pos.entry_price,
                            "exit_price": None,
                            "pnl": pos.unrealized_pnl,
                            "roi": round(roi, 4),
                            "openedAt": pos.entry_time,
                            "closedAt": None,
                            "holdHours": round((datetime.now(timezone.utc) - datetime.fromisoformat(pos.entry_time.replace("Z", "+00:00"))).total_seconds() / 3600, 2),
                            "closeReason": "OPEN",
                            "closeSummary": "Position is still open. Close summary will be generated when the position closes.",
                            "entryThesis": "",
                            "improvementNote": "",
                            "decisionInputs": {
                                "edge": pos.edge_at_entry,
                                "stop_loss": pos.stop_loss,
                                "take_profit": pos.take_profit,
                            },
                        }

                # Search closed trades
                for t in trader.closed_trades:
                    if t.get("position_id") == trade_id:
                        # v3 flywheel: on ?narrate=1, generate the on-demand
                        # narratives once and cache them on the trade record
                        # (persisted by the next save_state).
                        changed = False
                        if narrate:
                            if not t.get("closeSummary"):
                                s = await narrate_close(t)
                                if s:
                                    t["closeSummary"] = s
                                    changed = True
                            if not t.get("entryThesis"):
                                s = await narrate_entry_thesis(t)
                                if s:
                                    t["entryThesis"] = s
                                    changed = True
                            if not t.get("improvementNote"):
                                s = await narrate_improvement(t)
                                if s:
                                    t["improvementNote"] = s
                                    changed = True
                            if changed:
                                trader.save_state()
                        pnl = t.get("realized_pnl", 0)
                        cost = t.get("cost_basis", 1)
                        roi = pnl / cost if cost > 0 else 0
                        entry = t.get("entry_time", "")
                        closed = t.get("exit_time", "")
                        hold_hours = 0
                        if entry and closed:
                            try:
                                e = datetime.fromisoformat(entry.replace("Z", "+00:00"))
                                c = datetime.fromisoformat(closed.replace("Z", "+00:00"))
                                hold_hours = round((c - e).total_seconds() / 3600, 2)
                            except (ValueError, TypeError):
                                pass
                        return {
                            "ledgerId": t.get("position_id", ""),
                            "market": t.get("question", ""),
                            "market_id": t.get("market_id", ""),
                            "strategy": t.get("strategy", ""),
                            "direction": t.get("direction", ""),
                            "category": _category_for_strategy(t.get("strategy", "")),
                            "entry_price": t.get("entry_price", 0),
                            "exit_price": t.get("exit_price"),
                            "pnl": round(pnl, 2),
                            "roi": round(roi, 4),
                            "openedAt": entry,
                            "closedAt": closed,
                            "holdHours": hold_hours,
                            "closeReason": t.get("status", ""),
                            "closeSummary": t.get("closeSummary", ""),
                            "entryThesis": t.get("entryThesis", ""),
                            "improvementNote": t.get("improvementNote", ""),
                            "decisionInputs": {
                                "edge": t.get("edge_at_entry", 0),
                                "direction": t.get("direction", ""),
                                "strategy": t.get("strategy", ""),
                            },
                        }

                return JSONResponse(status_code=404, content={"error": "Trade not found"})

            # ---- /api/health (keep existing) ----
            @app.get("/api/health")
            async def health():
                return {
                    "status": "healthy",
                    "mode": "PAPER",
                    "equity": round(trader.equity, 2),
                    "cycle": trader._cycle,
                }

            config = uvicorn.Config(
                app, host="0.0.0.0", port=8080, log_level="warning"
            )
            server = uvicorn.Server(config)
            await server.serve()

        except Exception as e:
            logger.warning("v2.dashboard_failed", error=str(e))

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self):
        """Start all systems."""
        logger.info("v2.starting", mode="PAPER", bankroll=f"${self.bankroll:.0f}")

        self.load_state()
        self._running = True
        self._start_time = time.time()

        print("\n" + "=" * 70)
        print("  APEX V2 — Autonomous Kalshi Trading System")
        print("=" * 70)
        print(f"  Mode:        PAPER")
        print(f"  Bankroll:    ${self.bankroll:,.0f}")
        print(f"  Target:      $1,000,000 (365 days)")
        print(f"  Positions:   {len(self.positions)} open")
        print(f"  Trades:      {self.trades_executed} executed")
        print(f"  P&L:         ${self.total_realized_pnl:+,.2f}")
        print(f"  Breaker:     {self.breaker.level.value}")
        print(f"  Strategies:  Weather + Crypto + Macro + Sports + Events")
        print(f"  Agents:      Scanner + Analyzer + Learner + Reporter")
        print(f"  Cycle:       every {self.CYCLE_SECONDS}s (full scan)")
        print(f"  Pos Poll:    every {self.POSITION_POLL_SECONDS}s (open positions)")
        print(f"  Dashboard:   http://100.64.161.91:8080")
        print(f"  Telegram:    Hourly updates (Andrew + Scott)")
        print(f"  Commands:    UI changes via Telegram (Andrew & Scott)")
        print(f"  State:       {STATE_FILE}")
        print("=" * 70 + "\n")

        # Send startup notification
        await self.reporter.send(
            f"*🚀 APEX V2 Started*\n\n"
            f"Paper trading with ${self.bankroll:,.0f}\n"
            f"Target: $1,000,000 in 365 days\n\n"
            f"Strategies: Weather, Crypto, Macro\n"
            f"Dashboard: http://100.64.161.91:8080\n\n"
            f"Send 'UI <change>' to modify dashboards."
        )

        try:
            await asyncio.gather(
                self.trading_loop(),
                self.position_monitor_loop(),
                self.run_dashboard(),
                self.reporting_loop(),
                self.learning_loop(),
                self.commander.run(),
            )
        except asyncio.CancelledError:
            logger.info("v2.shutdown")
        except KeyboardInterrupt:
            logger.info("v2.shutdown_keyboard")
        finally:
            self._running = False
            self.save_state()
            logger.info("v2.state_saved_on_exit")


async def main():
    trader = ApexV2Trader(bankroll=1000.0)
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
