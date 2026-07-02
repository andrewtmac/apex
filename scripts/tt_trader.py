"""TastyTrade paper book — stocks, bond ETFs, and crypto for APEX.

Separate position model from the Kalshi binary book (equity prices aren't
probabilities). Quotes come from Yahoo (same source epik-trade uses — the
TastyTrade cert environment serves no real market data); the TastyTrade
sandbox session is the account/order layer for the eventual live path.

Strategy: the relative-strength momentum rule that survived epik-trade's
walk-forward validation (phase 10, OOS PF 2.0-3.1): 20-day return > 2%,
5-day return > 0, RSI(14) < 70, volume >= 80% of 10-day average. Exits are
ATR-based (2.5x stop / 2.0x target) with a trailing stop coherent with the
target, plus a 10-trading-day max hold. Sizing is risk-budgeted off equity.

Fidelity: 5bps slippage each side on stocks/ETFs, 30bps taker fee each side
on crypto. Integer shares (fractional for crypto).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import httpx
import structlog

logger = structlog.get_logger("apex.tt")

STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "tt_state_v2.json")

# Watchlists — stocks are the union of epik's two OOS-validated universes
STOCKS = ["SPY", "QQQ", "AMD", "GOOG", "AMZN",
          "SMCI", "DKNG", "HOOD", "TQQQ", "SOXL", "XLK", "ENPH", "ARKK"]
BONDS = ["TLT", "IEF", "LQD", "HYG"]
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD"]

CYCLE_SECONDS = 60
MAX_POSITIONS = 8
MAX_DEPLOYED_PCT = 0.80
RISK_PCT_PER_TRADE = 0.04      # risk budget: 4% of equity per position (ATR-stop distance)
MAX_POSITION_PCT = 0.10        # hard cap: 10% of equity per position
MIN_POSITION_USD = 20.0
MAX_HOLD_DAYS = 10             # validated hold from the phase-10 grid
MAX_DAILY_LOSS_PCT = 0.08
CONSEC_LOSS_COOLDOWN = (3, 2700)  # 3 straight losses -> 45min pause
STOCK_SLIPPAGE = 0.0005        # 5bps per side
CRYPTO_FEE = 0.003             # 30bps per side


def _asset_class(symbol: str) -> str:
    if symbol in BONDS:
        return "bonds"
    if symbol.endswith("-USD"):
        return "crypto"
    return "stocks"


def _category(symbol: str) -> str:
    return "crypto" if symbol.endswith("-USD") else "finance"


@dataclass
class TTPosition:
    position_id: str
    symbol: str
    asset_class: str
    direction: str            # LONG only for now (validated rule is long-only)
    entry_price: float
    shares: float
    cost_basis: float
    entry_time: str
    stop_price: float
    target_price: float
    atr: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    peak_pnl_pct: float = 0.0
    edge_at_entry: float = 0.0
    status: str = "OPEN"
    exit_price: float | None = None
    exit_time: str | None = None
    realized_pnl: float = 0.0
    entry_fee: float = 0.0


class TTPaperBook:
    def __init__(self, bankroll: float = 1000.0):
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.positions: dict[str, TTPosition] = {}
        self.closed_trades: list[dict] = []
        self.wins = 0
        self.losses = 0
        self.total_realized_pnl = 0.0
        self.signals_generated = 0
        self._daily_pnl = 0.0
        self._daily_date = ""
        self._consec_losses = 0
        self._cooldown_until = 0.0
        self._bars: dict[str, dict] = {}     # symbol -> {ts, closes, volumes, highs, lows}
        self._quotes: dict[str, float] = {}
        self._narrate = None                  # injected close-narration callback
        self.load_state()

    # ── equity ─────────────────────────────────────────────────────────
    @property
    def deployed(self) -> float:
        return sum(p.cost_basis for p in self.positions.values())

    @property
    def equity(self) -> float:
        return round(self.bankroll + self.deployed
                     + sum(p.unrealized_pnl for p in self.positions.values()), 2)

    # ── persistence ────────────────────────────────────────────────────
    def save_state(self):
        state = {
            "bankroll": round(self.bankroll, 2),
            "wins": self.wins, "losses": self.losses,
            "total_realized_pnl": round(self.total_realized_pnl, 2),
            "signals_generated": self.signals_generated,
            "daily_pnl": round(self._daily_pnl, 2),
            "daily_date": self._daily_date,
            "positions": [asdict(p) for p in self.positions.values()],
            "closed_trades": self.closed_trades[-500:],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, STATE_FILE)

    def load_state(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            s = json.load(open(STATE_FILE))
            self.bankroll = s.get("bankroll", self.bankroll)
            self.wins = s.get("wins", 0)
            self.losses = s.get("losses", 0)
            self.total_realized_pnl = s.get("total_realized_pnl", 0.0)
            self.signals_generated = s.get("signals_generated", 0)
            self._daily_pnl = s.get("daily_pnl", 0.0)
            self._daily_date = s.get("daily_date", "")
            self.closed_trades = s.get("closed_trades", [])
            for p in s.get("positions", []):
                pos = TTPosition(**p)
                self.positions[pos.symbol] = pos
        except Exception as e:
            logger.warning("tt.state_load_failed", error=str(e)[:200])

    # ── market data (Yahoo, same source epik uses) ─────────────────────
    async def _fetch_bars(self, client: httpx.AsyncClient, symbol: str) -> dict | None:
        cached = self._bars.get(symbol)
        if cached and time.time() - cached["ts"] < 1800:
            return cached
        try:
            r = await client.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                params={"range": "6mo", "interval": "1d"},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if r.status_code != 200:
                return cached
            res = r.json()["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            bars = {
                "ts": time.time(),
                "closes": [c for c in q["close"] if c is not None],
                "volumes": [v or 0 for v, c in zip(q["volume"], q["close"]) if c is not None],
                "highs": [h for h, c in zip(q["high"], q["close"]) if c is not None],
                "lows": [l for l, c in zip(q["low"], q["close"]) if c is not None],
                "live": float(res["meta"].get("regularMarketPrice") or 0),
            }
            if len(bars["closes"]) >= 30:
                self._bars[symbol] = bars
                return bars
        except Exception as e:
            logger.debug("tt.bars_error", symbol=symbol, error=str(e)[:120])
        return cached

    @staticmethod
    def _rsi(closes: list[float], period: int = 14) -> float | None:
        if len(closes) < period + 1:
            return None
        gains = losses = 0.0
        for i in range(-period, 0):
            d = closes[i] - closes[i - 1]
            if d > 0:
                gains += d
            else:
                losses -= d
        if losses == 0:
            return 100.0
        rs = gains / losses
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _atr(highs, lows, closes, period: int = 14) -> float | None:
        if len(closes) < period + 1:
            return None
        trs = []
        for i in range(-period, 0):
            trs.append(max(highs[i] - lows[i],
                           abs(highs[i] - closes[i - 1]),
                           abs(lows[i] - closes[i - 1])))
        return sum(trs) / period

    @staticmethod
    def _market_open() -> bool:
        now = datetime.now(timezone.utc)
        if now.weekday() >= 5:
            return False
        mins = now.hour * 60 + now.minute
        return 13 * 60 + 30 <= mins < 20 * 60  # 13:30-20:00 UTC ~ RTH

    # ── strategy: validated relative-strength momentum (long only) ─────
    def _evaluate(self, symbol: str, bars: dict) -> dict | None:
        closes, vols = bars["closes"], bars["volumes"]
        live = bars.get("live") or closes[-1]
        if len(closes) < 21 or live <= 0:
            return None
        ret20 = (live - closes[-21]) / closes[-21]
        ret5 = (live - closes[-6]) / closes[-6]
        rsi = self._rsi(closes)
        avg_vol = sum(vols[-10:]) / 10 if len(vols) >= 10 else 0
        vol_ok = vols[-1] >= 0.8 * avg_vol if avg_vol > 0 else True
        if ret20 > 0.02 and ret5 > 0 and (rsi is None or rsi < 70) and vol_ok:
            atr = self._atr(bars["highs"], bars["lows"], closes)
            if not atr or atr <= 0:
                return None
            return {"symbol": symbol, "price": live, "atr": atr,
                    "edge": round(ret20 * 0.3, 4), "ret20": round(ret20, 4)}
        return None

    # ── entries ─────────────────────────────────────────────────────────
    def _try_enter(self, sig: dict):
        symbol = sig["symbol"]
        if symbol in self.positions or len(self.positions) >= MAX_POSITIONS:
            return
        if time.time() < self._cooldown_until:
            return
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_date:
            self._daily_date, self._daily_pnl = today, 0.0
        if -self._daily_pnl >= self.equity * MAX_DAILY_LOSS_PCT:
            return

        price, atr = sig["price"], sig["atr"]
        stop_dist = 2.5 * atr
        # Risk-budget sizing: 4% of equity at the stop, 10% notional hard cap
        risk_budget = self.equity * RISK_PCT_PER_TRADE
        qty = risk_budget / stop_dist
        qty = min(qty, (self.equity * MAX_POSITION_PCT) / price)
        crypto = symbol.endswith("-USD")
        if not crypto:
            qty = float(int(qty))
        else:
            qty = round(qty, 6)
        cost = qty * price
        if qty <= 0 or cost < MIN_POSITION_USD or cost > self.bankroll * 0.95:
            return
        if self.deployed + cost > self.equity * MAX_DEPLOYED_PCT:
            return

        fill = price * (1 + (CRYPTO_FEE if crypto else STOCK_SLIPPAGE))
        fee = cost * (CRYPTO_FEE if crypto else STOCK_SLIPPAGE)
        pos = TTPosition(
            position_id=uuid.uuid4().hex[:8],
            symbol=symbol,
            asset_class=_asset_class(symbol),
            direction="LONG",
            entry_price=round(fill, 4),
            shares=qty,
            cost_basis=round(qty * fill, 2),
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_price=round(fill - stop_dist, 4),
            target_price=round(fill + 2.0 * atr, 4),
            atr=round(atr, 4),
            current_price=round(fill, 4),
            edge_at_entry=sig["edge"],
            entry_fee=round(fee, 2),
        )
        self.bankroll -= pos.cost_basis + fee
        self.positions[symbol] = pos
        self.save_state()
        logger.info("tt.trade_executed", symbol=symbol, qty=qty,
                    entry=pos.entry_price, stop=pos.stop_price,
                    target=pos.target_price, cost=pos.cost_basis,
                    asset=pos.asset_class)

    # ── exits ───────────────────────────────────────────────────────────
    def _close(self, pos: TTPosition, reason: str):
        crypto = pos.symbol.endswith("-USD")
        fill = pos.current_price * (1 - (CRYPTO_FEE if crypto else STOCK_SLIPPAGE))
        payout = pos.shares * fill
        pnl = round(payout - pos.cost_basis, 2)
        pos.status = reason
        pos.exit_price = round(fill, 4)
        pos.exit_time = datetime.now(timezone.utc).isoformat()
        pos.realized_pnl = pnl
        self.bankroll += payout
        self.total_realized_pnl += pnl
        self._daily_pnl += pnl
        if pnl >= 0:
            self.wins += 1
            self._consec_losses = 0
        else:
            self.losses += 1
            self._consec_losses += 1
            if self._consec_losses >= CONSEC_LOSS_COOLDOWN[0]:
                self._cooldown_until = time.time() + CONSEC_LOSS_COOLDOWN[1]
                self._consec_losses = 0
                logger.warning("tt.cooldown", minutes=CONSEC_LOSS_COOLDOWN[1] // 60)
        record = asdict(pos)
        # narrator-compatible fields
        record.update({
            "market_id": pos.symbol,
            "question": f"{pos.symbol} {pos.asset_class} momentum long",
            "venue": "tastytrade",
            "strategy": f"tt_{pos.asset_class}_momentum",
            "stop_loss": round((pos.stop_price - pos.entry_price) / pos.entry_price, 4),
            "take_profit": round((pos.target_price - pos.entry_price) / pos.entry_price, 4),
        })
        self.closed_trades.append(record)
        if self._narrate:
            try:
                self._narrate(record)
            except Exception:
                pass
        self.save_state()
        logger.info("tt.position_closed", symbol=pos.symbol, reason=reason,
                    pnl=f"${pnl:+.2f}", bankroll=f"${self.bankroll:.0f}")

    def check_exits(self):
        for symbol in list(self.positions):
            pos = self.positions[symbol]
            px = pos.current_price
            if px <= 0:
                continue
            roi = pos.unrealized_pnl / pos.cost_basis if pos.cost_basis else 0
            target_roi = (pos.target_price - pos.entry_price) / pos.entry_price
            if px >= pos.target_price:
                self._close(pos, "TAKE_PROFIT"); del self.positions[symbol]; continue
            if px <= pos.stop_price:
                self._close(pos, "STOP_LOSS"); del self.positions[symbol]; continue
            # trailing stop coherent with target: activate at 60% of target ROI
            if pos.peak_pnl_pct >= max(0.02, 0.6 * target_roi):
                if pos.peak_pnl_pct - roi >= max(0.01, pos.peak_pnl_pct / 3):
                    self._close(pos, "TRAILING_STOP"); del self.positions[symbol]; continue
            try:
                held_d = (datetime.now(timezone.utc)
                          - datetime.fromisoformat(pos.entry_time)).days
                if held_d >= MAX_HOLD_DAYS:
                    self._close(pos, "MAX_HOLD"); del self.positions[symbol]
            except (ValueError, TypeError):
                pass

    # ── main loop ───────────────────────────────────────────────────────
    async def run_loop(self):
        logger.info("tt.loop_started", stocks=len(STOCKS), bonds=len(BONDS),
                    crypto=len(CRYPTO), bankroll=self.bankroll)
        async with httpx.AsyncClient(timeout=15) as client:
            while True:
                try:
                    market_open = self._market_open()
                    symbols = list(CRYPTO) + (STOCKS + BONDS if market_open else [])
                    # marks for open equity positions even off-hours (last close)
                    for s in self.positions:
                        if s not in symbols:
                            symbols.append(s)

                    results = await asyncio.gather(
                        *[self._fetch_bars(client, s) for s in symbols],
                        return_exceptions=True)
                    bars_by_symbol = {
                        s: b for s, b in zip(symbols, results)
                        if isinstance(b, dict)
                    }

                    # mark to market
                    for s, pos in self.positions.items():
                        b = bars_by_symbol.get(s)
                        if b and (b.get("live") or 0) > 0:
                            pos.current_price = b["live"]
                            pos.unrealized_pnl = round(
                                (pos.current_price - pos.entry_price) * pos.shares, 2)
                            r = pos.unrealized_pnl / pos.cost_basis if pos.cost_basis else 0
                            if r > pos.peak_pnl_pct:
                                pos.peak_pnl_pct = r

                    self.check_exits()

                    # entries: crypto 24/7, stocks/bonds only in RTH
                    for s, b in bars_by_symbol.items():
                        if s in self.positions:
                            continue
                        if not s.endswith("-USD") and not market_open:
                            continue
                        sig = self._evaluate(s, b)
                        if sig:
                            self.signals_generated += 1
                            self._try_enter(sig)

                except Exception as e:
                    logger.warning("tt.cycle_error", error=str(e)[:200])
                await asyncio.sleep(CYCLE_SECONDS)


def get_tt_book() -> TTPaperBook:
    """Singleton accessor used by apex_v2 integration."""
    global _BOOK
    try:
        return _BOOK
    except NameError:
        _BOOK = TTPaperBook()
        return _BOOK
