"""TastyTrade paper book — full-universe quant stock engine for APEX.

Separate position model from the Kalshi binary book (equity prices aren't
probabilities). Quotes come from Yahoo (the TastyTrade cert environment
serves no real market data); the TastyTrade sandbox session is the
account/order layer for the eventual live path.

Strategy stack — the persistent, documented anomalies from four decades of
quantitative equity research, combined so every entry needs several
independent edges to line up:

  1. Cross-sectional momentum (Jegadeesh & Titman 1993): rank the whole
     S&P 500 universe by 12-month-minus-1-month return each day; only the
     top-ranked names are tradeable ("focus list").
  2. Trend / regime filter (Faber 2007): no new stock longs unless SPY is
     above its 200-day MA, and the name itself is above its own 50- and
     200-day MAs. Cuts the left tail that kills momentum in bear regimes.
  3. 52-week-high proximity (George & Hwang 2004): names near their
     52-week high continue to outperform; proximity feeds the rank score.
  4. Low-volatility tilt (Haugen & Baker 1991): between two candidates
     with equal momentum, prefer the calmer one — better risk-adjusted
     carry and smaller stop distances.
  5. Short-term reversal avoidance (Lehmann 1990): never chase a >8%
     5-day spike; overextended names mean-revert first.
  6. Volatility-budgeted sizing (modern risk parity practice): position
     size is set so a 2.5×ATR stop costs a fixed 4% of equity — every
     position contributes roughly equal risk regardless of the name.
  7. Entry trigger: the relative-strength rule that survived epik-trade's
     walk-forward validation (phase 10, OOS PF 2.0-3.1): 20-day return
     > 2%, 5-day return > 0, RSI(14) < 70, volume >= 80% of 10-day avg.
  8. Exits sized for momentum's 1-3 month payoff horizon: 2.5×ATR stop,
     4×ATR target with a trailing stop once 60% of the move is banked,
     trend-break exit below the 50dma, flat-time stop (10 trading days
     with <2% ROI), 30-day hard cap. Let winners run; cut the dead wood.

Universe: S&P 500 constituents scraped from Wikipedia (cached 7 days at
data/sp500_universe.json, embedded fallback list if the scrape fails)
plus the legacy high-beta watchlist, bond ETFs, and crypto. The full
universe is re-ranked once per trading day; only the top FOCUS_N stocks
are polled intraday, which keeps Yahoo request volume sane.

Fidelity: 5bps slippage each side on stocks/ETFs, 30bps taker fee each
side on crypto. Integer shares (fractional for crypto).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import httpx
import structlog

logger = structlog.get_logger("apex.tt")

STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "tt_state_v2.json")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
UNIVERSE_CACHE = os.path.join(DATA_DIR, "sp500_universe.json")
SCAN_FILE = os.path.join(DATA_DIR, "tt_daily_scan.json")

# Legacy high-beta watchlist (epik's two OOS-validated universes) — always in
# the universe even when not S&P 500 members.
CORE_STOCKS = ["SPY", "QQQ", "AMD", "GOOG", "AMZN",
               "SMCI", "DKNG", "HOOD", "TQQQ", "SOXL", "XLK", "ENPH", "ARKK"]
BONDS = ["TLT", "IEF", "LQD", "HYG"]
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD"]

# Embedded fallback if the Wikipedia scrape fails cold (top liquid megacaps).
FALLBACK_SP500 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "BRK-B",
    "JPM", "LLY", "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "WMT", "NFLX",
    "JNJ", "CRM", "BAC", "ORCL", "CVX", "ABBV", "KO", "MRK", "AMD", "PEP",
    "ADBE", "TMO", "WFC", "CSCO", "ACN", "MCD", "QCOM", "LIN", "IBM", "GE",
    "ABT", "CAT", "TXN", "DHR", "INTU", "VZ", "AMGN", "PFE", "NOW", "NEE",
    "DIS", "PM", "GS", "ISRG", "SPGI", "CMCSA", "UBER", "RTX", "AXP", "MS",
    "BKNG", "UNP", "T", "HON", "LOW", "COP", "BLK", "ETN", "SYK", "PANW",
    "AMAT", "LMT", "PLTR", "SCHW", "TJX", "C", "BSX", "DE", "ADP", "MDT",
    "VRTX", "GILD", "SBUX", "MMC", "BA", "CB", "FI", "MU", "LRCX", "ANET",
    "REGN", "PLD", "SO", "KLAC", "MO", "ICE", "DUK", "SHW", "SNPS", "CDNS",
]

CYCLE_SECONDS = 60
FOCUS_N = 40                   # top-ranked stocks polled intraday
MAX_POSITIONS = 12
MAX_STOCK_POSITIONS = 9
MAX_DEPLOYED_PCT = 0.80
RISK_PCT_PER_TRADE = 0.04      # risk budget: 4% of equity at the ATR stop
MAX_POSITION_PCT = 0.10        # hard cap: 10% of equity per position
MIN_POSITION_USD = 20.0
MIN_DOLLAR_VOLUME = 20e6       # liquidity floor for universe membership
MIN_PRICE = 5.0
FLAT_TIME_DAYS = 10            # exit if <2% ROI after this many days
MAX_HOLD_DAYS = 30             # hard cap sized for momentum's payoff horizon
MAX_DAILY_LOSS_PCT = 0.08
CONSEC_LOSS_COOLDOWN = (3, 2700)  # 3 straight losses -> 45min pause
STOCK_SLIPPAGE = 0.0005        # 5bps per side
CRYPTO_FEE = 0.003             # 30bps per side
FETCH_CONCURRENCY = 6          # Yahoo politeness for the full-universe scan


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
    direction: str            # LONG only (every validated edge here is long)
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
        self._universe: list[str] = []        # full stock universe (S&P 500 + core)
        self._focus: list[str] = []           # today's top-ranked tradeable stocks
        self._scan_meta: dict[str, dict] = {} # symbol -> {dma50, dma200, mom, score...}
        self._scan_date = ""
        self._risk_on = True                  # SPY > 200dma regime flag
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

    # ── universe (S&P 500 + core watchlist) ─────────────────────────────
    async def _load_universe(self, client: httpx.AsyncClient) -> list[str]:
        """S&P 500 constituents, cached 7 days; Wikipedia scrape with an
        embedded megacap fallback so a scrape failure never blanks the bot."""
        try:
            c = json.load(open(UNIVERSE_CACHE))
            if time.time() - c.get("fetched_at", 0) < 7 * 86400 and len(c.get("symbols", [])) > 100:
                return c["symbols"]
        except Exception:
            pass
        symbols: list[str] = []
        try:
            r = await client.get(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True)
            if r.status_code == 200:
                # Constituent tickers link to nyse.com/nasdaq.com/cboe quote pages
                found = re.findall(
                    r'href="https://www\.(?:nyse|nasdaq|cboe)\.com/[^"]*"[^>]*>([A-Z][A-Z.\-]{0,5})</a>',
                    r.text)
                symbols = sorted({t.replace(".", "-") for t in found})
        except Exception as e:
            logger.warning("tt.universe_scrape_failed", error=str(e)[:150])
        if len(symbols) < 100:
            logger.warning("tt.universe_fallback", scraped=len(symbols))
            symbols = list(FALLBACK_SP500)
        else:
            os.makedirs(DATA_DIR, exist_ok=True)
            json.dump({"fetched_at": time.time(), "symbols": symbols},
                      open(UNIVERSE_CACHE, "w"))
        return symbols

    async def _daily_universe_scan(self, client: httpx.AsyncClient):
        """Once per day: pull 1y daily bars for the whole universe, compute the
        quant factors, set the regime flag, and rank the focus list."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._scan_date == today and self._focus:
            return
        if not self._universe:
            self._universe = await self._load_universe(client)
        scan_start = time.time()
        universe = sorted(set(self._universe) | set(CORE_STOCKS))
        sem = asyncio.Semaphore(FETCH_CONCURRENCY)

        async def fetch(sym: str):
            async with sem:
                b = await self._fetch_bars(client, sym, rng="1y", cache_s=6 * 3600)
                await asyncio.sleep(0.05)
                return sym, b

        results = await asyncio.gather(*(fetch(s) for s in ["SPY"] + universe + BONDS),
                                       return_exceptions=True)
        bars = {s: b for r in results if isinstance(r, tuple)
                for s, b in [r] if isinstance(b, dict)}

        # Regime: SPY above its 200dma (Faber). Below it, momentum's left tail
        # opens up — no new stock longs.
        spy = bars.get("SPY")
        if spy and len(spy["closes"]) >= 200:
            spy_px = spy.get("live") or spy["closes"][-1]
            self._risk_on = spy_px > sum(spy["closes"][-200:]) / 200
        meta: dict[str, dict] = {}
        for sym, b in bars.items():
            closes, vols = b["closes"], b["volumes"]
            if len(closes) < 60:
                continue
            px = b.get("live") or closes[-1]
            n = len(closes)
            dma50 = sum(closes[-50:]) / min(50, n)
            dma200 = sum(closes[-200:]) / min(200, n)
            # 12-1 momentum: skip the most recent month (short-term reversal)
            back = closes[0] if n < 252 else closes[-252]
            mom = (closes[-22] - back) / back if n >= 44 and back > 0 else 0.0
            hi52 = max(closes[-252:]) if n >= 2 else px
            prox52 = px / hi52 if hi52 > 0 else 0
            rets = [(closes[i] - closes[i - 1]) / closes[i - 1]
                    for i in range(n - 20, n) if closes[i - 1] > 0]
            vol20 = (sum(r * r for r in rets) / len(rets)) ** 0.5 * (252 ** 0.5) if rets else 1.0
            adv = sum(v * c for v, c in zip(vols[-20:], closes[-20:])) / min(20, n)
            meta[sym] = {"px": px, "dma50": dma50, "dma200": dma200, "mom": mom,
                         "prox52": prox52, "vol20": vol20, "adv": adv}

        # Eligibility, then composite rank: momentum-led with 52wk-high and
        # low-vol tilts (per-factor rank averaging, robust to outliers).
        elig = [s for s, m in meta.items()
                if s not in BONDS and s != "SPY"
                and m["px"] >= MIN_PRICE and m["adv"] >= MIN_DOLLAR_VOLUME
                and m["px"] > m["dma50"] > 0 and m["px"] > m["dma200"] > 0
                and m["mom"] > 0]

        def ranks(key, reverse):
            order = sorted(elig, key=lambda s: meta[s][key], reverse=reverse)
            return {s: i for i, s in enumerate(order)}

        r_mom = ranks("mom", True)
        r_prox = ranks("prox52", True)
        r_vol = ranks("vol20", False)
        scored = sorted(elig, key=lambda s: 0.5 * r_mom[s] + 0.25 * r_prox[s] + 0.25 * r_vol[s])
        self._focus = scored[:FOCUS_N]
        self._scan_meta = meta
        self._scan_date = today
        for s in self._focus:
            meta[s]["focus"] = True
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            json.dump({"date": today, "risk_on": self._risk_on,
                       "universe": len(universe), "eligible": len(elig),
                       "focus": self._focus,
                       "top10": [{"symbol": s, **{k: round(v, 4) for k, v in meta[s].items()
                                                  if isinstance(v, float)}}
                                 for s in self._focus[:10]]},
                      open(SCAN_FILE, "w"), indent=2)
        except Exception:
            pass
        logger.info("tt.daily_scan", universe=len(universe), with_bars=len(bars),
                    eligible=len(elig), focus=len(self._focus),
                    risk_on=self._risk_on, secs=round(time.time() - scan_start, 1),
                    top5=self._focus[:5])

    # ── market data (Yahoo, same source epik uses) ─────────────────────
    async def _fetch_bars(self, client: httpx.AsyncClient, symbol: str,
                          rng: str = "1y", cache_s: int = 1800) -> dict | None:
        cached = self._bars.get(symbol)
        if cached and time.time() - cached["ts"] < cache_s:
            return cached
        try:
            r = await client.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                params={"range": rng, "interval": "1d"},
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

    # ── entry trigger: validated relative-strength rule + quant gates ───
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
        stock = _asset_class(symbol) == "stocks"
        if stock:
            # Cross-sectional + regime + trend gates on top of the trigger
            if not self._risk_on or symbol not in self._focus:
                return None
            m = self._scan_meta.get(symbol)
            if m and live <= m["dma50"]:
                return None
            if ret5 > 0.08:      # short-term reversal: don't chase spikes
                return None
        if ret20 > 0.02 and ret5 > 0 and (rsi is None or rsi < 70) and vol_ok:
            atr = self._atr(bars["highs"], bars["lows"], closes)
            if not atr or atr <= 0:
                return None
            m = self._scan_meta.get(symbol, {})
            return {"symbol": symbol, "price": live, "atr": atr,
                    "edge": round(ret20 * 0.3 + m.get("mom", 0) * 0.1, 4),
                    "ret20": round(ret20, 4)}
        return None

    # ── entries ─────────────────────────────────────────────────────────
    def _try_enter(self, sig: dict):
        symbol = sig["symbol"]
        if symbol in self.positions or len(self.positions) >= MAX_POSITIONS:
            return
        if (_asset_class(symbol) == "stocks"
                and sum(1 for p in self.positions.values()
                        if p.asset_class == "stocks") >= MAX_STOCK_POSITIONS):
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
            # 4×ATR target: momentum pays over 1-3 months, trailing stop
            # (below) banks the move if it stalls earlier
            target_price=round(fill + 4.0 * atr, 4),
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
            # trend break: a losing momentum long under its 50dma is dead wood
            m = self._scan_meta.get(symbol)
            if (m and pos.asset_class == "stocks" and roi < 0
                    and px < m["dma50"]):
                self._close(pos, "TREND_BREAK"); del self.positions[symbol]; continue
            try:
                held_d = (datetime.now(timezone.utc)
                          - datetime.fromisoformat(pos.entry_time)).days
                if held_d >= MAX_HOLD_DAYS:
                    self._close(pos, "MAX_HOLD"); del self.positions[symbol]; continue
                if held_d >= FLAT_TIME_DAYS and roi < 0.02:
                    self._close(pos, "TIME_FLAT"); del self.positions[symbol]
            except (ValueError, TypeError):
                pass

    # ── main loop ───────────────────────────────────────────────────────
    async def run_loop(self):
        logger.info("tt.loop_started", core=len(CORE_STOCKS), bonds=len(BONDS),
                    crypto=len(CRYPTO), bankroll=self.bankroll)
        async with httpx.AsyncClient(timeout=15) as client:
            while True:
                try:
                    market_open = self._market_open()
                    # Daily full-universe rank (also on cold start so the
                    # focus list exists before the bell)
                    await self._daily_universe_scan(client)

                    symbols = list(CRYPTO) + (self._focus + BONDS if market_open else [])
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
