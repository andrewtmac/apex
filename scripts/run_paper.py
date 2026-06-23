#!/usr/bin/env python3
"""APEX Paper Trading + Dashboard Entry Point.

Starts:
1. Data ingestion (Polymarket, Kalshi scanning)
2. Model loading (XGBoost, LightGBM, Calibration)
3. Signal generation loop (scan markets -> features -> ensemble -> trade gate)
4. Paper trade execution with position tracking and P&L
5. Market resolution monitoring
6. Dashboard on port 8080
7. State persistence to paper_state.json
"""

import asyncio
import json
import os
import pickle
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import httpx
import numpy as np
import pandas as pd
import structlog

from apex.risk.circuit_breaker import CircuitBreaker

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
)

logger = structlog.get_logger()

MODELS_DIR = Path(__file__).parent.parent / "models_store"
STATE_FILE = Path(__file__).parent.parent / "paper_state.json"


# ---------------------------------------------------------------------------
# Paper Position
# ---------------------------------------------------------------------------


@dataclass
class PaperPosition:
    """A tracked paper trading position in a prediction market."""

    position_id: str
    market_id: str
    venue: str
    question: str
    direction: str  # BUY (bet YES) or SELL (bet NO)
    entry_price: float
    cost_basis: float  # USD deployed
    shares: float
    entry_time: str
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    status: str = "OPEN"
    exit_price: float | None = None
    exit_time: str | None = None
    realized_pnl: float = 0.0
    resolution: str | None = None
    edge_at_entry: float = 0.0
    peak_pnl_pct: float = 0.0  # Peak unrealized ROI for trailing stop


# Stop words for event token extraction
_EVENT_STOP = frozenset(
    "will the a an in on of at is to for and or by vs with be do does did "
    "win lose end draw score both teams over under spread this that have "
    "has not been what when".split()
)


# ---------------------------------------------------------------------------
# Live Sports Data (ESPN scores + The Odds API)
# ---------------------------------------------------------------------------


class SportsDataService:
    """Live scores from ESPN and bookmaker odds for sports trade validation."""

    ESPN_SOCCER = [
        "fifa.world", "fifa.worldq",
        "eng.1", "esp.1", "ger.1", "ita.1", "fra.1", "usa.1",
        "uefa.champions", "uefa.europa",
    ]
    ODDS_SPORTS = [
        "soccer_fifa_world_cup", "soccer_epl", "soccer_spain_la_liga",
        "soccer_germany_bundesliga", "soccer_italy_serie_a",
        "soccer_france_ligue_one", "soccer_usa_mls",
    ]

    def __init__(self):
        self.scores: list[dict] = []
        self.odds: list[dict] = []
        self._scores_ts = 0.0
        self._odds_ts = 0.0

    # -- Fetchers ----------------------------------------------------------

    async def refresh_scores(self):
        """Fetch live scores from ESPN (free, every 2 min)."""
        if time.time() - self._scores_ts < 90:
            return
        results: list[dict] = []
        async with httpx.AsyncClient(timeout=10) as client:
            for league in self.ESPN_SOCCER:
                try:
                    r = await client.get(
                        f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
                    )
                    if r.status_code != 200:
                        continue
                    for ev in r.json().get("events", []):
                        p = self._parse_espn(ev)
                        if p:
                            results.append(p)
                except Exception:
                    continue
        self.scores = results
        self._scores_ts = time.time()
        logger.debug("sports.scores_refreshed", n=len(results))

    async def refresh_odds(self):
        """Fetch bookmaker odds (rate-limited, every 30 min)."""
        key = os.getenv("ODDS_API_KEY", "")
        if not key or time.time() - self._odds_ts < 1800:
            return
        results: list[dict] = []
        async with httpx.AsyncClient(timeout=15) as client:
            for sport in self.ODDS_SPORTS:
                try:
                    r = await client.get(
                        f"https://api.the-odds-api.com/v4/sports/{sport}/odds",
                        params={
                            "apiKey": key,
                            "regions": "us,eu",
                            "markets": "h2h",
                            "oddsFormat": "decimal",
                        },
                    )
                    if r.status_code != 200:
                        continue
                    for ev in r.json():
                        p = self._parse_odds(ev)
                        if p:
                            results.append(p)
                except Exception:
                    continue
        self.odds = results
        self._odds_ts = time.time()
        remaining = None
        logger.info("sports.odds_refreshed", n=len(results))

    # -- Parsers -----------------------------------------------------------

    def _parse_espn(self, event: dict) -> dict | None:
        comps = event.get("competitions", [])
        if not comps:
            return None
        comp = comps[0]
        teams = comp.get("competitors", [])
        if len(teams) < 2:
            return None
        home = away = None
        for t in teams:
            if t.get("homeAway") == "home":
                home = t
            else:
                away = t
        if not home or not away:
            home, away = teams[0], teams[1]
        st = comp.get("status", {}).get("type", {})
        state = st.get("state", "")
        return {
            "home": home.get("team", {}).get("displayName", ""),
            "away": away.get("team", {}).get("displayName", ""),
            "home_score": int(home.get("score", 0) or 0),
            "away_score": int(away.get("score", 0) or 0),
            "status": {"pre": "scheduled", "in": "live", "post": "final"}.get(
                state, state
            ),
            "completed": st.get("completed", False),
        }

    def _parse_odds(self, event: dict) -> dict | None:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        bms = event.get("bookmakers", [])
        if not bms:
            return None
        buckets: dict[str, list[float]] = {"home": [], "away": [], "draw": []}
        for bm in bms:
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "h2h":
                    continue
                for oc in mkt.get("outcomes", []):
                    price = oc.get("price", 0)
                    if price <= 1:
                        continue
                    name = oc.get("name", "")
                    if name == home:
                        buckets["home"].append(1 / price)
                    elif name == away:
                        buckets["away"].append(1 / price)
                    elif name.lower() == "draw":
                        buckets["draw"].append(1 / price)

        def avg(lst: list[float]) -> float | None:
            return sum(lst) / len(lst) if lst else None

        hp, ap, dp = avg(buckets["home"]), avg(buckets["away"]), avg(buckets["draw"])
        total = (hp or 0) + (ap or 0) + (dp or 0)
        if total > 0:
            if hp:
                hp /= total
            if ap:
                ap /= total
            if dp:
                dp /= total
        return {
            "home": home,
            "away": away,
            "home_prob": round(hp, 4) if hp else None,
            "away_prob": round(ap, 4) if ap else None,
            "draw_prob": round(dp, 4) if dp else None,
            "n_books": len(bms),
        }

    # -- Matching ----------------------------------------------------------

    def _match(self, question: str, items: list[dict]) -> dict | None:
        """Match a market question to a score/odds entry by team names."""
        q = question.lower()
        best, best_n = None, 0
        for item in items:
            n = 0
            for key in ("home", "away"):
                for part in item.get(key, "").lower().split():
                    if len(part) > 3 and part in q:
                        n += 1
            if n > best_n:
                best, best_n = item, n
        return best if best_n >= 1 else None

    def find_score(self, question: str) -> dict | None:
        return self._match(question, self.scores)

    def find_odds(self, question: str) -> dict | None:
        return self._match(question, self.odds)

    # -- Pre-trade validation ----------------------------------------------

    def validate_entry(self, signal: dict) -> tuple[bool, str]:
        """Check live data before entering a sports position."""
        q = signal["question"].lower()

        # Live scores
        score = self.find_score(signal["question"])
        if score:
            hs, aws = score["home_score"], score["away_score"]
            total = hs + aws

            if score["status"] == "final" or score["completed"]:
                return False, (
                    f"game_over ({score['home']} {hs}-{aws} {score['away']})"
                )

            if score["status"] == "live":
                # O/U: don't SELL if total already over the line
                ou = re.search(r"o/u\s+(\d+\.?\d*)", q)
                if ou and signal["direction"] == "SELL":
                    if total > float(ou.group(1)):
                        return False, f"ou_busted (total={total}>{ou.group(1)})"

                # BTTS: don't SELL if both teams scored
                if "both" in q and "score" in q and signal["direction"] == "SELL":
                    if hs > 0 and aws > 0:
                        return False, "btts_already_true"

                # Draw: don't BUY if score gap >= 2
                if "draw" in q and signal["direction"] == "BUY":
                    if abs(hs - aws) >= 2:
                        return False, f"draw_unlikely ({hs}-{aws})"

        # Bookmaker odds
        odds = self.find_odds(signal["question"])
        if odds:
            book_prob = self._prob_for_question(q, odds)
            if book_prob is not None:
                mkt = signal["market_price"]
                if signal["direction"] == "BUY" and book_prob < mkt - 0.08:
                    return False, (
                        f"books_disagree (book={book_prob:.2f} < mkt={mkt:.2f})"
                    )
                if signal["direction"] == "SELL" and book_prob > mkt + 0.08:
                    return False, (
                        f"books_disagree (book={book_prob:.2f} > mkt={mkt:.2f})"
                    )

        return True, "ok"

    # -- Ongoing position reassessment -------------------------------------

    def reassess_position(self, pos: PaperPosition) -> str | None:
        """Check if live scores mean we should exit. Returns reason or None."""
        score = self.find_score(pos.question)
        if not score:
            return None

        q = pos.question.lower()
        hs, aws = score["home_score"], score["away_score"]
        total = hs + aws

        if score["status"] == "final" or score["completed"]:
            return f"GAME_FINAL ({score['home']} {hs}-{aws} {score['away']})"

        if score["status"] == "live":
            # O/U: SELL is dead if total already over the line
            ou = re.search(r"o/u\s+(\d+\.?\d*)", q)
            if ou and pos.direction == "SELL" and total > float(ou.group(1)):
                return f"LIVE_OU_BUSTED ({total}>{ou.group(1)})"

            # BTTS: SELL is dead if both scored
            if "both" in q and "score" in q and pos.direction == "SELL":
                if hs > 0 and aws > 0:
                    return f"LIVE_BTTS_TRUE ({hs}-{aws})"

            # "Will X win?": BUY bad if team losing by 2+
            if "win" in q:
                home_in_q = any(
                    p in q for p in score["home"].lower().split() if len(p) > 3
                )
                away_in_q = any(
                    p in q for p in score["away"].lower().split() if len(p) > 3
                )
                if home_in_q and pos.direction == "BUY" and aws - hs >= 2:
                    return f"LIVE_LOSING ({hs}-{aws})"
                if away_in_q and pos.direction == "BUY" and hs - aws >= 2:
                    return f"LIVE_LOSING ({hs}-{aws})"
                # SELL: exit early if team winning big (our SELL will resolve as loss)
                if home_in_q and pos.direction == "SELL" and hs - aws >= 3:
                    return f"LIVE_OPPONENT_WINNING ({hs}-{aws})"
                if away_in_q and pos.direction == "SELL" and aws - hs >= 3:
                    return f"LIVE_OPPONENT_WINNING ({hs}-{aws})"

            # Draw: BUY bad if gap >= 2
            if "draw" in q and pos.direction == "BUY" and abs(hs - aws) >= 2:
                return f"LIVE_DRAW_UNLIKELY ({hs}-{aws})"

        return None

    # -- Helpers -----------------------------------------------------------

    def _prob_for_question(self, q: str, odds: dict) -> float | None:
        """Determine which bookmaker probability applies to a question."""
        if "draw" in q:
            return odds.get("draw_prob")
        home_parts = [p for p in odds["home"].lower().split() if len(p) > 3]
        away_parts = [p for p in odds["away"].lower().split() if len(p) > 3]
        if any(p in q for p in home_parts) and "win" in q:
            return odds.get("home_prob")
        if any(p in q for p in away_parts) and "win" in q:
            return odds.get("away_prob")
        return None


# ---------------------------------------------------------------------------
# Paper Trader
# ---------------------------------------------------------------------------


class ApexPaperTrader:
    """Paper trading orchestrator with real position tracking and P&L."""

    MAX_POSITIONS = 20
    MAX_DEPLOYED_PCT = 0.70
    MIN_EDGE = 0.04               # 4% minimum edge (lowered for Kalshi weather)
    MAX_SPREAD = 0.12
    MIN_VOLUME_POLYMARKET = 500    # Polymarket needs decent volume
    MIN_VOLUME_KALSHI = 0          # Kalshi weather always has series liquidity
    MIN_POSITION_USD = 10.0
    RESOLUTION_CHECK_EVERY = 10    # Every 10 cycles (~20 min)
    STALE_POSITION_HOURS = 168     # Close after 7 days
    CYCLE_SECONDS = 120

    # Exit rules
    TAKE_PROFIT_ROI = 0.20         # Take profit at 20% ROI
    STOP_LOSS_ROI = -0.30          # Stop loss at 30% loss
    TRAILING_ACTIVATE = 0.15  # Activate trailing stop at 15% peak
    TRAILING_GIVEBACK = 0.08  # Exit if drops 8% from peak
    MAX_PER_EVENT = 1  # Max positions per sporting event

    def __init__(self, bankroll: float = 5000.0):
        self.xgb_model = None
        self.lgbm_model = None
        self.calibrator = None
        self.feature_names: list[str] = []

        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.positions: dict[str, PaperPosition] = {}
        self.closed_trades: list[dict] = []
        self.signals_generated = 0
        self.trades_executed = 0
        self.total_realized_pnl = 0.0
        self.wins = 0
        self.losses = 0

        self.breaker = CircuitBreaker(initial_equity=bankroll, auto_recovery=True)
        self.sports = SportsDataService()
        self.equity_history: list[dict] = []

        self._running = False
        self._cycle = 0
        self._start_time = time.time()

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
        """Restore trading state from disk."""
        if not STATE_FILE.exists():
            logger.info("apex.no_saved_state", msg="Starting fresh")
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

            for p in state.get("positions", []):
                pos = PaperPosition(**p)
                self.positions[pos.market_id] = pos

            if "breaker" in state:
                self.breaker = CircuitBreaker.from_state_dict(state["breaker"])

            logger.info(
                "apex.state_restored",
                bankroll=f"${self.bankroll:.0f}",
                positions=len(self.positions),
                trades=self.trades_executed,
                realized_pnl=f"${self.total_realized_pnl:+.2f}",
                breaker=self.breaker.level.value,
            )
        except Exception as e:
            logger.warning("apex.state_restore_failed", error=str(e))

    def save_state(self):
        """Persist trading state to disk (atomic write)."""
        state = {
            "bankroll": round(self.bankroll, 2),
            "trades_executed": self.trades_executed,
            "total_realized_pnl": round(self.total_realized_pnl, 2),
            "wins": self.wins,
            "losses": self.losses,
            "signals_generated": self.signals_generated,
            "cycle": self._cycle,
            "positions": [asdict(p) for p in self.positions.values()],
            "closed_trades": self.closed_trades[-500:],
            "equity_history": self.equity_history[-5000:],
            "breaker": self.breaker.state_dict(),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = STATE_FILE.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, default=str)
        tmp.replace(STATE_FILE)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(self):
        """Load trained models from disk."""
        logger.info("apex.loading_models")

        xgb_path = MODELS_DIR / "xgboost_prob_v1.pkl"
        if xgb_path.exists():
            with open(xgb_path, "rb") as f:
                data = pickle.load(f)
            self.xgb_model = data["model"]
            self.feature_names = data["feature_names"]
            logger.info("apex.xgboost_loaded", n_features=len(self.feature_names))

        lgbm_path = MODELS_DIR / "lgbm_return_v1.pkl"
        if lgbm_path.exists():
            with open(lgbm_path, "rb") as f:
                data = pickle.load(f)
            self.lgbm_model = data["model"]
            logger.info("apex.lightgbm_loaded")

        cal_path = MODELS_DIR / "bayesian_calibration_v1.pkl"
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                data = pickle.load(f)
            self.calibrator = data["calibrator"]
            logger.info("apex.calibrator_loaded")

    # ------------------------------------------------------------------
    # Market scanning
    # ------------------------------------------------------------------

    async def scan_polymarket_markets(self) -> list[dict]:
        """Scan Polymarket for active tradeable markets."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    "https://gamma-api.polymarket.com/markets",
                    params={
                        "limit": 50,
                        "active": True,
                        "closed": False,
                        "order": "volume24hr",
                        "ascending": False,
                    },
                )
                resp.raise_for_status()
                markets = resp.json()

                active = []
                for m in markets:
                    vol = float(m.get("volume24hr", 0) or 0)
                    if vol < 100:
                        continue

                    try:
                        prices = json.loads(
                            m.get("outcomePrices", '["0.5","0.5"]')
                        )
                        current_price = float(prices[0])
                    except (ValueError, IndexError, TypeError):
                        current_price = 0.5

                    active.append(
                        {
                            "market_id": m.get("conditionId", m.get("id", "")),
                            "question": m.get("question", ""),
                            "category": m.get("category", "other"),
                            "current_price": current_price,
                            "volume_24h": vol,
                            "spread": float(m.get("spread", 0) or 0),
                            "end_date": m.get("endDateIso", ""),
                            "venue": "polymarket",
                        }
                    )

                return active

        except Exception as e:
            logger.warning("apex.polymarket_scan_failed", error=str(e))
            return []

    # High-volume Kalshi series to scan (weather, crypto, indices, macro)
    KALSHI_SERIES = [
        # Weather (highest volume -- $300K+/day)
        "KXHIGHNY", "KXHIGHCHI", "KXHIGHLA", "KXHIGHMIA", "KXHIGHDC",
        "KXHIGHHOU", "KXHIGHDAL", "KXHIGHDEN", "KXHIGHPHX", "KXHIGHATL",
        "KXHIGHSF", "KXHIGHBOS", "KXHIGHSEA",
        # Crypto ranges
        "KXBTC", "KXETH",
        # Stock indices
        "KXINX", "KXNDX",
        # Macro
        "KXFED", "KXCPI", "KXJOBLESS", "KXGDP",
        # World Cup / Sports
        "KXWCWINNER", "KXWCGROUP",
        # Other high-interest
        "KXFDAAPPROVE", "KXFUSION",
    ]

    async def scan_kalshi_markets(self) -> list[dict]:
        """Scan Kalshi for active tradeable markets by series.

        Queries specific high-volume series (weather, crypto, indices)
        instead of the default endpoint which returns MVE parlays.
        """
        active: list[dict] = []
        seen: set[str] = set()

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                for series in self.KALSHI_SERIES:
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

                            # Use mid price if last_price is 0
                            if price == 0 and bid > 0 and ask > 0:
                                price = (bid + ask) / 2
                            elif price == 0 and ask > 0:
                                price = ask
                            elif price == 0:
                                continue  # Skip markets with no price

                            vol_24h = float(m.get("volume_24h_fp", 0) or 0)
                            spread = round(ask - bid, 4) if ask > 0 and bid > 0 else 0.0

                            active.append({
                                "market_id": ticker,
                                "question": m.get("title", "")[:100],
                                "category": series,
                                "current_price": price,
                                "volume_24h": vol_24h,
                                "spread": spread,
                                "end_date": m.get("expiration_time", ""),
                                "venue": "kalshi",
                            })

                    except Exception:
                        continue  # Skip failed series, try next

            logger.debug(
                "apex.kalshi_scan",
                n_markets=len(active),
                series_checked=len(self.KALSHI_SERIES),
            )
            return active

        except Exception as e:
            logger.warning("apex.kalshi_scan_failed", error=str(e))
            return []

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def build_features(self, market: dict) -> pd.DataFrame | None:
        """Build feature DataFrame for a market."""
        try:
            price = market["current_price"]
            vol = market["volume_24h"]

            features = {
                "final_price": price,
                "price_distance_from_50": abs(price - 0.5),
                "price_near_round": min(
                    abs(price - round(price * 10) / 10), 0.05
                ),
                "implied_prob": price,
                "log_odds": np.log(max(price, 0.01) / max(1 - price, 0.01)),
                "log_volume": np.log1p(vol),
                "volume_zscore": 0.0,
                "log_duration_hours": 0.0,
                "short_duration": 0.0,
                "log_liquidity": 0.0,
                "is_polymarket": 1.0 if market["venue"] == "polymarket" else 0.0,
                "is_kalshi": 1.0 if market["venue"] == "kalshi" else 0.0,
                "is_heavy_favorite": 1.0 if price > 0.85 else 0.0,
                "is_heavy_underdog": 1.0 if price < 0.15 else 0.0,
                "is_tossup": 1.0 if 0.4 < price < 0.6 else 0.0,
            }

            for name in self.feature_names:
                if name.startswith("cat_") and name not in features:
                    features[name] = 0.0

            row = [features.get(name, 0.0) for name in self.feature_names]
            return pd.DataFrame([row], columns=self.feature_names)

        except Exception as e:
            logger.debug("apex.feature_build_failed", error=str(e))
            return None

    # ------------------------------------------------------------------
    # Signal evaluation
    # ------------------------------------------------------------------

    def evaluate_signal(self, market: dict, features: pd.DataFrame) -> dict | None:
        """Run ensemble and return trade signal if edge is sufficient."""
        if self.xgb_model is None:
            return None

        try:
            xgb_prob = float(self.xgb_model.predict(features)[0])

            if self.calibrator is not None:
                cal_prob = float(self.calibrator.transform([xgb_prob])[0])
            else:
                cal_prob = xgb_prob

            market_price = market["current_price"]
            edge = cal_prob - market_price

            if self.lgbm_model is not None:
                predicted_return = float(self.lgbm_model.predict(features)[0])
            else:
                predicted_return = edge

            spread = market.get("spread", 0)

            if abs(edge) < self.MIN_EDGE:
                return None
            if spread > self.MAX_SPREAD:
                return None
            # Venue-specific volume thresholds
            min_vol = (
                self.MIN_VOLUME_KALSHI
                if market.get("venue") == "kalshi"
                else self.MIN_VOLUME_POLYMARKET
            )
            if market["volume_24h"] < min_vol:
                return None

            direction = "BUY" if edge > 0 else "SELL"

            # Simplified Bayesian Kelly with circuit breaker
            kelly_fraction = 0.20
            cb_mult = self.breaker.sizing_multiplier()
            if cb_mult <= 0:
                return None

            size_pct = min(kelly_fraction * abs(edge) / 0.10, 0.10) * cb_mult
            size_usd = self.bankroll * size_pct

            if size_usd < self.MIN_POSITION_USD:
                return None

            self.signals_generated += 1

            return {
                "market_id": market["market_id"],
                "venue": market["venue"],
                "question": market["question"][:80],
                "direction": direction,
                "edge": round(edge, 4),
                "xgb_prob": round(xgb_prob, 4),
                "cal_prob": round(cal_prob, 4),
                "market_price": round(market_price, 4),
                "predicted_return": round(predicted_return, 4),
                "size_usd": round(size_usd, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.debug("apex.signal_eval_failed", error=str(e))
            return None

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def execute_paper_trade(self, signal: dict) -> PaperPosition | None:
        """Execute a paper trade from an approved signal."""
        market_id = signal["market_id"]

        if market_id in self.positions:
            return None

        if len(self.positions) >= self.MAX_POSITIONS:
            return None

        if not self.breaker.can_open_new_position():
            return None

        # Event correlation: max 1 position per sporting event
        if self._overlaps_existing_event(signal["question"]):
            return None

        # Live sports data validation (ESPN scores + bookmaker odds)
        approved, reason = self.sports.validate_entry(signal)
        if not approved:
            logger.info(
                "apex.sports_rejected",
                reason=reason,
                question=signal["question"][:50],
            )
            return None

        cost = signal["size_usd"]
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

        pos = PaperPosition(
            position_id=uuid.uuid4().hex[:8],
            market_id=market_id,
            venue=signal["venue"],
            question=signal["question"],
            direction=signal["direction"],
            entry_price=entry_price,
            cost_basis=cost,
            shares=round(shares, 2),
            entry_time=datetime.now(timezone.utc).isoformat(),
            current_price=entry_price,
            edge_at_entry=signal["edge"],
        )

        self.bankroll -= cost
        self.positions[market_id] = pos
        self.trades_executed += 1

        logger.info(
            "apex.trade_executed",
            id=pos.position_id,
            venue=pos.venue,
            direction=pos.direction,
            entry=f"{pos.entry_price:.3f}",
            cost=f"${pos.cost_basis:.0f}",
            shares=f"{pos.shares:.1f}",
            edge=f"{pos.edge_at_entry:+.4f}",
            open_positions=len(self.positions),
            bankroll=f"${self.bankroll:.0f}",
            question=pos.question[:50],
        )

        return pos

    # ------------------------------------------------------------------
    # Mark-to-market
    # ------------------------------------------------------------------

    def mark_to_market(self, price_map: dict[str, float]):
        """Update unrealized P&L and peak tracking for all open positions."""
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

            # Track peak ROI for trailing stop
            roi = pos.unrealized_pnl / pos.cost_basis if pos.cost_basis > 0 else 0
            if roi > pos.peak_pnl_pct:
                pos.peak_pnl_pct = roi

    # ------------------------------------------------------------------
    # Event grouping (prevent competing positions)
    # ------------------------------------------------------------------

    def _event_tokens(self, question: str) -> set[str]:
        """Extract entity tokens from a market question."""
        q = re.sub(r"[^\w\s'-]", " ", question.lower())
        q = re.sub(r"\b\d+\b", "", q)
        return {w for w in q.split() if len(w) > 2 and w not in _EVENT_STOP}

    def _overlaps_existing_event(self, question: str) -> bool:
        """Check if a question shares an event with any open position."""
        new = self._event_tokens(question)
        if not new:
            return False
        for pos in self.positions.values():
            existing = self._event_tokens(pos.question)
            shared = new & existing
            if len(shared) >= 2:
                return True
            # Single shared token: block if it's a likely proper noun (>3 chars)
            if len(shared) == 1 and len(list(shared)[0]) > 3:
                smaller = min(len(new), len(existing))
                if smaller <= 3:
                    return True
        return False

    # ------------------------------------------------------------------
    # Exit rules (profit-taking, stop-loss, trailing stop)
    # ------------------------------------------------------------------

    def _close_position_mtm(self, pos: PaperPosition, reason: str) -> float:
        """Close a position at current mark-to-market price. Returns P&L."""
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
        else:
            self.losses += 1

        self.breaker.update(equity=self.equity, trade_result=pnl)
        self.closed_trades.append(asdict(pos))
        return pnl

    def check_exits(self):
        """Check all positions for profit-taking, stop-loss, trailing stop."""
        exit_ids: list[str] = []

        for market_id, pos in self.positions.items():
            roi = pos.unrealized_pnl / pos.cost_basis if pos.cost_basis > 0 else 0

            # Take profit
            if roi >= self.TAKE_PROFIT_ROI:
                pnl = self._close_position_mtm(pos, "TAKE_PROFIT")
                exit_ids.append(market_id)
                logger.info(
                    "apex.take_profit",
                    id=pos.position_id,
                    roi=f"{roi:.0%}",
                    pnl=f"${pnl:+.2f}",
                    question=pos.question[:50],
                )
                continue

            # Stop loss
            if roi <= self.STOP_LOSS_ROI:
                pnl = self._close_position_mtm(pos, "STOP_LOSS")
                exit_ids.append(market_id)
                logger.info(
                    "apex.stop_loss",
                    id=pos.position_id,
                    roi=f"{roi:.0%}",
                    pnl=f"${pnl:+.2f}",
                    question=pos.question[:50],
                )
                continue

            # Trailing stop
            if pos.peak_pnl_pct >= self.TRAILING_ACTIVATE:
                giveback = pos.peak_pnl_pct - roi
                if giveback >= self.TRAILING_GIVEBACK:
                    pnl = self._close_position_mtm(pos, "TRAILING_STOP")
                    exit_ids.append(market_id)
                    logger.info(
                        "apex.trailing_stop",
                        id=pos.position_id,
                        peak=f"{pos.peak_pnl_pct:.0%}",
                        current=f"{roi:.0%}",
                        pnl=f"${pnl:+.2f}",
                        question=pos.question[:50],
                    )
                    continue

        for mid in exit_ids:
            del self.positions[mid]

        return len(exit_ids)

    def _check_live_sports(self) -> int:
        """Reassess open positions against live ESPN scores."""
        exit_ids: list[str] = []
        for market_id, pos in list(self.positions.items()):
            reason = self.sports.reassess_position(pos)
            if reason:
                pnl = self._close_position_mtm(pos, reason)
                exit_ids.append(market_id)
                logger.info(
                    "apex.live_sports_exit",
                    id=pos.position_id,
                    reason=reason,
                    pnl=f"${pnl:+.2f}",
                    question=pos.question[:50],
                )
        for mid in exit_ids:
            del self.positions[mid]
        return len(exit_ids)

    # ------------------------------------------------------------------
    # Period P&L and insights
    # ------------------------------------------------------------------

    def _equity_at_cutoff(self, cutoff_iso: str) -> float:
        """Get the equity snapshot closest to (at or before) a cutoff time."""
        result = None
        for entry in self.equity_history:
            if entry["ts"] <= cutoff_iso:
                result = entry["equity"]
            else:
                break
        if result is not None:
            return result
        # No entries before cutoff — use initial bankroll
        return self.initial_bankroll

    def compute_periods(self) -> dict:
        """Compute P&L for standard reporting periods."""
        now = datetime.now(timezone.utc)
        eq = self.equity

        def pnl_since(dt: datetime) -> tuple[float, float]:
            ref = self._equity_at_cutoff(dt.isoformat())
            change = eq - ref
            pct = (change / ref * 100) if ref > 0 else 0.0
            return round(change, 2), round(pct, 1)

        # Calendar periods
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_of_week = start_of_day - timedelta(days=start_of_day.weekday())
        start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        today_pnl, today_pct = pnl_since(start_of_day)
        week_pnl, week_pct = pnl_since(start_of_week)
        month_pnl, month_pct = pnl_since(start_of_month)

        # Rolling periods
        last_7d_pnl, last_7d_pct = pnl_since(now - timedelta(days=7))
        last_30d_pnl, last_30d_pct = pnl_since(now - timedelta(days=30))

        # Worst drawdown from equity history
        worst_dip = 0.0
        peak = 0.0
        for entry in self.equity_history:
            v = entry["equity"]
            if v > peak:
                peak = v
            if peak > 0:
                dd = (peak - v) / peak * 100
                if dd > worst_dip:
                    worst_dip = dd

        return {
            "today_pnl": today_pnl, "today_pct": today_pct,
            "week_pnl": week_pnl, "week_pct": week_pct,
            "month_pnl": month_pnl, "month_pct": month_pct,
            "last_7d_pnl": last_7d_pnl, "last_7d_pct": last_7d_pct,
            "last_30d_pnl": last_30d_pnl, "last_30d_pct": last_30d_pct,
            "worst_dip": round(-worst_dip, 1),
        }

    def compute_venue_stats(self) -> dict:
        """Compute per-venue statistics."""
        venues: dict[str, dict] = {}
        total_deployed = self.deployed_capital or 1

        for pos in self.positions.values():
            v = venues.setdefault(pos.venue, {
                "profit": 0.0, "wins": 0, "losses": 0, "total": 0,
                "deployed": 0.0, "open": 0, "unrealized": 0.0,
            })
            v["open"] += 1
            v["deployed"] += pos.cost_basis
            v["unrealized"] += pos.unrealized_pnl

        for trade in self.closed_trades:
            venue = trade.get("venue", "unknown")
            v = venues.setdefault(venue, {
                "profit": 0.0, "wins": 0, "losses": 0, "total": 0,
                "deployed": 0.0, "open": 0, "unrealized": 0.0,
            })
            pnl = trade.get("realized_pnl", 0)
            v["profit"] += pnl
            v["total"] += 1
            if pnl >= 0:
                v["wins"] += 1
            else:
                v["losses"] += 1

        # Add open trade counts to totals
        for pos in self.positions.values():
            venues[pos.venue]["total"] = venues[pos.venue].get("total", 0)

        for name, v in venues.items():
            v["win_rate"] = v["wins"] / v["total"] if v["total"] > 0 else 0.0
            v["share"] = (v["deployed"] / total_deployed * 100) if total_deployed > 0 else 0.0
            v["profit"] = round(v["profit"], 2)
            v["unrealized"] = round(v["unrealized"], 2)

        return venues

    def generate_insights(self) -> list[dict]:
        """Generate talking points for the dashboard."""
        insights = []

        # Venue concentration
        venues_used = set(p.venue for p in self.positions.values())
        if len(self.positions) >= 3 and len(venues_used) == 1:
            venue = list(venues_used)[0]
            insights.append({
                "tag": "Watch", "color": "amber",
                "title": f"All positions are on {venue.title()}",
                "body": "No diversification across venues. Consider whether "
                        "Kalshi markets offer complementary opportunities.",
            })

        # Direction bias
        if len(self.positions) >= 3:
            dirs = [p.direction for p in self.positions.values()]
            buy_pct = dirs.count("BUY") / len(dirs) * 100
            sell_pct = 100 - buy_pct
            if buy_pct >= 80:
                insights.append({
                    "tag": "Note", "color": "blue",
                    "title": f"{buy_pct:.0f}% of positions are BUY",
                    "body": "Strong directional bias toward YES outcomes. "
                            "The model may be systematically underestimating "
                            "market prices.",
                })
            elif sell_pct >= 80:
                insights.append({
                    "tag": "Note", "color": "blue",
                    "title": f"{sell_pct:.0f}% of positions are SELL",
                    "body": "Strong directional bias toward NO outcomes. "
                            "The model consistently thinks markets are "
                            "overpriced.",
                })

        # Deployment level
        dep_pct = (self.deployed_capital / self.equity * 100) if self.equity > 0 else 0
        if dep_pct > 50:
            insights.append({
                "tag": "Working", "color": "green",
                "title": f"{dep_pct:.0f}% of capital is deployed",
                "body": f"${self.deployed_capital:,.0f} in active positions, "
                        f"${self.bankroll:,.0f} in cash. "
                        f"Limit is {self.MAX_DEPLOYED_PCT*100:.0f}%.",
            })
        elif dep_pct < 10 and self.trades_executed > 0:
            insights.append({
                "tag": "Idle cash", "color": "amber",
                "title": "Most capital is sitting idle",
                "body": f"Only {dep_pct:.0f}% deployed. The models may not be "
                        "finding enough edge in current markets.",
            })

        # Win/loss streak
        total = self.wins + self.losses
        if total >= 5:
            wr = self.win_rate * 100
            if wr >= 60:
                insights.append({
                    "tag": "Opportunity", "color": "green",
                    "title": f"{wr:.0f}% win rate across {total} trades",
                    "body": "The model is performing well. Consider whether "
                            "deployment limits could be gradually increased.",
                })
            elif wr <= 35:
                insights.append({
                    "tag": "Risk", "color": "red",
                    "title": f"Win rate is {wr:.0f}% across {total} trades",
                    "body": "Below expected performance. Worth reviewing "
                            "whether the model needs retraining on recent data.",
                })

        # Circuit breaker
        if self.breaker.level.value != "GREEN":
            insights.append({
                "tag": "Risk", "color": "red",
                "title": f"Circuit breaker is {self.breaker.level.value}",
                "body": f"Drawdown has reached {self.breaker.drawdown_pct:.1f}%. "
                        "Trading capacity is reduced until equity recovers.",
            })

        # No resolved trades yet
        if self.trades_executed > 0 and total == 0:
            insights.append({
                "tag": "Note", "color": "blue",
                "title": "No trades have resolved yet",
                "body": f"{self.trades_executed} positions opened, waiting for "
                        "markets to settle. Resolution is checked every "
                        f"~{self.RESOLUTION_CHECK_EVERY * self.CYCLE_SECONDS // 60} minutes.",
            })

        # Large unrealized P&L
        unrealized = self.total_unrealized_pnl
        if abs(unrealized) > self.initial_bankroll * 0.05:
            direction = "up" if unrealized > 0 else "down"
            insights.append({
                "tag": "Watch" if unrealized < 0 else "Opportunity",
                "color": "amber" if unrealized < 0 else "green",
                "title": f"Unrealized P&L is {direction} ${abs(unrealized):,.0f}",
                "body": "This will become realized when markets settle. "
                        "Mark-to-market values update every cycle.",
            })

        return insights

    # ------------------------------------------------------------------
    # Resolution monitoring
    # ------------------------------------------------------------------

    async def check_resolutions(self):
        """Check if any open positions' markets have resolved."""
        resolved_ids: list[str] = []

        async with httpx.AsyncClient(timeout=15) as client:
            for market_id, pos in list(self.positions.items()):
                try:
                    outcome = None
                    if pos.venue == "polymarket":
                        outcome = await self._check_poly_resolution(
                            client, market_id
                        )
                    elif pos.venue == "kalshi":
                        outcome = await self._check_kalshi_resolution(
                            client, market_id
                        )

                    if outcome is not None:
                        self._resolve_position(pos, outcome)
                        resolved_ids.append(market_id)
                except Exception as e:
                    logger.debug(
                        "apex.resolution_check_error",
                        market=market_id,
                        error=str(e),
                    )

        for mid in resolved_ids:
            del self.positions[mid]

    async def _check_poly_resolution(
        self, client: httpx.AsyncClient, condition_id: str
    ) -> str | None:
        """Check Polymarket for market resolution."""
        resp = await client.get(
            "https://gamma-api.polymarket.com/markets",
            params={"condition_id": condition_id, "limit": 1},
        )
        resp.raise_for_status()
        markets = resp.json()
        if not markets:
            return None

        m = markets[0] if isinstance(markets, list) else markets
        if not (m.get("closed") or m.get("resolved")):
            return None

        outcome = str(m.get("resolvedOutcome", ""))
        if outcome in ("1", "Yes", "yes"):
            return "YES"
        elif outcome in ("0", "No", "no"):
            return "NO"
        return None

    async def _check_kalshi_resolution(
        self, client: httpx.AsyncClient, ticker: str
    ) -> str | None:
        """Check Kalshi for market settlement."""
        resp = await client.get(
            f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}",
        )
        resp.raise_for_status()
        data = resp.json()
        market = data.get("market", data)

        if market.get("status") != "settled":
            return None

        result = (market.get("result") or "").lower()
        if result in ("yes", "y"):
            return "YES"
        elif result in ("no", "n"):
            return "NO"
        return None

    def _resolve_position(self, pos: PaperPosition, outcome: str):
        """Close a position against a market resolution outcome."""
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
        else:
            self.losses += 1

        self.breaker.update(equity=self.equity, trade_result=pnl)

        self.closed_trades.append(asdict(pos))

        logger.info(
            "apex.position_resolved",
            id=pos.position_id,
            outcome=outcome,
            direction=pos.direction,
            entry=f"{pos.entry_price:.3f}",
            cost=f"${pos.cost_basis:.0f}",
            payout=f"${payout:.0f}",
            pnl=f"${pnl:+.2f}",
            bankroll=f"${self.bankroll:.0f}",
            breaker=self.breaker.level.value,
            question=pos.question[:50],
        )

    async def close_stale_positions(self):
        """Close positions that have been open too long at current MTM."""
        now = datetime.now(timezone.utc)
        stale_ids: list[str] = []

        for market_id, pos in self.positions.items():
            entry = datetime.fromisoformat(pos.entry_time)
            hours_open = (now - entry).total_seconds() / 3600
            if hours_open < self.STALE_POSITION_HOURS:
                continue

            self._close_position_mtm(pos, "CLOSED_STALE")
            stale_ids.append(market_id)

            logger.info(
                "apex.stale_position_closed",
                id=pos.position_id,
                hours_open=f"{hours_open:.0f}h",
                pnl=f"${pnl:+.2f}",
                question=pos.question[:50],
            )

        for mid in stale_ids:
            del self.positions[mid]

    # ------------------------------------------------------------------
    # Main trading loop
    # ------------------------------------------------------------------

    async def trading_loop(self):
        """Main loop: scan -> evaluate -> trade -> track -> repeat."""
        while self._running:
            self._cycle += 1
            t0 = time.time()

            # 1. Scan markets + refresh live sports data
            poly_markets, kalshi_markets, _, _ = await asyncio.gather(
                self.scan_polymarket_markets(),
                self.scan_kalshi_markets(),
                self.sports.refresh_scores(),
                self.sports.refresh_odds(),
            )
            all_markets = poly_markets + kalshi_markets

            # 2. Mark-to-market open positions
            price_map = {m["market_id"]: m["current_price"] for m in all_markets}
            self.mark_to_market(price_map)

            # 2b. Update breaker with current equity (enables auto-recovery)
            self.breaker.update(equity=self.equity)

            # 2c. Live sports reassessment (exit on score changes)
            live_exits = self._check_live_sports()

            # 2c. Check exit rules (profit-taking, stop-loss, trailing stop)
            exits = self.check_exits() + live_exits

            # 2c. Snapshot equity for chart
            self.equity_history.append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "equity": round(self.equity, 2),
            })

            # 3. Generate signals
            signals = []
            for market in all_markets:
                features = self.build_features(market)
                if features is None:
                    continue
                signal = self.evaluate_signal(market, features)
                if signal is not None:
                    signals.append(signal)

            # 4. Sort by absolute edge (strongest conviction first) and execute
            signals.sort(key=lambda s: abs(s["edge"]), reverse=True)

            new_trades = 0
            for signal in signals:
                pos = self.execute_paper_trade(signal)
                if pos is not None:
                    new_trades += 1

            # 5. Check resolutions periodically
            if self._cycle % self.RESOLUTION_CHECK_EVERY == 0:
                await self.check_resolutions()
                await self.close_stale_positions()

            # 6. Persist state
            self.save_state()

            elapsed = time.time() - t0

            # 7. Log cycle summary
            total_closed = self.wins + self.losses
            logger.info(
                "apex.cycle",
                cycle=self._cycle,
                markets=len(all_markets),
                signals=len(signals),
                new_trades=new_trades,
                exits=exits,
                open=len(self.positions),
                deployed=f"${self.deployed_capital:.0f}",
                cash=f"${self.bankroll:.0f}",
                equity=f"${self.equity:.0f}",
                unrealized=f"${self.total_unrealized_pnl:+.0f}",
                realized=f"${self.total_realized_pnl:+.0f}",
                trades=self.trades_executed,
                record=f"{self.wins}W-{self.losses}L"
                if total_closed > 0
                else "0-0",
                breaker=self.breaker.level.value,
                elapsed=f"{elapsed:.1f}s",
            )

            for sig in signals[:3]:
                logger.info(
                    "apex.top_signal",
                    venue=sig["venue"],
                    direction=sig["direction"],
                    edge=f"{sig['edge']:+.4f}",
                    size=f"${sig['size_usd']:.0f}",
                    question=sig["question"][:50],
                )

            await asyncio.sleep(self.CYCLE_SECONDS)

    # ------------------------------------------------------------------
    # Telegram daily digest
    # ------------------------------------------------------------------

    async def send_telegram(self, text: str):
        """Send a message via Telegram bot API."""
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            return

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                await client.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    data={
                        "chat_id": chat_id,
                        "text": text,
                        "parse_mode": "Markdown",
                    },
                )
        except Exception as e:
            logger.warning("apex.telegram_send_failed", error=str(e))

    def build_daily_digest(self) -> str:
        """Build the daily summary message."""
        eq = self.equity
        roi = ((eq - self.initial_bankroll) / self.initial_bankroll) * 100
        total_closed = self.wins + self.losses
        wr = f"{self.win_rate:.0%}" if total_closed > 0 else "N/A"
        dd = self.breaker.drawdown_pct
        uptime_h = (time.time() - self._start_time) / 3600

        # Today's resolved trades
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        todays_trades = [
            t for t in self.closed_trades
            if (t.get("exit_time") or "")[:10] == today
        ]
        todays_pnl = sum(t.get("realized_pnl", 0) for t in todays_trades)
        todays_wins = sum(1 for t in todays_trades if t.get("realized_pnl", 0) >= 0)
        todays_losses = len(todays_trades) - todays_wins

        # Top/bottom open positions
        sorted_pos = sorted(
            self.positions.values(),
            key=lambda p: p.unrealized_pnl,
            reverse=True,
        )

        lines = [
            f"*APEX Daily Digest*",
            f"_{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
            "",
            f"*Portfolio*",
            f"  Equity: `${eq:,.0f}`",
            f"  Cash: `${self.bankroll:,.0f}`",
            f"  Deployed: `${self.deployed_capital:,.0f}` ({self.deployed_capital / eq * 100:.0f}%)" if eq > 0 else "  Deployed: $0",
            f"  ROI: `{roi:+.1f}%`",
            "",
            f"*P&L*",
            f"  Realized: `${self.total_realized_pnl:+,.2f}`",
            f"  Unrealized: `${self.total_unrealized_pnl:+,.2f}`",
            f"  Today: `${todays_pnl:+,.2f}` ({todays_wins}W-{todays_losses}L)",
            "",
            f"*Performance*",
            f"  All-time: {self.wins}W-{self.losses}L ({wr})",
            f"  Total trades: {self.trades_executed}",
            f"  Signals generated: {self.signals_generated:,}",
            "",
            f"*Risk*",
            f"  Breaker: {self.breaker.level.value}",
            f"  Drawdown: {dd:.1f}%",
            f"  Open positions: {len(self.positions)}",
        ]

        if sorted_pos:
            lines.append("")
            lines.append(f"*Open Positions* ({len(sorted_pos)})")
            for p in sorted_pos[:5]:
                emoji = "+" if p.unrealized_pnl >= 0 else ""
                lines.append(
                    f"  {p.direction} {p.question[:35]}  "
                    f"`{emoji}${p.unrealized_pnl:.0f}`"
                )
            if len(sorted_pos) > 5:
                lines.append(f"  ... +{len(sorted_pos) - 5} more")

        lines.append("")
        lines.append(f"_Uptime: {uptime_h:.0f}h | Cycle: {self._cycle}_")

        return "\n".join(lines)

    async def digest_loop(self):
        """Send a Telegram digest 3x/day: 08:00, 16:00, 00:00 UTC."""
        schedule_hours = [0, 8, 16]

        while self._running:
            now = datetime.now(timezone.utc)
            # Find the next scheduled hour
            upcoming = []
            for h in schedule_hours:
                t = now.replace(hour=h, minute=0, second=0, microsecond=0)
                if t <= now:
                    from datetime import timedelta
                    t += timedelta(days=1)
                upcoming.append(t)
            target = min(upcoming)
            wait_seconds = (target - now).total_seconds()

            label = {0: "evening", 8: "morning", 16: "midday"}
            logger.info(
                "apex.digest_scheduled",
                slot=label.get(target.hour, ""),
                next_at=target.isoformat(),
                wait_h=f"{wait_seconds / 3600:.1f}",
            )
            await asyncio.sleep(wait_seconds)

            if not self._running:
                break

            digest = self.build_daily_digest()
            await self.send_telegram(digest)
            logger.info("apex.digest_sent", slot=label.get(target.hour, ""))

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    async def run_dashboard(self):
        """Start the monitoring dashboard on port 8080."""
        try:
            import uvicorn
            from fastapi import FastAPI
            from fastapi.responses import FileResponse

            app = FastAPI(title="APEX Dashboard")
            trader = self
            dashboard_html = Path(__file__).parent / "dashboard.html"

            @app.get("/")
            async def dashboard():
                return FileResponse(dashboard_html, media_type="text/html")

            @app.get("/api/dashboard-data")
            async def dashboard_data():
                periods = trader.compute_periods()
                venues = trader.compute_venue_stats()
                insights = trader.generate_insights()
                # Thin out equity history for the response (max 500 points)
                hist = trader.equity_history
                if len(hist) > 500:
                    step = len(hist) // 500
                    hist = hist[::step] + [hist[-1]]
                return {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "equity": round(trader.equity, 2),
                    "bankroll": round(trader.bankroll, 2),
                    "deployed": round(trader.deployed_capital, 2),
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
                    "signals_generated": trader.signals_generated,
                    "cycle": trader._cycle,
                    "uptime_seconds": time.time() - trader._start_time,
                    "periods": periods,
                    "equity_history": hist,
                    "positions": [asdict(p) for p in trader.positions.values()],
                    "closed_trades": trader.closed_trades[-50:],
                    "venues": venues,
                    "insights": insights,
                }

            @app.get("/api/health")
            async def health():
                return {
                    "status": "healthy",
                    "mode": "PAPER",
                    "equity": round(trader.equity, 2),
                    "bankroll": round(trader.bankroll, 2),
                    "deployed": round(trader.deployed_capital, 2),
                    "trades": trader.trades_executed,
                    "open_positions": len(trader.positions),
                    "breaker": trader.breaker.level.value,
                    "cycle": trader._cycle,
                }

            @app.get("/api/positions")
            async def positions():
                return [asdict(p) for p in trader.positions.values()]

            @app.get("/api/trades")
            async def trades():
                return trader.closed_trades[-50:]

            config = uvicorn.Config(
                app, host="0.0.0.0", port=8080, log_level="warning"
            )
            server = uvicorn.Server(config)
            await server.serve()

        except Exception as e:
            logger.warning("apex.dashboard_failed", error=str(e))

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self):
        """Start all systems."""
        logger.info("apex.starting", mode="PAPER", bankroll=f"${self.bankroll:.0f}")

        self.load_models()
        if self.xgb_model is None:
            logger.error(
                "apex.no_models", msg="No trained models found. Run training first."
            )
            return

        self.load_state()

        self._running = True
        self._start_time = time.time()

        print("\n" + "=" * 70)
        print("  APEX Paper Trading -- LIVE")
        print("=" * 70)
        print(f"  Mode:      PAPER")
        print(f"  Cash:      ${self.bankroll:,.0f}")
        print(f"  Equity:    ${self.equity:,.0f}")
        print(f"  Positions: {len(self.positions)} open")
        print(f"  Trades:    {self.trades_executed} executed")
        print(f"  P&L:       ${self.total_realized_pnl:+,.2f} realized")
        print(f"  Breaker:   {self.breaker.level.value}")
        print(f"  Models:    XGBoost + LightGBM + Calibration")
        print(f"  Venues:    Polymarket + Kalshi")
        print(f"  Cycle:     every {self.CYCLE_SECONDS}s")
        print(f"  Dashboard: http://localhost:8080")
        print(f"  State:     {STATE_FILE}")
        print("=" * 70 + "\n")

        try:
            await asyncio.gather(
                self.trading_loop(),
                self.run_dashboard(),
                self.digest_loop(),
            )
        except asyncio.CancelledError:
            logger.info("apex.shutdown")
        except KeyboardInterrupt:
            logger.info("apex.shutdown_keyboard")
        finally:
            self._running = False
            self.save_state()
            logger.info("apex.state_saved_on_exit")


async def main():
    trader = ApexPaperTrader(bankroll=5000.0)
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
