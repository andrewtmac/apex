#!/usr/bin/env python3
"""Crypto Strategy Agent for APEX V2.

Analyzes BTC/ETH price action, momentum, and sentiment to generate
signals for Kalshi crypto range markets.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx
import numpy as np
import math as _math
import pandas as pd
import structlog

logger = structlog.get_logger()

# CoinGecko IDs
ASSET_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
}

# Kalshi series tickers
CRYPTO_SERIES = ["KXBTC", "KXETH"]


@dataclass
class CryptoSnapshot:
    """Current state of a crypto asset."""
    asset: str
    price: float
    rsi_14: float
    macd_signal: float  # MACD line - signal line
    bb_position: float  # 0-1 position within Bollinger Bands
    momentum_24h: float  # % change over 24h
    volatility_20d: float  # Annualized volatility
    fear_greed: int  # 0-100
    timestamp: str


class CryptoAgent:
    """Analyzes crypto markets and generates signals for Kalshi."""

    def __init__(self):
        self._price_cache: dict[str, pd.DataFrame] = {}
        self._price_cache_ts: dict[str, float] = {}
        self._fg_cache: Optional[dict] = None
        self._fg_cache_ts: float = 0
        self.snapshots: dict[str, CryptoSnapshot] = {}

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    async def fetch_prices(self, asset: str, days: int = 7) -> pd.DataFrame:
        """Fetch hourly price data from CoinGecko."""
        now = time.time()
        cache_key = f"{asset}_{days}"
        if cache_key in self._price_cache and (now - self._price_cache_ts.get(cache_key, 0)) < 300:
            return self._price_cache[cache_key]

        coin_id = ASSET_MAP.get(asset, "bitcoin")

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
                    params={"vs_currency": "usd", "days": days, "interval": "hourly"},
                )
                resp.raise_for_status()
                data = resp.json()

            prices = data.get("prices", [])
            if not prices:
                return self._price_cache.get(cache_key, pd.DataFrame())

            df = pd.DataFrame(prices, columns=["timestamp", "price"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp")

            self._price_cache[cache_key] = df
            self._price_cache_ts[cache_key] = now

            logger.debug("crypto.prices_fetched", asset=asset, points=len(df))
            return df

        except Exception as e:
            logger.warning("crypto.price_fetch_failed", asset=asset, error=str(e))
            return self._price_cache.get(cache_key, pd.DataFrame())

    async def fetch_fear_greed(self) -> dict:
        """Fetch the Fear & Greed Index from alternative.me."""
        now = time.time()
        if self._fg_cache and (now - self._fg_cache_ts) < 3600:
            return self._fg_cache

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get("https://api.alternative.me/fng/?limit=1")
                resp.raise_for_status()
                data = resp.json()["data"][0]

            self._fg_cache = {
                "value": int(data["value"]),
                "classification": data["value_classification"],
            }
            self._fg_cache_ts = now
            return self._fg_cache

        except Exception as e:
            logger.warning("crypto.fg_fetch_failed", error=str(e))
            return {"value": 50, "classification": "Neutral"}

    # ------------------------------------------------------------------
    # Technical indicators
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0

        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

    @staticmethod
    def calculate_macd(prices: pd.Series) -> tuple[float, float]:
        """Calculate MACD line and signal. Returns (macd_line, signal_line)."""
        if len(prices) < 35:
            return 0.0, 0.0

        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])

    @staticmethod
    def calculate_bollinger(prices: pd.Series, period: int = 20) -> tuple[float, float, float]:
        """Calculate Bollinger Bands. Returns (upper, middle, lower)."""
        if len(prices) < period:
            p = float(prices.iloc[-1])
            return p, p, p

        middle = prices.rolling(window=period).mean().iloc[-1]
        std = prices.rolling(window=period).std().iloc[-1]
        return float(middle + 2 * std), float(middle), float(middle - 2 * std)

    def calculate_indicators(self, prices: pd.DataFrame) -> dict:
        """Calculate all technical indicators from price data."""
        if prices.empty:
            return {}

        p = prices["price"]

        rsi = self.calculate_rsi(p)
        macd_line, macd_signal = self.calculate_macd(p)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger(p)

        current = float(p.iloc[-1])

        # BB position: 0 = at lower band, 1 = at upper band
        bb_range = bb_upper - bb_lower
        bb_position = (current - bb_lower) / bb_range if bb_range > 0 else 0.5

        # 24h momentum
        if len(p) >= 24:
            price_24h_ago = float(p.iloc[-24])
            momentum_24h = (current - price_24h_ago) / price_24h_ago * 100
        else:
            momentum_24h = 0.0

        # 20-day volatility (annualized)
        returns = p.pct_change().dropna()
        if len(returns) >= 20:
            volatility_20d = float(returns.tail(20).std() * np.sqrt(365 * 24) * 100)
        else:
            volatility_20d = float(returns.std() * np.sqrt(365 * 24) * 100) if len(returns) > 0 else 50.0

        return {
            "current_price": current,
            "rsi": rsi,
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_histogram": macd_line - macd_signal,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
            "bb_position": bb_position,
            "momentum_24h": momentum_24h,
            "volatility_20d": volatility_20d,
        }

    # ------------------------------------------------------------------
    # Probability estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_move_probability(
        current_price: float,
        target_price: float,
        hours: float,
        daily_volatility_pct: float,
    ) -> float:
        """Estimate probability of price reaching target using GBM.

        Uses geometric Brownian motion to estimate the probability that
        the price will be above (or below) the target at expiry.
        """
        if current_price <= 0 or target_price <= 0:
            return 0.5

        # Convert daily volatility to hourly
        hourly_vol = daily_volatility_pct / 100 / np.sqrt(24)

        # Log return needed
        log_return = np.log(target_price / current_price)

        # Standard deviation of log return over the time period
        std = hourly_vol * np.sqrt(hours)

        if std <= 0:
            return 0.5

        # P(S_T > target) = P(Z > (log_return) / std) for driftless GBM
        # With slight upward drift assumption for crypto
        z = -log_return / std  # negative because we want P(S > target)

        # Standard normal CDF
        prob = 0.5 * (1 + _math.erf(z / _math.sqrt(2)))

        return np.clip(prob, 0.01, 0.99)

    def calculate_crypto_probability(
        self, current_price: float, threshold: float, direction: str,
        indicators: dict, fear_greed: dict, hours_to_expiry: float = 24
    ) -> float:
        """Estimate probability that crypto price will be above/below threshold.

        Combines:
        1. GBM-based distance probability
        2. Momentum adjustment (RSI, MACD)
        3. Sentiment adjustment (Fear & Greed)
        """
        # Base probability from distance
        base_prob = self.estimate_move_probability(
            current_price, threshold, hours_to_expiry,
            indicators.get("volatility_20d", 50)
        )

        if direction == "below":
            base_prob = 1 - base_prob

        # Momentum adjustment
        rsi = indicators.get("rsi", 50)
        macd_hist = indicators.get("macd_histogram", 0)
        momentum = indicators.get("momentum_24h", 0)

        # RSI: oversold (<30) = bullish, overbought (>70) = bearish
        rsi_adj = 0.0
        if rsi < 30:
            rsi_adj = 0.05  # Bullish
        elif rsi > 70:
            rsi_adj = -0.05  # Bearish

        # MACD: positive histogram = bullish
        macd_adj = np.clip(macd_hist / current_price * 100, -0.03, 0.03)

        # Momentum: strong moves tend to continue (momentum)
        momentum_adj = np.clip(momentum / 100, -0.03, 0.03)

        # Fear & Greed: extreme values are contrarian
        fg_value = fear_greed.get("value", 50)
        fg_adj = 0.0
        if fg_value < 20:  # Extreme fear = contrarian bullish
            fg_adj = 0.04
        elif fg_value > 80:  # Extreme greed = contrarian bearish
            fg_adj = -0.04

        # Combine
        adjusted = base_prob + rsi_adj + macd_adj + momentum_adj + fg_adj

        # For "above" direction, adjustments push probability up for bullish signals
        # For "below" direction, reverse
        if direction == "below":
            adjusted = base_prob - (rsi_adj + macd_adj + momentum_adj + fg_adj)

        return np.clip(adjusted, 0.02, 0.98)

    # ------------------------------------------------------------------
    # Market parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_crypto_market(question: str) -> Optional[dict]:
        """Parse a Kalshi crypto market question.

        Returns {asset, threshold, direction} or None.
        """
        import re
        q = question.lower()

        # Identify asset
        asset = None
        if "btc" in q or "bitcoin" in q:
            asset = "BTC"
        elif "eth" in q or "ethereum" in q:
            asset = "ETH"

        if not asset:
            return None

        # Extract price threshold
        # Patterns: "above $110,000", "below $3,500", "> $100k"
        price_match = re.search(r'\$?([\d,]+(?:\.\d+)?)\s*k?', q)
        if not price_match:
            return None

        price_str = price_match.group(1).replace(",", "")
        threshold = float(price_str)

        # Handle "k" suffix
        if "k" in q[price_match.start():price_match.end() + 2]:
            threshold *= 1000

        # Direction
        if any(w in q for w in ["above", "over", ">", "higher"]):
            direction = "above"
        elif any(w in q for w in ["below", "under", "<", "lower"]):
            direction = "below"
        else:
            direction = "above"  # Default

        return {"asset": asset, "threshold": threshold, "direction": direction}

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    async def build_snapshot(self, asset: str) -> Optional[CryptoSnapshot]:
        """Build a full snapshot for a crypto asset."""
        prices = await self.fetch_prices(asset, days=7)
        if prices.empty:
            return None

        indicators = self.calculate_indicators(prices)
        fg = await self.fetch_fear_greed()

        return CryptoSnapshot(
            asset=asset,
            price=indicators["current_price"],
            rsi_14=indicators["rsi"],
            macd_signal=indicators["macd_histogram"],
            bb_position=indicators["bb_position"],
            momentum_24h=indicators["momentum_24h"],
            volatility_20d=indicators["volatility_20d"],
            fear_greed=fg["value"],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def evaluate_crypto_signal(
        self, market: dict, indicators: dict, fear_greed: dict,
        current_price: float
    ) -> Optional[dict]:
        """Evaluate a single Kalshi crypto market."""
        question = market.get("question", "")
        market_price = market.get("current_price", 0.5)

        parsed = self.parse_crypto_market(question)
        if parsed is None:
            return None

        threshold = parsed["threshold"]
        direction = parsed["direction"]

        # Estimate hours to expiry
        end_date = market.get("end_date", "")
        if end_date:
            try:
                expiry = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                hours_to_expiry = max((expiry - datetime.now(timezone.utc)).total_seconds() / 3600, 1)
            except Exception:
                hours_to_expiry = 24
        else:
            hours_to_expiry = 24

        # Calculate true probability
        true_prob = self.calculate_crypto_probability(
            current_price, threshold, direction,
            indicators, fear_greed, hours_to_expiry
        )

        # Calculate edge
        edge = true_prob - market_price

        if abs(edge) < 0.05:  # Minimum 5% edge for crypto (higher than weather)
            return None

        direction_trade = "BUY" if edge > 0 else "SELL"

        # Position sizing
        kelly_fraction = 0.20  # More conservative for crypto
        edge_abs = abs(edge)
        size_pct = min(kelly_fraction * edge_abs / 0.10, 0.08)

        # Reduce sizing in high volatility
        vol = indicators.get("volatility_20d", 50)
        if vol > 80:
            size_pct *= 0.6
        elif vol > 60:
            size_pct *= 0.8

        return {
            "market_id": market["market_id"],
            "venue": "kalshi",
            "question": question[:80],
            "direction": direction_trade,
            "edge": round(edge, 4),
            "true_prob": round(true_prob, 4),
            "market_price": round(market_price, 4),
            "size_pct": round(size_pct, 4),
            "strategy": "crypto",
            "confidence": round(1 - vol / 200, 3),  # Lower confidence in high vol
            "current_price": current_price,
            "threshold": threshold,
            "rsi": round(indicators.get("rsi", 50), 1),
            "fear_greed": fear_greed.get("value", 50),
            "reason": (
                f"{parsed['asset']} @ ${current_price:,.0f} vs "
                f"${threshold:,.0f} ({direction}), "
                f"RSI={indicators.get('rsi', 50):.0f}, "
                f"F&G={fear_greed.get('value', 50)}, "
                f"edge {edge:+.1%}"
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "end_date": market.get("end_date", ""),
        }

    async def generate_signals(self, markets: list[dict]) -> list[dict]:
        """Generate trading signals for Kalshi crypto markets.

        CONVERGENCE RULE: For each event (asset + expiry), only ONE
        trade is emitted — the single best edge. If multiple buckets
        have similar edge, that means we're uncertain and should skip.
        """
        all_signals = []

        # Fetch data for both assets
        for asset in ["BTC", "ETH"]:
            prices = await self.fetch_prices(asset, days=7)
            if prices.empty:
                continue

            indicators = self.calculate_indicators(prices)
            fg = await self.fetch_fear_greed()
            current_price = indicators["current_price"]

            # Update snapshot
            self.snapshots[asset] = CryptoSnapshot(
                asset=asset,
                price=current_price,
                rsi_14=indicators["rsi"],
                macd_signal=indicators["macd_histogram"],
                bb_position=indicators["bb_position"],
                momentum_24h=indicators["momentum_24h"],
                volatility_20d=indicators["volatility_20d"],
                fear_greed=fg["value"],
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            # Evaluate each market for this asset
            for market in markets:
                ticker = market.get("market_id", "")
                question = market.get("question", "").lower()

                is_match = False
                if asset == "BTC" and ("btc" in ticker.lower() or "bitcoin" in question):
                    is_match = True
                elif asset == "ETH" and ("eth" in ticker.lower() or "ethereum" in question):
                    is_match = True

                if not is_match:
                    continue

                signal = await self.evaluate_crypto_signal(
                    market, indicators, fg, current_price
                )
                if signal is not None:
                    signal["size_usd"] = 0
                    all_signals.append(signal)

        # CONVERGENCE: Group by event (asset + close_time) and pick
        # only the single best trade per event.
        from collections import defaultdict
        event_groups: dict[str, list[dict]] = defaultdict(list)
        for s in all_signals:
            # Group key: asset + event date + close time
            close = s.get("end_date", "")[:16]
            question = s.get("question", "").lower()
            asset_key = "btc" if "btc" in question or "bitcoin" in question else "eth"
            event_key = f"{asset_key}_{close}"
            event_groups[event_key].append(s)

        signals = []
        for event_key, group in event_groups.items():
            if len(group) == 1:
                # Only one signal for this event — take it
                signals.append(group[0])
                continue

            # Multiple signals for same event = CONFLICT
            # Sort by absolute edge descending
            group.sort(key=lambda s: abs(s["edge"]), reverse=True)
            best = group[0]
            second = group[1]

            # CONVERGENCE CHECK: If top 2 signals have similar edge
            # (< 5% apart), we're uncertain — skip the event entirely.
            edge_gap = abs(abs(best["edge"]) - abs(second["edge"]))
            if edge_gap < 0.05:
                logger.info(
                    "crypto.convergence_skip",
                    event_key=event_key,
                    n_signals=len(group),
                    best_edge=best["edge"],
                    second_edge=second["edge"],
                    gap=edge_gap,
                )
                continue

            # If best and second-best point in OPPOSITE directions,
            # we're deeply uncertain — skip.
            if best["direction"] != second["direction"]:
                logger.info(
                    "crypto.conflict_skip",
                    event_key=event_key,
                    best_dir=best["direction"],
                    second_dir=second["direction"],
                )
                continue

            # Only take the best signal
            signals.append(best)

        signals.sort(key=lambda s: abs(s["edge"]), reverse=True)

        logger.info(
            "crypto.signals_generated",
            n=len(signals),
            markets_scanned=len(markets),
            events_filtered=len(event_groups),
        )

        return signals
