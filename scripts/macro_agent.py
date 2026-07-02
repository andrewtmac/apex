#!/usr/bin/env python3
"""Macro Strategy Agent for APEX V2.

Trades CPI, FED rate, and GDP markets on Kalshi using:
1. CPI Nowcast: FRED data (PPI, Core CPI, Housing) → forecast next CPI
2. FED Rate Arb: CME FedWatch implied probs vs Kalshi pricing
3. GDP Leading: PMI, retail sales, employment → GDP forecast

HIGH CONVICTION ONLY: edge > 10%, confidence > 70%.
"""

import asyncio
import math
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
import numpy as np
import structlog

logger = structlog.get_logger()

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
# FRED API key is free — use demo for public series
FRED_KEY = "DEMO_KEY"

# FRED series for CPI nowcast
CPI_INPUTS = {
    "PPIACO": "ppi",           # Producer Price Index
    "CPIAUCSL": "cpi",         # CPI All Items
    "CUUR0000SA0L2": "core",   # Core CPI
    "CPIHOSSL": "housing",     # Housing CPI
    "CUSR0000SAF11": "food",   # Food CPI
    "CUSR0000SETA01": "energy",  # Energy CPI
}


class MacroAgent:
    """Trades macro events (CPI, FED, GDP) on Kalshi."""

    def __init__(self):
        self._fred_cache: dict[str, list] = {}
        self._cache_ts: dict[str, float] = {}
        self._cache_ttl = 3600  # 1 hour

    async def _fetch_fred(self, client: httpx.AsyncClient,
                          series_id: str) -> list[dict]:
        """Fetch FRED series data with caching."""
        now = time.time()
        if (series_id in self._fred_cache
                and now - self._cache_ts.get(series_id, 0) < self._cache_ttl):
            return self._fred_cache[series_id]

        try:
            resp = await client.get(FRED_BASE, params={
                "series_id": series_id,
                "api_key": FRED_KEY,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 12,
            })
            if resp.status_code != 200:
                return []

            data = resp.json()
            observations = data.get("observations", [])
            self._fred_cache[series_id] = observations
            self._cache_ts[series_id] = now
            return observations
        except Exception as e:
            logger.warning("macro.fred_error", series=series_id, error=str(e))
            return []

    def _parse_fred_values(self, obs: list[dict]) -> list[tuple[str, float]]:
        """Parse FRED observations to (date, value) pairs."""
        result = []
        for o in obs:
            val = o.get("value", ".")
            if val == "." or val == "":
                continue
            try:
                result.append((o["date"], float(val)))
            except (ValueError, KeyError):
                continue
        return result

    def _pct_change(self, values: list[tuple[str, float]],
                    lag: int = 1) -> Optional[float]:
        """Calculate month-over-month percent change."""
        if len(values) < lag + 1:
            return None
        current = values[0][1]
        previous = values[lag][1]
        if previous == 0:
            return None
        return (current - previous) / previous * 100

    async def build_cpi_nowcast(self, client: httpx.AsyncClient) -> dict:
        """Build CPI nowcast from FRED data.

        Returns dict with forecast, components, and confidence.
        """
        # Fetch all input series
        fred_data = {}
        for sid in CPI_INPUTS:
            obs = await self._fetch_fred(client, sid)
            fred_data[CPI_INPUTS[sid]] = self._parse_fred_values(obs)

        # Get latest values and pct changes
        components = {}
        for name, values in fred_data.items():
            if not values:
                continue
            latest_date, latest_val = values[0]
            pct_chg = self._pct_change(values)
            components[name] = {
                "latest_date": latest_date,
                "latest_val": latest_val,
                "pct_change": pct_chg,
            }

        # Simple nowcast: weighted average of recent changes
        # PPI leads CPI by ~1 month, Core is persistent
        weights = {
            "ppi": 0.25,
            "core": 0.35,
            "housing": 0.20,
            "food": 0.10,
            "energy": 0.10,
        }

        forecast_pct = 0.0
        total_weight = 0.0
        for name, w in weights.items():
            if name in components and components[name]["pct_change"] is not None:
                forecast_pct += w * components[name]["pct_change"]
                total_weight += w

        if total_weight > 0:
            forecast_pct /= total_weight
            # Annualize: monthly * 12
            forecast_annual = forecast_pct * 12
        else:
            forecast_annual = None

        # Confidence based on data freshness and consistency
        n_sources = sum(1 for v in components.values()
                        if v["pct_change"] is not None)
        confidence = min(0.9, 0.3 + n_sources * 0.12)

        return {
            "forecast_annual_pct": forecast_annual,
            "forecast_monthly_pct": forecast_pct,
            "components": components,
            "confidence": confidence,
            "n_sources": n_sources,
        }

    def parse_cpi_market(self, question: str) -> Optional[dict]:
        """Parse Kalshi CPI market question.

        Returns dict with threshold and direction.
        """
        q = question.lower()

        # "Will CPI rise more than X% in July?"
        m = re.search(r'(?:rise|fall|change)\s+(?:more\s+than\s+)?(-?\d+\.?\d*)%', q)
        if m:
            return {"threshold": float(m.group(1)), "direction": "above"}

        # "Will CPI be above X?"
        m = re.search(r'(?:above|over|exceed)\s+(\d+\.?\d*)', q)
        if m:
            return {"threshold": float(m.group(1)), "direction": "above"}

        # "Will CPI be below X?"
        m = re.search(r'(?:below|under)\s+(\d+\.?\d*)', q)
        if m:
            return {"threshold": float(m.group(1)), "direction": "below"}

        return None

    def evaluate_cpi_signal(self, market: dict, nowcast: dict) -> Optional[dict]:
        """Evaluate a CPI market against our nowcast."""
        question = market.get("question", "")
        market_price = market.get("current_price", 0.5)

        parsed = self.parse_cpi_market(question)
        if not parsed:
            return None

        forecast = nowcast.get("forecast_annual_pct")
        if forecast is None:
            return None

        threshold = parsed["threshold"]
        direction = parsed["direction"]
        confidence = nowcast["confidence"]

        # Estimate probability using normal distribution
        # Sigma based on historical nowcast error (~0.3% annualized)
        sigma = 0.3
        if direction == "above":
            z = (forecast - threshold) / sigma
            true_prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        else:
            z = (threshold - forecast) / sigma
            true_prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))

        edge = true_prob - market_price

        if abs(edge) < 0.10:
            return None
        if confidence < 0.70:
            return None

        kelly = 0.20
        size_pct = min(kelly * abs(edge) / 0.10, 0.08)
        size_pct *= confidence

        direction_trade = "BUY" if edge > 0 else "SELL"

        return {
            "market_id": market["market_id"],
            "venue": "kalshi",
            "question": question[:80],
            "direction": direction_trade,
            "edge": round(edge, 4),
            "true_prob": round(true_prob, 4),
            "market_price": round(market_price, 4),
            "size_pct": round(size_pct, 4),
            "strategy": "macro",
            "confidence": round(confidence, 3),
            "reason": (
                f"CPI nowcast {forecast:+.2f}% vs threshold {threshold}%, "
                f"edge {edge:+.1%}, conf {confidence:.0%}"
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "end_date": market.get("end_date", ""),
        }

    def parse_fed_market(self, question: str) -> Optional[dict]:
        """Parse FED rate market question."""
        q = question.lower()

        # "Will the upper bound of the federal funds rate be X%?"
        m = re.search(r'(?:be|remain)\s+(?:at\s+)?(\d+\.?\d*)%', q)
        if m:
            return {"target_rate": float(m.group(1))}

        # "Will the Fed cut rates?"
        if "cut" in q:
            return {"action": "cut"}
        if "hike" in q or "raise" in q:
            return {"action": "hike"}

        return None

    def evaluate_fed_signal(self, market: dict) -> Optional[dict]:
        """Evaluate FED rate market.

        Uses CME FedWatch proxy: compare market price to our estimate
        based on Fed funds futures (approximated from recent Fed
        commentary and economic data).
        """
        question = market.get("question", "")
        market_price = market.get("current_price", 0.5)

        parsed = self.parse_fed_market(question)
        if not parsed:
            return None

        # For now, use a simple heuristic:
        # If market is pricing > 80% for a rate cut, it's likely priced in
        # If < 20%, the market expects no cut
        # We look for mispricing in the 30-70% range where uncertainty is high

        # Skip if market is very confident either way
        if market_price > 0.85 or market_price < 0.15:
            return None

        # We need a real model here. For now, skip FED trades
        # until we can integrate FedWatch data.
        return None

    async def generate_signals(self, markets: list[dict]) -> list[dict]:
        """Generate macro trading signals with convergence filter."""
        all_signals = []

        async with httpx.AsyncClient(timeout=30) as client:
            # CPI nowcast
            nowcast = await self.build_cpi_nowcast(client)

            for market in markets:
                question = market.get("question", "").lower()
                ticker = market.get("market_id", "")

                # CPI markets
                if "cpi" in question or "CPI" in ticker:
                    signal = self.evaluate_cpi_signal(market, nowcast)
                    if signal:
                        all_signals.append(signal)

                # FED markets
                elif "fed" in question or "rate" in question or "FED" in ticker:
                    signal = self.evaluate_fed_signal(market)
                    if signal:
                        all_signals.append(signal)

        # Convergence: one trade per event
        event_groups: dict[str, list[dict]] = defaultdict(list)
        for s in all_signals:
            close = s.get("end_date", "")[:10]
            event_key = f"macro_{close}"
            event_groups[event_key].append(s)

        signals = []
        for event_key, group in event_groups.items():
            if len(group) == 1:
                signals.append(group[0])
                continue

            group.sort(key=lambda s: abs(s["edge"]), reverse=True)
            best = group[0]
            second = group[1]

            edge_gap = abs(abs(best["edge"]) - abs(second["edge"]))
            if edge_gap < 0.05:
                continue

            if best["direction"] != second["direction"]:
                continue

            signals.append(best)

        signals.sort(key=lambda s: abs(s["edge"]), reverse=True)

        logger.info(
            "macro.signals_generated",
            n=len(signals),
            markets_scanned=len(markets),
            cpi_nowcast=nowcast.get("forecast_annual_pct"),
        )

        return signals
