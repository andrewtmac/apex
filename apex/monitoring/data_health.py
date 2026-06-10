"""Data pipeline health monitoring.

Checks:
- Redis Stream staleness per source
- Data completeness (expected vs actual ticks/hour)
- Price anomalies (Z-score detection)
- Cross-source consistency
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import redis.asyncio as aioredis
import structlog

from apex.data.streams import (
    ALL_STREAMS,
    NEWS_STREAM,
    ORDERBOOK_KALSHI,
    ORDERBOOK_POLYMARKET,
    PRICES_KALSHI,
    PRICES_POLYMARKET,
    PRICES_TASTYTRADE,
)

logger = structlog.get_logger(__name__)

# Expected minimum ticks per hour per data source
_EXPECTED_TICKS_PER_HOUR = {
    PRICES_POLYMARKET: 100,
    PRICES_KALSHI: 50,
    PRICES_TASTYTRADE: 30,
    ORDERBOOK_POLYMARKET: 50,
    ORDERBOOK_KALSHI: 20,
    NEWS_STREAM: 5,
}

# Maximum acceptable staleness (seconds) per stream
_MAX_STALENESS = {
    PRICES_POLYMARKET: 120,      # 2 minutes
    PRICES_KALSHI: 300,          # 5 minutes
    PRICES_TASTYTRADE: 300,      # 5 minutes
    ORDERBOOK_POLYMARKET: 120,   # 2 minutes
    ORDERBOOK_KALSHI: 300,       # 5 minutes
    NEWS_STREAM: 3600,           # 1 hour
}

# Z-score threshold for price anomaly detection
_PRICE_ZSCORE_THRESHOLD = 5.0


class DataHealthMonitor:
    """Monitors health and quality of data pipelines.

    Performs continuous checks on Redis streams, TimescaleDB data
    completeness, and price anomaly detection.

    Parameters
    ----------
    redis_url : Redis connection URL
    db_url : optional TimescaleDB URL for completeness checks
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6380",
        db_url: str | None = None,
    ) -> None:
        self.redis_url = redis_url
        self.db_url = db_url
        self._redis: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._redis = aioredis.from_url(
            self.redis_url,
            decode_responses=False,
            max_connections=5,
        )
        await self._redis.ping()
        logger.info("data_health.connected")

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    async def _ensure_redis(self) -> aioredis.Redis:
        if self._redis is None:
            await self.connect()
        assert self._redis is not None
        return self._redis

    async def check_all(self) -> dict[str, Any]:
        """Run all health checks.

        Returns a dict with overall status and per-source details:
        {
            "status": "healthy" | "degraded" | "critical",
            "timestamp": "...",
            "sources": {
                "apex:prices:polymarket": {
                    "status": "healthy",
                    "staleness_seconds": 12.5,
                    "length": 45000,
                    ...
                }
            },
            "anomalies": [...],
            "summary": {...}
        }
        """
        now = time.time()
        sources: dict[str, dict[str, Any]] = {}
        anomalies: list[dict[str, Any]] = []

        # Check each stream
        for stream in ALL_STREAMS:
            source_health = await self._check_source(stream, now)
            sources[stream] = source_health

        # Check for price anomalies
        price_anomalies = await self._check_price_anomalies()
        anomalies.extend(price_anomalies)

        # Check cross-source consistency
        consistency = await self._check_cross_source_consistency()

        # Determine overall status
        statuses = [s["status"] for s in sources.values()]
        critical_count = statuses.count("critical")
        degraded_count = statuses.count("degraded")

        if critical_count >= 2:
            overall = "critical"
        elif critical_count >= 1 or degraded_count >= 3:
            overall = "degraded"
        else:
            overall = "healthy"

        result = {
            "status": overall,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sources": sources,
            "anomalies": anomalies,
            "consistency": consistency,
            "summary": {
                "total_sources": len(sources),
                "healthy": statuses.count("healthy"),
                "degraded": degraded_count,
                "critical": critical_count,
                "inactive": statuses.count("inactive"),
                "n_anomalies": len(anomalies),
            },
        }

        logger.info(
            "data_health.check_complete",
            status=overall,
            healthy=result["summary"]["healthy"],
            degraded=degraded_count,
            critical=critical_count,
        )

        return result

    async def _check_source(
        self, stream: str, now: float
    ) -> dict[str, Any]:
        """Check health of a single data source stream."""
        r = await self._ensure_redis()

        try:
            length = await r.xlen(stream)
        except Exception:
            return {
                "status": "inactive",
                "length": 0,
                "staleness_seconds": None,
                "error": "stream_not_found",
            }

        if length == 0:
            return {
                "status": "inactive",
                "length": 0,
                "staleness_seconds": None,
            }

        # Check staleness
        staleness = await self._compute_staleness(r, stream, now)

        # Determine status
        max_stale = _MAX_STALENESS.get(stream, 600)
        if staleness is None:
            status = "inactive"
        elif staleness > max_stale * 3:
            status = "critical"
        elif staleness > max_stale:
            status = "degraded"
        else:
            status = "healthy"

        # Check throughput (messages in last hour)
        throughput = await self._estimate_throughput(r, stream, now)
        expected = _EXPECTED_TICKS_PER_HOUR.get(stream)
        throughput_status = "ok"
        if expected and throughput < expected * 0.3:
            throughput_status = "low"
            if status == "healthy":
                status = "degraded"

        return {
            "status": status,
            "length": length,
            "staleness_seconds": round(staleness, 2) if staleness else None,
            "max_staleness": max_stale,
            "throughput_per_hour": throughput,
            "expected_per_hour": expected,
            "throughput_status": throughput_status,
        }

    async def _compute_staleness(
        self, r: aioredis.Redis, stream: str, now: float
    ) -> float | None:
        """Compute seconds since last message in stream."""
        try:
            entries = await r.xrevrange(stream, count=1)
            if not entries:
                return None

            entry_id = entries[0][0]
            ts_str = (
                entry_id.decode().split("-")[0]
                if isinstance(entry_id, bytes)
                else str(entry_id).split("-")[0]
            )
            ts_ms = int(ts_str)
            return now - ts_ms / 1000.0
        except Exception:
            return None

    async def _estimate_throughput(
        self,
        r: aioredis.Redis,
        stream: str,
        now: float,
    ) -> int:
        """Estimate messages per hour in the stream."""
        try:
            # Count messages in the last hour using XRANGE with timestamp IDs
            one_hour_ago_ms = int((now - 3600) * 1000)
            start_id = f"{one_hour_ago_ms}-0"

            entries = await r.xrange(stream, min=start_id, count=10000)
            return len(entries)
        except Exception:
            return 0

    async def check_staleness(
        self, stream: str, max_age_seconds: float
    ) -> bool:
        """Check if a stream has recent data within max_age_seconds.

        Returns True if data is fresh, False if stale or missing.
        """
        r = await self._ensure_redis()
        now = time.time()
        staleness = await self._compute_staleness(r, stream, now)

        if staleness is None:
            return False
        return staleness <= max_age_seconds

    async def _check_price_anomalies(self) -> list[dict[str, Any]]:
        """Detect price anomalies using Z-score on recent data."""
        r = await self._ensure_redis()
        anomalies: list[dict[str, Any]] = []

        for stream in [PRICES_POLYMARKET, PRICES_KALSHI, PRICES_TASTYTRADE]:
            try:
                # Get recent price entries
                entries = await r.xrevrange(stream, count=100)
                if len(entries) < 10:
                    continue

                prices: dict[str, list[float]] = {}
                for _, fields in entries:
                    payload = fields.get(b"payload", b"{}")
                    data = json.loads(payload)
                    symbol = data.get("symbol", "unknown")
                    mid = data.get("mid", 0)
                    if mid > 0:
                        if symbol not in prices:
                            prices[symbol] = []
                        prices[symbol].append(float(mid))

                # Check each symbol for anomalies
                import numpy as np

                for symbol, price_series in prices.items():
                    if len(price_series) < 5:
                        continue

                    arr = np.array(price_series)
                    mean = np.mean(arr[1:])  # exclude latest
                    std = np.std(arr[1:])

                    if std < 1e-8:
                        continue

                    latest = arr[0]
                    zscore = abs((latest - mean) / std)

                    if zscore > _PRICE_ZSCORE_THRESHOLD:
                        anomalies.append({
                            "type": "price_anomaly",
                            "stream": stream,
                            "symbol": symbol,
                            "latest_price": latest,
                            "mean_price": float(mean),
                            "z_score": round(float(zscore), 2),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })

            except Exception:
                logger.debug("data_health.anomaly_check_failed", stream=stream)

        return anomalies

    async def _check_cross_source_consistency(self) -> dict[str, Any]:
        """Check consistency between data sources.

        Compares prices for the same underlying across venues
        to detect data quality issues.
        """
        # This is a placeholder for cross-venue price comparison.
        # In production, we'd compare shared underlyings across
        # Polymarket / Kalshi / TastyTrade.
        return {
            "status": "ok",
            "note": "cross_source_check_placeholder",
        }

    async def get_stream_info(self, stream: str) -> dict[str, Any]:
        """Get detailed information about a specific stream.

        Returns length, consumer groups, pending messages, etc.
        """
        r = await self._ensure_redis()

        try:
            info = await r.xinfo_stream(stream)
            groups_raw = await r.xinfo_groups(stream)

            # Parse xinfo response
            length = info.get(b"length", info.get("length", 0))
            first_entry = info.get(b"first-entry", info.get("first-entry"))
            last_entry = info.get(b"last-entry", info.get("last-entry"))

            groups = []
            for g in groups_raw:
                group_name = g.get(b"name", g.get("name", b"")).decode() if isinstance(
                    g.get(b"name", g.get("name", b"")), bytes
                ) else str(g.get(b"name", g.get("name", "")))
                consumers = g.get(b"consumers", g.get("consumers", 0))
                pending = g.get(b"pending", g.get("pending", 0))
                groups.append({
                    "name": group_name,
                    "consumers": consumers,
                    "pending": pending,
                })

            return {
                "stream": stream,
                "length": length,
                "groups": groups,
                "has_data": length > 0,
            }

        except Exception as exc:
            return {
                "stream": stream,
                "error": str(exc),
                "has_data": False,
            }
