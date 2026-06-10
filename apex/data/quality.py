"""
APEX Data Quality Monitor

Monitors staleness, anomalies, and cross-source consistency across
all Redis Streams.  Generates health reports for the monitoring dashboard.
"""

from __future__ import annotations

import asyncio
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import redis.asyncio as aioredis
import structlog

from apex.data.streams import (
    ALL_STREAMS,
    PRICES_KALSHI,
    PRICES_POLYMARKET,
    QUALITY_STREAM,
    StreamPublisher,
    stream_health,
)

logger = structlog.get_logger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    DOWN = "DOWN"


# Staleness thresholds per stream (seconds)
_STALENESS_THRESHOLDS: dict[str, float] = {
    "apex:prices:polymarket": 120,
    "apex:prices:kalshi": 120,
    "apex:prices:tastytrade": 120,
    "apex:orderbook:polymarket": 120,
    "apex:orderbook:kalshi": 120,
    "apex:news": 300,
    "apex:weather": 600,
    "apex:sports": 300,
    "apex:social": 300,
    "apex:onchain": 120,
    "apex:signals": 600,
    "apex:trades": 3600,
    "apex:risk": 600,
    "apex:quality": 600,
}

_DEFAULT_STALENESS = 300  # 5 minutes

# Z-score threshold for price anomaly detection
_ZSCORE_THRESHOLD = 3.5

# Window size for rolling statistics
_PRICE_WINDOW = 100


@dataclass
class StreamHealth:
    """Health status for a single stream."""

    stream: str
    status: HealthStatus
    length: int = 0
    last_entry_ts: Optional[float] = None
    lag_seconds: Optional[float] = None
    staleness_threshold: float = _DEFAULT_STALENESS
    message: str = ""


@dataclass
class PriceAnomaly:
    """Detected price anomaly."""

    symbol: str
    venue: str
    price: float
    mean: float
    std: float
    zscore: float
    ts: str = ""


@dataclass
class HealthReport:
    """Aggregated health report across all streams."""

    ts: str
    overall_status: HealthStatus
    streams: list[StreamHealth] = field(default_factory=list)
    anomalies: list[PriceAnomaly] = field(default_factory=list)
    cross_source_issues: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""


class DataQualityMonitor:
    """
    Monitors data quality across all APEX data streams.

    Checks:
    1. Stream staleness (is data flowing?)
    2. Price anomalies (Z-score outlier detection)
    3. Cross-source consistency (do venues agree?)

    Lifecycle::

        monitor = DataQualityMonitor(redis_url)
        await monitor.start()
        report = await monitor.generate_report()
        await monitor.stop()
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6380",
        check_interval: float = 60.0,
    ):
        self._redis_url = redis_url
        self._check_interval = check_interval
        self._redis: Optional[aioredis.Redis] = None
        self._publisher: Optional[StreamPublisher] = None
        self._running = False
        self._tasks: list[asyncio.Task] = []

        # Rolling price windows for anomaly detection: {symbol: deque of prices}
        self._price_windows: dict[str, deque[float]] = {}

        # Cross-source price cache: {symbol: {venue: latest_mid}}
        self._cross_source_cache: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._redis = aioredis.from_url(self._redis_url, decode_responses=False)
        await self._redis.ping()

        self._publisher = StreamPublisher(self._redis_url)
        await self._publisher.connect()

        self._running = True
        self._tasks = [
            asyncio.create_task(self._monitor_loop(), name="quality-monitor"),
        ]
        logger.info("data_quality_monitor.started")

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._redis:
            await self._redis.aclose()
        if self._publisher:
            await self._publisher.close()
        logger.info("data_quality_monitor.stopped")

    # ------------------------------------------------------------------
    # Monitor loop
    # ------------------------------------------------------------------

    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                report = await self.generate_report()
                # Publish report to quality stream
                if self._publisher:
                    await self._publisher.publish(QUALITY_STREAM, {
                        "ts": report.ts,
                        "overall_status": report.overall_status.value,
                        "n_healthy": sum(
                            1 for s in report.streams if s.status == HealthStatus.HEALTHY
                        ),
                        "n_degraded": sum(
                            1 for s in report.streams if s.status == HealthStatus.DEGRADED
                        ),
                        "n_stale": sum(
                            1 for s in report.streams if s.status == HealthStatus.STALE
                        ),
                        "n_down": sum(
                            1 for s in report.streams if s.status == HealthStatus.DOWN
                        ),
                        "n_anomalies": len(report.anomalies),
                        "n_cross_source_issues": len(report.cross_source_issues),
                        "summary": report.summary,
                    })
                logger.info(
                    "quality.report",
                    status=report.overall_status.value,
                    summary=report.summary,
                )
            except Exception:
                logger.exception("quality.monitor_error")

            await asyncio.sleep(self._check_interval)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    async def generate_report(self) -> HealthReport:
        """Generate a comprehensive data quality report."""
        now = datetime.now(timezone.utc)

        # 1. Check stream staleness
        stream_statuses = await self._check_staleness()

        # 2. Check price anomalies
        anomalies = await self._check_price_anomalies()

        # 3. Check cross-source consistency
        cross_issues = await self._check_cross_source_consistency()

        # Compute overall status
        statuses = [s.status for s in stream_statuses]
        if any(s == HealthStatus.DOWN for s in statuses):
            overall = HealthStatus.DOWN
        elif any(s == HealthStatus.STALE for s in statuses) or anomalies:
            overall = HealthStatus.DEGRADED
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        healthy = sum(1 for s in statuses if s == HealthStatus.HEALTHY)
        total = len(statuses)
        summary = (
            f"{healthy}/{total} streams healthy, "
            f"{len(anomalies)} anomalies, "
            f"{len(cross_issues)} cross-source issues"
        )

        return HealthReport(
            ts=now.isoformat(),
            overall_status=overall,
            streams=stream_statuses,
            anomalies=anomalies,
            cross_source_issues=cross_issues,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Staleness check
    # ------------------------------------------------------------------

    async def _check_staleness(self) -> list[StreamHealth]:
        """Check each stream for staleness based on last entry timestamp."""
        health_data = await stream_health(self._redis_url, ALL_STREAMS)
        results: list[StreamHealth] = []
        now = time.time()

        for stream in ALL_STREAMS:
            info = health_data.get(stream, {})
            length = info.get("length", 0)
            last_ts = info.get("last_entry_ts")
            lag = info.get("lag_seconds")
            threshold = _STALENESS_THRESHOLDS.get(stream, _DEFAULT_STALENESS)

            if length == 0:
                status = HealthStatus.DOWN
                message = "Stream is empty (no data ever published)"
            elif lag is None:
                status = HealthStatus.DOWN
                message = "Cannot determine last entry timestamp"
            elif lag > threshold * 2:
                status = HealthStatus.STALE
                message = f"Data is {lag:.0f}s old (threshold: {threshold}s)"
            elif lag > threshold:
                status = HealthStatus.DEGRADED
                message = f"Data is aging: {lag:.0f}s (threshold: {threshold}s)"
            else:
                status = HealthStatus.HEALTHY
                message = f"OK ({lag:.0f}s lag, {length} entries)"

            results.append(StreamHealth(
                stream=stream,
                status=status,
                length=length,
                last_entry_ts=last_ts,
                lag_seconds=lag,
                staleness_threshold=threshold,
                message=message,
            ))

        return results

    # ------------------------------------------------------------------
    # Price anomaly detection
    # ------------------------------------------------------------------

    async def _check_price_anomalies(self) -> list[PriceAnomaly]:
        """
        Detect price anomalies using Z-score against a rolling window.

        Reads the latest entries from price streams and checks if any
        deviate significantly from the rolling mean.
        """
        anomalies: list[PriceAnomaly] = []
        assert self._redis is not None

        for stream, venue in [
            (PRICES_POLYMARKET, "polymarket"),
            (PRICES_KALSHI, "kalshi"),
        ]:
            try:
                # Read last 10 entries
                entries = await self._redis.xrevrange(stream, count=10)
                for _entry_id, fields in entries:
                    import json
                    payload_raw = fields.get(b"payload", b"{}")
                    data = json.loads(payload_raw)

                    symbol = data.get("symbol", "unknown")
                    mid = data.get("mid")
                    if mid is None:
                        continue
                    mid = float(mid)

                    key = f"{venue}:{symbol}"

                    # Update rolling window
                    if key not in self._price_windows:
                        self._price_windows[key] = deque(maxlen=_PRICE_WINDOW)
                    window = self._price_windows[key]
                    window.append(mid)

                    # Update cross-source cache
                    self._cross_source_cache.setdefault(symbol, {})[venue] = mid

                    # Need at least 10 observations for meaningful Z-score
                    if len(window) < 10:
                        continue

                    mean = statistics.mean(window)
                    std = statistics.stdev(window)
                    if std < 1e-9:
                        continue

                    zscore = abs((mid - mean) / std)
                    if zscore >= _ZSCORE_THRESHOLD:
                        anomaly = PriceAnomaly(
                            symbol=symbol,
                            venue=venue,
                            price=mid,
                            mean=round(mean, 6),
                            std=round(std, 6),
                            zscore=round(zscore, 2),
                            ts=data.get("ts", ""),
                        )
                        anomalies.append(anomaly)
                        logger.warning(
                            "quality.price_anomaly",
                            symbol=symbol,
                            venue=venue,
                            zscore=zscore,
                            price=mid,
                        )

            except Exception:
                logger.debug("quality.anomaly_check_failed", stream=stream)

        return anomalies

    # ------------------------------------------------------------------
    # Cross-source consistency
    # ------------------------------------------------------------------

    async def _check_cross_source_consistency(self) -> list[dict[str, Any]]:
        """
        Compare prices for the same underlying across venues.

        If a market is listed on both Polymarket and Kalshi, their prices
        should be reasonably close.  Large deviations may indicate stale
        data or an arbitrage opportunity.
        """
        issues: list[dict[str, Any]] = []
        max_divergence = 0.15  # 15 percentage points

        for symbol, venue_prices in self._cross_source_cache.items():
            if len(venue_prices) < 2:
                continue

            prices = list(venue_prices.values())
            venues = list(venue_prices.keys())

            for i in range(len(prices)):
                for j in range(i + 1, len(prices)):
                    divergence = abs(prices[i] - prices[j])
                    if divergence > max_divergence:
                        issues.append({
                            "symbol": symbol,
                            "venue_a": venues[i],
                            "price_a": round(prices[i], 4),
                            "venue_b": venues[j],
                            "price_b": round(prices[j], 4),
                            "divergence": round(divergence, 4),
                            "ts": datetime.now(timezone.utc).isoformat(),
                        })
                        logger.warning(
                            "quality.cross_source_divergence",
                            symbol=symbol,
                            divergence=divergence,
                        )

        return issues
