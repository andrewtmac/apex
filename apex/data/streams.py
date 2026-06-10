"""
APEX Redis Streams Pub/Sub Bus

Provides StreamPublisher / StreamConsumer wrappers around Redis Streams
with typed JSON serialisation, consumer groups, and health checks.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import redis.asyncio as aioredis
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Stream name constants
# ---------------------------------------------------------------------------

# Price streams
PRICES_POLYMARKET = "apex:prices:polymarket"
PRICES_KALSHI = "apex:prices:kalshi"
PRICES_TASTYTRADE = "apex:prices:tastytrade"

# Orderbook streams
ORDERBOOK_POLYMARKET = "apex:orderbook:polymarket"
ORDERBOOK_KALSHI = "apex:orderbook:kalshi"

# Alternative data streams
NEWS_STREAM = "apex:news"
WEATHER_STREAM = "apex:weather"
SPORTS_STREAM = "apex:sports"
SOCIAL_STREAM = "apex:social"
ONCHAIN_STREAM = "apex:onchain"

# Internal streams
SIGNALS_STREAM = "apex:signals"
TRADES_STREAM = "apex:trades"
RISK_STREAM = "apex:risk"
QUALITY_STREAM = "apex:quality"

ALL_STREAMS: list[str] = [
    PRICES_POLYMARKET,
    PRICES_KALSHI,
    PRICES_TASTYTRADE,
    ORDERBOOK_POLYMARKET,
    ORDERBOOK_KALSHI,
    NEWS_STREAM,
    WEATHER_STREAM,
    SPORTS_STREAM,
    SOCIAL_STREAM,
    ONCHAIN_STREAM,
    SIGNALS_STREAM,
    TRADES_STREAM,
    RISK_STREAM,
    QUALITY_STREAM,
]

# Default max stream length (trim older entries)
DEFAULT_MAXLEN = 50_000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize(msg: Any) -> dict[str, str]:
    """Convert a message (dict or dataclass) to a flat {key: str} mapping for Redis."""
    if hasattr(msg, "__dataclass_fields__"):
        payload = asdict(msg)  # type: ignore[arg-type]
    elif isinstance(msg, dict):
        payload = msg
    else:
        payload = {"data": str(msg)}
    return {"payload": json.dumps(payload, default=str)}


def _deserialize(raw: dict[bytes, bytes]) -> dict[str, Any]:
    """Inverse of _serialize."""
    payload_bytes = raw.get(b"payload", b"{}")
    return json.loads(payload_bytes)


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------


class StreamPublisher:
    """
    Publishes typed JSON messages to named Redis Streams.

    Usage::

        pub = StreamPublisher(redis_url)
        await pub.connect()
        await pub.publish(PRICES_POLYMARKET, {"symbol": "BTC-YES", "mid": 0.65})
        await pub.close()
    """

    def __init__(self, redis_url: str = "redis://localhost:6380", maxlen: int = DEFAULT_MAXLEN):
        self._redis_url = redis_url
        self._maxlen = maxlen
        self._redis: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        self._redis = aioredis.from_url(
            self._redis_url,
            decode_responses=False,
            max_connections=10,
        )
        await self._redis.ping()
        logger.info("stream_publisher.connected", redis_url=self._redis_url)

    async def publish(self, stream: str, message: Any) -> str:
        """Publish *message* to *stream*. Returns the message ID."""
        assert self._redis is not None, "Call connect() first"
        fields = _serialize(message)
        # XADD with approximate MAXLEN trimming
        msg_id: bytes = await self._redis.xadd(
            stream,
            fields,  # type: ignore[arg-type]
            maxlen=self._maxlen,
            approximate=True,
        )
        return msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            logger.info("stream_publisher.closed")


# ---------------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------------


@dataclass
class StreamMessage:
    stream: str
    msg_id: str
    data: dict[str, Any]
    raw_id: bytes = b""


class StreamConsumer:
    """
    Consumes messages from one or more Redis Streams via consumer groups.

    Usage::

        consumer = StreamConsumer(redis_url, group="pipeline", consumer="worker-1")
        await consumer.connect()
        await consumer.ensure_groups([PRICES_POLYMARKET, PRICES_KALSHI])
        async for msg in consumer.listen([PRICES_POLYMARKET, PRICES_KALSHI]):
            process(msg)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6380",
        group: str = "apex",
        consumer: str = "worker-0",
        batch_size: int = 100,
        block_ms: int = 2000,
    ):
        self._redis_url = redis_url
        self._group = group
        self._consumer = consumer
        self._batch_size = batch_size
        self._block_ms = block_ms
        self._redis: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        self._redis = aioredis.from_url(
            self._redis_url,
            decode_responses=False,
            max_connections=10,
        )
        await self._redis.ping()
        logger.info(
            "stream_consumer.connected",
            group=self._group,
            consumer=self._consumer,
        )

    async def ensure_groups(self, streams: Sequence[str]) -> None:
        """Create consumer groups (idempotent)."""
        assert self._redis is not None
        for stream in streams:
            try:
                await self._redis.xgroup_create(
                    stream,
                    self._group,
                    id="0",
                    mkstream=True,
                )
                logger.info("stream_consumer.group_created", stream=stream, group=self._group)
            except aioredis.ResponseError as exc:
                if "BUSYGROUP" in str(exc):
                    pass  # group already exists
                else:
                    raise

    async def listen(
        self,
        streams: Sequence[str],
        start_id: str = ">",
    ):
        """
        Async generator that yields StreamMessage objects.

        *start_id* = ">" means only new messages; "0" replays pending.
        """
        assert self._redis is not None
        stream_ids = {s: start_id for s in streams}

        while True:
            results = await self._redis.xreadgroup(
                groupname=self._group,
                consumername=self._consumer,
                streams=stream_ids,  # type: ignore[arg-type]
                count=self._batch_size,
                block=self._block_ms,
            )
            if not results:
                continue

            for stream_bytes, messages in results:
                stream_name = (
                    stream_bytes.decode() if isinstance(stream_bytes, bytes) else stream_bytes
                )
                for msg_id_bytes, fields in messages:
                    msg_id = (
                        msg_id_bytes.decode() if isinstance(msg_id_bytes, bytes) else msg_id_bytes
                    )
                    data = _deserialize(fields)
                    yield StreamMessage(
                        stream=stream_name,
                        msg_id=msg_id,
                        data=data,
                        raw_id=msg_id_bytes,
                    )
                    # Auto-ack
                    await self._redis.xack(stream_name, self._group, msg_id_bytes)

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            logger.info("stream_consumer.closed")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


async def stream_health(redis_url: str, streams: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Return a dict mapping stream names to health info:
    ``{"length": int, "last_entry_ts": float | None, "lag_seconds": float | None}``.
    """
    streams = streams or ALL_STREAMS
    r = aioredis.from_url(redis_url, decode_responses=False)
    report: dict[str, Any] = {}
    now = time.time()

    try:
        for stream in streams:
            try:
                length = await r.xlen(stream)
            except Exception:
                report[stream] = {"length": 0, "last_entry_ts": None, "lag_seconds": None}
                continue

            last_ts: float | None = None
            lag: float | None = None
            if length > 0:
                # XREVRANGE returns newest first
                entries = await r.xrevrange(stream, count=1)
                if entries:
                    entry_id = entries[0][0]
                    # Redis stream IDs are "<ms>-<seq>"
                    ts_ms = int(entry_id.decode().split("-")[0]) if isinstance(entry_id, bytes) else int(str(entry_id).split("-")[0])
                    last_ts = ts_ms / 1000.0
                    lag = now - last_ts

            report[stream] = {
                "length": length,
                "last_entry_ts": last_ts,
                "lag_seconds": round(lag, 2) if lag is not None else None,
            }
    finally:
        await r.aclose()

    return report
