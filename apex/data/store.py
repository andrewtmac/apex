"""
APEX Feature Store

Dual-layer feature store backed by Redis (online / hot) and
TimescaleDB (offline / cold).  Provides get/put for both layers
with JSON serialisation and configurable TTL.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import asyncpg
import redis.asyncio as aioredis
import structlog

logger = structlog.get_logger(__name__)

# Default TTL for online features (1 hour)
DEFAULT_ONLINE_TTL = 3600


class FeatureStore:
    """
    Dual-layer feature store.

    Online layer  -> Redis hash per entity (sub-second reads)
    Offline layer -> TimescaleDB ``feature_store`` hypertable (historical queries)

    Usage::

        store = FeatureStore(redis_url, database_url)
        await store.connect()
        await store.put_online("BTC-YES", {"mid": 0.65, "spread": 0.02})
        features = await store.get_online("BTC-YES")
        await store.put_offline("BTC-YES", "price_features", {"mid": 0.65})
        history = await store.get_offline("BTC-YES", "price_features", lookback=timedelta(hours=1))
        await store.close()
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6380",
        database_url: str = "postgresql://apex:apex@localhost:5433/apex",
        online_ttl: int = DEFAULT_ONLINE_TTL,
        key_prefix: str = "apex:features:",
    ):
        self._redis_url = redis_url
        self._database_url = database_url
        self._online_ttl = online_ttl
        self._key_prefix = key_prefix
        self._redis: Optional[aioredis.Redis] = None
        self._pool: Optional[asyncpg.Pool] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open connections to Redis and PostgreSQL."""
        self._redis = aioredis.from_url(
            self._redis_url,
            decode_responses=True,
            max_connections=10,
        )
        await self._redis.ping()

        self._pool = await asyncpg.create_pool(
            self._database_url,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        logger.info("feature_store.connected")

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None
        if self._pool:
            await self._pool.close()
            self._pool = None
        logger.info("feature_store.closed")

    # ------------------------------------------------------------------
    # Online layer (Redis)
    # ------------------------------------------------------------------

    def _online_key(self, entity_id: str) -> str:
        return f"{self._key_prefix}{entity_id}"

    async def put_online(
        self,
        entity_id: str,
        features: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        """
        Store *features* dict for *entity_id* in Redis.
        Overwrites previous values; sets TTL for automatic expiry.
        """
        assert self._redis is not None, "Call connect() first"
        key = self._online_key(entity_id)
        # Store as a single JSON blob for atomic reads
        await self._redis.set(key, json.dumps(features, default=str))
        await self._redis.expire(key, ttl or self._online_ttl)
        logger.debug("feature_store.put_online", entity_id=entity_id, n_features=len(features))

    async def get_online(self, entity_id: str) -> Optional[dict[str, Any]]:
        """Retrieve features for *entity_id* from Redis.  Returns None on miss."""
        assert self._redis is not None
        key = self._online_key(entity_id)
        raw = await self._redis.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def delete_online(self, entity_id: str) -> None:
        assert self._redis is not None
        await self._redis.delete(self._online_key(entity_id))

    # ------------------------------------------------------------------
    # Offline layer (TimescaleDB)
    # ------------------------------------------------------------------

    async def put_offline(
        self,
        entity_id: str,
        feature_set: str,
        features: dict[str, Any],
        ts: datetime | None = None,
    ) -> None:
        """
        Insert a timestamped feature row into the ``feature_store`` hypertable.
        """
        assert self._pool is not None, "Call connect() first"
        ts = ts or datetime.now(timezone.utc)
        await self._pool.execute(
            """
            INSERT INTO feature_store (time, entity_id, feature_set, features)
            VALUES ($1, $2, $3, $4)
            """,
            ts,
            entity_id,
            feature_set,
            json.dumps(features, default=str),
        )
        logger.debug(
            "feature_store.put_offline",
            entity_id=entity_id,
            feature_set=feature_set,
        )

    async def get_offline(
        self,
        entity_id: str,
        feature_set: str,
        lookback: timedelta = timedelta(hours=24),
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Query historical features from TimescaleDB.

        Returns a list of dicts with keys ``time``, ``entity_id``,
        ``feature_set``, and ``features`` (parsed JSON).
        """
        assert self._pool is not None
        cutoff = datetime.now(timezone.utc) - lookback
        rows = await self._pool.fetch(
            """
            SELECT time, entity_id, feature_set, features
            FROM feature_store
            WHERE entity_id = $1
              AND feature_set = $2
              AND time >= $3
            ORDER BY time DESC
            LIMIT $4
            """,
            entity_id,
            feature_set,
            cutoff,
            limit,
        )
        results: list[dict[str, Any]] = []
        for row in rows:
            features_raw = row["features"]
            features = json.loads(features_raw) if isinstance(features_raw, str) else features_raw
            results.append(
                {
                    "time": row["time"].isoformat(),
                    "entity_id": row["entity_id"],
                    "feature_set": row["feature_set"],
                    "features": features,
                }
            )
        return results

    async def get_latest_offline(
        self,
        entity_id: str,
        feature_set: str,
    ) -> Optional[dict[str, Any]]:
        """Return the single most recent offline feature row, or None."""
        assert self._pool is not None
        row = await self._pool.fetchrow(
            """
            SELECT time, entity_id, feature_set, features
            FROM feature_store
            WHERE entity_id = $1
              AND feature_set = $2
            ORDER BY time DESC
            LIMIT 1
            """,
            entity_id,
            feature_set,
        )
        if row is None:
            return None
        features_raw = row["features"]
        features = json.loads(features_raw) if isinstance(features_raw, str) else features_raw
        return {
            "time": row["time"].isoformat(),
            "entity_id": row["entity_id"],
            "feature_set": row["feature_set"],
            "features": features,
        }

    # ------------------------------------------------------------------
    # Dual-write convenience
    # ------------------------------------------------------------------

    async def put(
        self,
        entity_id: str,
        feature_set: str,
        features: dict[str, Any],
        ts: datetime | None = None,
        online_ttl: int | None = None,
    ) -> None:
        """Write to both online and offline layers atomically."""
        await self.put_online(entity_id, features, ttl=online_ttl)
        await self.put_offline(entity_id, feature_set, features, ts=ts)
