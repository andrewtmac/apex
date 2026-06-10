"""
APEX Polymarket Data Ingester

Connects to the Polymarket CLOB REST + WebSocket APIs to ingest
market metadata, price ticks, and orderbook snapshots.  Publishes
everything to Redis Streams and persists ticks in TimescaleDB.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog
import websockets
import websockets.exceptions

from apex.config import ApexConfig
from apex.data.store import FeatureStore
from apex.data.streams import (
    ORDERBOOK_POLYMARKET,
    PRICES_POLYMARKET,
    StreamPublisher,
)

logger = structlog.get_logger(__name__)

# Reconnection constants
_INITIAL_BACKOFF = 1.0
_MAX_BACKOFF = 60.0
_MARKET_SCAN_INTERVAL = 120  # seconds


class PolymarketIngester:
    """
    Real-time data collector for Polymarket.

    Lifecycle::

        ingester = PolymarketIngester(config)
        await ingester.start()   # runs forever
        await ingester.stop()
    """

    def __init__(self, config: ApexConfig):
        self._cfg = config.venues.polymarket
        self._redis_url = config.infra.redis_url
        self._db_url = config.infra.database_url
        self._publisher: Optional[StreamPublisher] = None
        self._store: Optional[FeatureStore] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._active_tokens: dict[str, dict[str, Any]] = {}  # token_id -> market meta
        self._running = False
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise connections and launch background loops."""
        self._publisher = StreamPublisher(self._redis_url)
        await self._publisher.connect()

        self._store = FeatureStore(self._redis_url, self._db_url)
        await self._store.connect()

        self._http = httpx.AsyncClient(
            base_url=self._cfg.clob_url,
            timeout=httpx.Timeout(15.0),
            headers={"Authorization": f"Bearer {self._cfg.api_key}"},
        )

        self._running = True

        # Initial market scan
        await self._scan_markets()

        # Launch long-running tasks
        self._tasks = [
            asyncio.create_task(self._market_scan_loop(), name="poly-scan"),
            asyncio.create_task(self._ws_loop(), name="poly-ws"),
        ]
        logger.info("polymarket_ingester.started", n_tokens=len(self._active_tokens))

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._http:
            await self._http.aclose()
        if self._publisher:
            await self._publisher.close()
        if self._store:
            await self._store.close()
        logger.info("polymarket_ingester.stopped")

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------

    async def _scan_markets(self) -> None:
        """Discover active markets from the CLOB REST API."""
        assert self._http is not None
        try:
            resp = await self._http.get("/markets", params={"limit": 200, "active": True})
            resp.raise_for_status()
            markets = resp.json()

            # The CLOB /markets endpoint returns a list of market objects.
            # Each market may have multiple tokens (YES / NO).
            for market in markets if isinstance(markets, list) else markets.get("data", []):
                tokens = market.get("tokens", [])
                for token in tokens:
                    token_id = token.get("token_id")
                    if token_id:
                        self._active_tokens[token_id] = {
                            "market_id": market.get("condition_id", market.get("id", "")),
                            "question": market.get("question", ""),
                            "outcome": token.get("outcome", ""),
                            "token_id": token_id,
                        }

            logger.info("polymarket.markets_scanned", n_tokens=len(self._active_tokens))
        except Exception:
            logger.exception("polymarket.market_scan_failed")

    async def _market_scan_loop(self) -> None:
        """Periodically re-scan for new / removed markets."""
        while self._running:
            await asyncio.sleep(_MARKET_SCAN_INTERVAL)
            await self._scan_markets()

    # ------------------------------------------------------------------
    # REST snapshots
    # ------------------------------------------------------------------

    async def _fetch_price(self, token_id: str) -> Optional[dict[str, Any]]:
        assert self._http is not None
        try:
            resp = await self._http.get("/price", params={"token_id": token_id})
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.warning("polymarket.price_fetch_failed", token_id=token_id)
            return None

    async def _fetch_orderbook(self, token_id: str) -> Optional[dict[str, Any]]:
        assert self._http is not None
        try:
            resp = await self._http.get("/book", params={"token_id": token_id})
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.warning("polymarket.orderbook_fetch_failed", token_id=token_id)
            return None

    async def snapshot_all(self) -> None:
        """Take a REST snapshot of prices and orderbooks for all active tokens."""
        assert self._publisher is not None and self._store is not None
        for token_id, meta in list(self._active_tokens.items()):
            price_data = await self._fetch_price(token_id)
            if price_data:
                tick = self._build_tick(token_id, meta, price_data)
                await self._publisher.publish(PRICES_POLYMARKET, tick)
                await self._store.put(
                    entity_id=token_id,
                    feature_set="price_tick",
                    features=tick,
                )

            book_data = await self._fetch_orderbook(token_id)
            if book_data:
                snapshot = self._build_orderbook(token_id, meta, book_data)
                await self._publisher.publish(ORDERBOOK_POLYMARKET, snapshot)

            # Small delay to be polite to the API
            await asyncio.sleep(0.1)

    # ------------------------------------------------------------------
    # WebSocket real-time feed
    # ------------------------------------------------------------------

    async def _ws_loop(self) -> None:
        """Connect to the Polymarket WebSocket and stream price updates."""
        backoff = _INITIAL_BACKOFF

        while self._running:
            try:
                async with websockets.connect(
                    self._cfg.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    logger.info("polymarket.ws_connected")
                    backoff = _INITIAL_BACKOFF

                    # Subscribe to all active tokens
                    await self._ws_subscribe(ws)

                    async for raw_msg in ws:
                        if not self._running:
                            break
                        await self._handle_ws_message(raw_msg)

            except websockets.exceptions.ConnectionClosed as exc:
                logger.warning("polymarket.ws_closed", code=exc.code, reason=exc.reason)
            except Exception:
                logger.exception("polymarket.ws_error")

            if not self._running:
                break

            logger.info("polymarket.ws_reconnecting", backoff=backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF)

    async def _ws_subscribe(self, ws: Any) -> None:
        """Send subscription messages for active tokens."""
        token_ids = list(self._active_tokens.keys())
        if not token_ids:
            return
        # Polymarket WS expects a subscribe message per asset
        for token_id in token_ids:
            sub_msg = json.dumps({
                "type": "subscribe",
                "channel": "market",
                "assets_ids": [token_id],
            })
            await ws.send(sub_msg)
        logger.info("polymarket.ws_subscribed", n_tokens=len(token_ids))

    async def _handle_ws_message(self, raw: str | bytes) -> None:
        """Parse and publish a WebSocket message."""
        assert self._publisher is not None and self._store is not None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = data.get("type", data.get("event_type", ""))

        if msg_type in ("price_change", "book", "trade"):
            token_id = data.get("asset_id", data.get("market", ""))
            meta = self._active_tokens.get(token_id, {"token_id": token_id})

            if msg_type == "price_change":
                tick = self._build_tick(token_id, meta, data)
                await self._publisher.publish(PRICES_POLYMARKET, tick)
                await self._store.put(
                    entity_id=token_id,
                    feature_set="price_tick",
                    features=tick,
                )

            elif msg_type == "book":
                snapshot = self._build_orderbook(token_id, meta, data)
                await self._publisher.publish(ORDERBOOK_POLYMARKET, snapshot)

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tick(
        token_id: str,
        meta: dict[str, Any],
        price_data: dict[str, Any],
    ) -> dict[str, Any]:
        mid = float(price_data.get("price", price_data.get("mid", 0)))
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "venue": "polymarket",
            "symbol": token_id,
            "market_id": meta.get("market_id", ""),
            "question": meta.get("question", ""),
            "outcome": meta.get("outcome", ""),
            "mid": mid,
            "bid": float(price_data.get("bid", mid)),
            "ask": float(price_data.get("ask", mid)),
            "volume": float(price_data.get("volume", 0)),
        }

    @staticmethod
    def _build_orderbook(
        token_id: str,
        meta: dict[str, Any],
        book_data: dict[str, Any],
    ) -> dict[str, Any]:
        bids = book_data.get("bids", [])[:5]
        asks = book_data.get("asks", [])[:5]

        best_bid = float(bids[0]["price"]) if bids else 0
        best_ask = float(asks[0]["price"]) if asks else 1
        spread = best_ask - best_bid

        bid_vol = sum(float(b.get("size", 0)) for b in bids)
        ask_vol = sum(float(a.get("size", 0)) for a in asks)
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0

        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "venue": "polymarket",
            "symbol": token_id,
            "market_id": meta.get("market_id", ""),
            "bids": bids,
            "asks": asks,
            "spread": round(spread, 6),
            "imbalance": round(imbalance, 4),
        }
