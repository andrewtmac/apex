"""
APEX Kalshi Data Ingester

Authenticates via RSA-PSS signed requests and ingests events, markets,
and orderbooks from the Kalshi v2 API.  Publishes to Redis Streams.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from apex.config import ApexConfig
from apex.data.store import FeatureStore
from apex.data.streams import (
    ORDERBOOK_KALSHI,
    PRICES_KALSHI,
    StreamPublisher,
)

logger = structlog.get_logger(__name__)

_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
_POLL_INTERVAL = 30  # seconds
_MARKET_SCAN_INTERVAL = 120


class KalshiIngester:
    """
    Ingests market data from the Kalshi v2 REST API.

    Lifecycle::

        ingester = KalshiIngester(config)
        await ingester.start()
        await ingester.stop()
    """

    def __init__(self, config: ApexConfig):
        self._api_key = config.venues.kalshi.api_key
        self._private_key_path = config.venues.kalshi.private_key_path
        self._redis_url = config.infra.redis_url
        self._db_url = config.infra.database_url
        self._private_key: Optional[rsa.RSAPrivateKey] = None
        self._publisher: Optional[StreamPublisher] = None
        self._store: Optional[FeatureStore] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._active_markets: dict[str, dict[str, Any]] = {}  # ticker -> meta
        self._running = False
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------
    # RSA-PSS Authentication
    # ------------------------------------------------------------------

    def _load_private_key(self) -> None:
        """Load the RSA private key from the PEM file."""
        pem_path = Path(self._private_key_path)
        if not pem_path.exists():
            logger.error("kalshi.private_key_not_found", path=str(pem_path))
            raise FileNotFoundError(f"Kalshi private key not found: {pem_path}")
        pem_data = pem_path.read_bytes()
        self._private_key = serialization.load_pem_private_key(pem_data, password=None)  # type: ignore[assignment]

    def _sign_request(self, method: str, path: str, timestamp_ms: int) -> str:
        """
        Generate RSA-PSS signature for Kalshi API authentication.

        Signature = RSA-PSS(SHA256, timestamp + method + path)
        """
        assert self._private_key is not None
        message = f"{timestamp_ms}{method}{path}".encode()
        signature = self._private_key.sign(  # type: ignore[union-attr]
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        """Build authentication headers for a Kalshi API request."""
        ts_ms = int(time.time() * 1000)
        signature = self._sign_request(method, path, ts_ms)
        return {
            "KALSHI-ACCESS-KEY": self._api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": str(ts_ms),
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._load_private_key()

        self._publisher = StreamPublisher(self._redis_url)
        await self._publisher.connect()

        self._store = FeatureStore(self._redis_url, self._db_url)
        await self._store.connect()

        self._http = httpx.AsyncClient(timeout=httpx.Timeout(15.0))
        self._running = True

        await self._scan_markets()

        self._tasks = [
            asyncio.create_task(self._market_scan_loop(), name="kalshi-scan"),
            asyncio.create_task(self._poll_loop(), name="kalshi-poll"),
        ]
        logger.info("kalshi_ingester.started", n_markets=len(self._active_markets))

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
        logger.info("kalshi_ingester.stopped")

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------

    async def _api_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Authenticated GET request to Kalshi API."""
        assert self._http is not None
        url = f"{_BASE_URL}{path}"
        headers = self._auth_headers("GET", path)
        resp = await self._http.get(url, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()

    async def _scan_markets(self) -> None:
        """Discover active markets."""
        try:
            data = await self._api_get("/events", params={"limit": 100, "status": "open"})
            events = data.get("events", [])
            for event in events:
                for market in event.get("markets", []):
                    ticker = market.get("ticker", "")
                    if ticker:
                        self._active_markets[ticker] = {
                            "event_ticker": event.get("event_ticker", ""),
                            "title": market.get("title", ""),
                            "ticker": ticker,
                            "category": event.get("category", ""),
                            "status": market.get("status", ""),
                        }

            # Also fetch markets directly
            mkt_data = await self._api_get("/markets", params={"limit": 200, "status": "open"})
            for market in mkt_data.get("markets", []):
                ticker = market.get("ticker", "")
                if ticker and ticker not in self._active_markets:
                    self._active_markets[ticker] = {
                        "event_ticker": market.get("event_ticker", ""),
                        "title": market.get("title", ""),
                        "ticker": ticker,
                        "category": market.get("category", ""),
                        "status": market.get("status", ""),
                    }

            logger.info("kalshi.markets_scanned", n_markets=len(self._active_markets))
        except Exception:
            logger.exception("kalshi.market_scan_failed")

    async def _market_scan_loop(self) -> None:
        while self._running:
            await asyncio.sleep(_MARKET_SCAN_INTERVAL)
            await self._scan_markets()

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Poll prices and orderbooks for active markets."""
        while self._running:
            await self._snapshot_all()
            await asyncio.sleep(_POLL_INTERVAL)

    async def _snapshot_all(self) -> None:
        assert self._publisher is not None and self._store is not None

        for ticker, meta in list(self._active_markets.items()):
            try:
                book = await self._fetch_orderbook(ticker)
                if book:
                    tick = self._build_tick(ticker, meta, book)
                    await self._publisher.publish(PRICES_KALSHI, tick)
                    await self._store.put(
                        entity_id=ticker,
                        feature_set="price_tick",
                        features=tick,
                    )

                    snapshot = self._build_orderbook(ticker, meta, book)
                    await self._publisher.publish(ORDERBOOK_KALSHI, snapshot)

            except Exception:
                logger.warning("kalshi.snapshot_failed", ticker=ticker)

            await asyncio.sleep(0.15)  # rate-limit

    async def _fetch_orderbook(self, ticker: str) -> Optional[dict[str, Any]]:
        try:
            return await self._api_get(f"/markets/{ticker}/orderbook")
        except Exception:
            logger.warning("kalshi.orderbook_fetch_failed", ticker=ticker)
            return None

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tick(
        ticker: str,
        meta: dict[str, Any],
        book: dict[str, Any],
    ) -> dict[str, Any]:
        ob = book.get("orderbook", book)
        yes_bids = ob.get("yes", ob.get("bids", []))
        yes_asks = ob.get("no", ob.get("asks", []))

        # Kalshi prices are in cents (0-100)
        best_bid = yes_bids[0][0] / 100 if yes_bids else 0
        best_ask = (100 - (yes_asks[0][0] if yes_asks else 100)) / 100 if yes_asks else 1
        mid = (best_bid + best_ask) / 2

        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "venue": "kalshi",
            "symbol": ticker,
            "title": meta.get("title", ""),
            "category": meta.get("category", ""),
            "mid": round(mid, 4),
            "bid": round(best_bid, 4),
            "ask": round(best_ask, 4),
        }

    @staticmethod
    def _build_orderbook(
        ticker: str,
        meta: dict[str, Any],
        book: dict[str, Any],
    ) -> dict[str, Any]:
        ob = book.get("orderbook", book)
        yes_side = ob.get("yes", ob.get("bids", []))[:5]
        no_side = ob.get("no", ob.get("asks", []))[:5]

        bid_vol = sum(level[1] for level in yes_side) if yes_side else 0
        ask_vol = sum(level[1] for level in no_side) if no_side else 0
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0

        best_bid = yes_side[0][0] / 100 if yes_side else 0
        best_ask = (100 - no_side[0][0]) / 100 if no_side else 1
        spread = best_ask - best_bid

        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "venue": "kalshi",
            "symbol": ticker,
            "bids": yes_side,
            "asks": no_side,
            "spread": round(spread, 4),
            "imbalance": round(imbalance, 4),
        }
