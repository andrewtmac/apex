"""
APEX TastyTrade Data Ingester

Session-based authentication against the TastyTrade (TastyWorks) API.
Ingests account info, positions, and streaming quotes.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog

from apex.config import ApexConfig
from apex.data.store import FeatureStore
from apex.data.streams import PRICES_TASTYTRADE, StreamPublisher

logger = structlog.get_logger(__name__)

_POLL_INTERVAL = 30  # seconds


class TastyTradeIngester:
    """
    Ingests positions and quotes from the TastyTrade REST API.

    Lifecycle::

        ingester = TastyTradeIngester(config)
        await ingester.start()
        await ingester.stop()
    """

    def __init__(self, config: ApexConfig):
        self._cfg = config.venues.tastytrade
        self._redis_url = config.infra.redis_url
        self._db_url = config.infra.database_url
        self._session_token: Optional[str] = None
        self._accounts: list[dict[str, Any]] = []
        self._publisher: Optional[StreamPublisher] = None
        self._store: Optional[FeatureStore] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._running = False
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _authenticate(self) -> None:
        """Obtain a session token via POST /sessions."""
        assert self._http is not None
        url = f"{self._cfg.base_url}/sessions"
        payload = {
            "login": self._cfg.username,
            "password": self._cfg.password,
        }
        resp = await self._http.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        self._session_token = data["data"]["session-token"]
        logger.info("tastytrade.authenticated", sandbox=self._cfg.sandbox)

    def _auth_headers(self) -> dict[str, str]:
        assert self._session_token is not None
        return {
            "Authorization": self._session_token,
            "Content-Type": "application/json",
        }

    async def _ensure_session(self) -> None:
        """Re-authenticate if the session has expired."""
        if self._session_token is None:
            await self._authenticate()
            return
        # Validate session
        assert self._http is not None
        try:
            resp = await self._http.post(
                f"{self._cfg.base_url}/sessions/validate",
                headers=self._auth_headers(),
            )
            if resp.status_code != 200:
                await self._authenticate()
        except Exception:
            await self._authenticate()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._publisher = StreamPublisher(self._redis_url)
        await self._publisher.connect()

        self._store = FeatureStore(self._redis_url, self._db_url)
        await self._store.connect()

        self._http = httpx.AsyncClient(timeout=httpx.Timeout(15.0))
        self._running = True

        await self._authenticate()
        await self._fetch_accounts()

        self._tasks = [
            asyncio.create_task(self._poll_loop(), name="tt-poll"),
        ]
        logger.info(
            "tastytrade_ingester.started",
            n_accounts=len(self._accounts),
            sandbox=self._cfg.sandbox,
        )

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        # Delete session
        if self._http and self._session_token:
            try:
                await self._http.delete(
                    f"{self._cfg.base_url}/sessions",
                    headers=self._auth_headers(),
                )
            except Exception:
                pass
        if self._http:
            await self._http.aclose()
        if self._publisher:
            await self._publisher.close()
        if self._store:
            await self._store.close()
        logger.info("tastytrade_ingester.stopped")

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    async def _api_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        assert self._http is not None
        await self._ensure_session()
        url = f"{self._cfg.base_url}{path}"
        resp = await self._http.get(url, headers=self._auth_headers(), params=params)
        resp.raise_for_status()
        return resp.json()

    async def _fetch_accounts(self) -> None:
        try:
            data = await self._api_get("/customers/me/accounts")
            items = data.get("data", {}).get("items", [])
            self._accounts = [
                {
                    "account_number": acct.get("account", {}).get("account-number", ""),
                    "nickname": acct.get("account", {}).get("nickname", ""),
                    "account_type": acct.get("account", {}).get("account-type-name", ""),
                }
                for acct in items
            ]
            logger.info("tastytrade.accounts_fetched", accounts=self._accounts)
        except Exception:
            logger.exception("tastytrade.accounts_fetch_failed")

    async def _fetch_positions(self, account_number: str) -> list[dict[str, Any]]:
        try:
            data = await self._api_get(f"/accounts/{account_number}/positions")
            items = data.get("data", {}).get("items", [])
            positions = []
            for item in items:
                pos = item if isinstance(item, dict) else {}
                positions.append({
                    "symbol": pos.get("symbol", ""),
                    "instrument_type": pos.get("instrument-type", ""),
                    "quantity": pos.get("quantity", "0"),
                    "quantity_direction": pos.get("quantity-direction", ""),
                    "close_price": pos.get("close-price", "0"),
                    "average_open_price": pos.get("average-open-price", "0"),
                    "multiplier": pos.get("multiplier", 1),
                })
            return positions
        except Exception:
            logger.warning("tastytrade.positions_fetch_failed", account=account_number)
            return []

    async def _fetch_balances(self, account_number: str) -> dict[str, Any]:
        try:
            data = await self._api_get(f"/accounts/{account_number}/balances")
            bal = data.get("data", {})
            return {
                "cash_balance": bal.get("cash-balance", "0"),
                "net_liquidating_value": bal.get("net-liquidating-value", "0"),
                "equity_buying_power": bal.get("equity-buying-power", "0"),
                "maintenance_excess": bal.get("maintenance-excess", "0"),
            }
        except Exception:
            logger.warning("tastytrade.balances_fetch_failed", account=account_number)
            return {}

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        while self._running:
            await self._poll_once()
            await asyncio.sleep(_POLL_INTERVAL)

    async def _poll_once(self) -> None:
        assert self._publisher is not None and self._store is not None
        now_iso = datetime.now(timezone.utc).isoformat()

        for acct in self._accounts:
            acct_num = acct["account_number"]

            balances = await self._fetch_balances(acct_num)
            positions = await self._fetch_positions(acct_num)

            # Publish account snapshot
            snapshot = {
                "ts": now_iso,
                "venue": "tastytrade",
                "account": acct_num,
                "balances": balances,
                "positions": positions,
                "n_positions": len(positions),
            }
            await self._publisher.publish(PRICES_TASTYTRADE, snapshot)
            await self._store.put(
                entity_id=f"tt:{acct_num}",
                feature_set="account_snapshot",
                features=snapshot,
            )

            # Publish individual positions
            for pos in positions:
                symbol = pos.get("symbol", "unknown")
                tick = {
                    "ts": now_iso,
                    "venue": "tastytrade",
                    "symbol": symbol,
                    "mid": float(pos.get("close_price", 0)),
                    "quantity": float(pos.get("quantity", 0)),
                    "avg_open_price": float(pos.get("average_open_price", 0)),
                }
                await self._store.put_online(f"tt:{symbol}", tick)

        logger.debug("tastytrade.poll_complete", n_accounts=len(self._accounts))
