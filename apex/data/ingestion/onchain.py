"""
APEX On-Chain Data Ingester

Monitors the Polygon blockchain for large transactions, whale movements,
and smart-contract events via JSON-RPC.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog

from apex.config import ApexConfig
from apex.data.store import FeatureStore
from apex.data.streams import ONCHAIN_STREAM, StreamPublisher

logger = structlog.get_logger(__name__)

_POLL_INTERVAL = 15  # seconds — poll every ~2 blocks on Polygon
_WHALE_THRESHOLD_MATIC = 100_000  # MATIC  (approx $50k+)
_WHALE_THRESHOLD_USDC = 50_000  # USDC

# Well-known contract addresses on Polygon
_USDC_CONTRACT = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"  # USDC (native) on Polygon
_POLYMARKET_CTF = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"  # Polymarket CTF Exchange


class OnChainIngester:
    """
    Monitors the Polygon blockchain for whale activity and large transfers.

    Uses JSON-RPC via the configured POLYGON_RPC_URL (e.g. Alchemy/Infura).

    Lifecycle::

        ingester = OnChainIngester(config)
        await ingester.start()
        await ingester.stop()
    """

    def __init__(self, config: ApexConfig):
        self._rpc_url = config.data_sources.polygon_rpc_url
        self._redis_url = config.infra.redis_url
        self._db_url = config.infra.database_url
        self._publisher: Optional[StreamPublisher] = None
        self._store: Optional[FeatureStore] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._last_block: int = 0
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._rpc_id = 0

    # ------------------------------------------------------------------
    # JSON-RPC helpers
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        self._rpc_id += 1
        return self._rpc_id

    async def _rpc_call(self, method: str, params: list[Any] | None = None) -> Any:
        """Execute a JSON-RPC call against the Polygon node."""
        assert self._http is not None
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params or [],
        }
        resp = await self._http.post(self._rpc_url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            logger.warning("onchain.rpc_error", method=method, error=data["error"])
            return None
        return data.get("result")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not self._rpc_url:
            logger.warning("onchain_ingester.no_rpc_url, skipping")
            return

        self._publisher = StreamPublisher(self._redis_url)
        await self._publisher.connect()

        self._store = FeatureStore(self._redis_url, self._db_url)
        await self._store.connect()

        self._http = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        self._running = True

        # Get current block number as starting point
        latest = await self._rpc_call("eth_blockNumber")
        if latest:
            self._last_block = int(latest, 16)

        self._tasks = [
            asyncio.create_task(self._poll_loop(), name="onchain-poll"),
        ]
        logger.info(
            "onchain_ingester.started",
            rpc_url=self._rpc_url[:40] + "...",
            start_block=self._last_block,
        )

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
        logger.info("onchain_ingester.stopped")

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        while self._running:
            await self._scan_new_blocks()
            await asyncio.sleep(_POLL_INTERVAL)

    async def _scan_new_blocks(self) -> None:
        """Scan blocks from last_block+1 to latest for whale transactions."""
        latest_hex = await self._rpc_call("eth_blockNumber")
        if not latest_hex:
            return

        latest_block = int(latest_hex, 16)
        if latest_block <= self._last_block:
            return

        # Process up to 5 blocks at a time to avoid falling behind
        end_block = min(self._last_block + 5, latest_block)

        for block_num in range(self._last_block + 1, end_block + 1):
            await self._process_block(block_num)

        self._last_block = end_block

    async def _process_block(self, block_num: int) -> None:
        """Fetch a block and scan transactions for whale activity."""
        block_hex = hex(block_num)
        block_data = await self._rpc_call("eth_getBlockByNumber", [block_hex, True])
        if not block_data:
            return

        transactions = block_data.get("transactions", [])
        timestamp_hex = block_data.get("timestamp", "0x0")
        block_timestamp = int(timestamp_hex, 16)

        for tx in transactions:
            await self._analyze_transaction(tx, block_num, block_timestamp)

    async def _analyze_transaction(
        self,
        tx: dict[str, Any],
        block_num: int,
        block_timestamp: int,
    ) -> None:
        """Check if a transaction is a whale movement worth tracking."""
        assert self._publisher is not None and self._store is not None

        value_hex = tx.get("value", "0x0")
        value_wei = int(value_hex, 16)
        value_matic = value_wei / 1e18

        from_addr = tx.get("from", "").lower()
        to_addr = tx.get("to", "").lower() if tx.get("to") else ""
        tx_hash = tx.get("hash", "")

        # Check for large native MATIC transfers
        if value_matic >= _WHALE_THRESHOLD_MATIC:
            event = self._build_whale_event(
                tx_hash=tx_hash,
                from_addr=from_addr,
                to_addr=to_addr,
                asset="MATIC",
                amount=round(value_matic, 2),
                block_num=block_num,
                block_timestamp=block_timestamp,
            )
            await self._publisher.publish(ONCHAIN_STREAM, event)
            await self._store.put(
                entity_id=f"onchain:{tx_hash}",
                feature_set="whale_tx",
                features=event,
            )
            logger.info(
                "onchain.whale_detected",
                asset="MATIC",
                amount=value_matic,
                tx=tx_hash[:16],
            )

        # Check for interactions with Polymarket contracts
        if to_addr == _POLYMARKET_CTF.lower():
            event = self._build_contract_event(
                tx_hash=tx_hash,
                from_addr=from_addr,
                contract="polymarket_ctf",
                block_num=block_num,
                block_timestamp=block_timestamp,
                input_data=tx.get("input", "")[:20],  # just the function selector
            )
            await self._publisher.publish(ONCHAIN_STREAM, event)
            await self._store.put(
                entity_id=f"onchain:{tx_hash}",
                feature_set="contract_interaction",
                features=event,
            )

        # Check for USDC transfers (ERC-20 transfer topic)
        if to_addr == _USDC_CONTRACT.lower():
            input_data = tx.get("input", "")
            # ERC-20 transfer: function selector 0xa9059cbb
            if input_data.startswith("0xa9059cbb") and len(input_data) >= 138:
                # Decode amount (last 32 bytes)
                try:
                    amount_hex = input_data[-64:]
                    amount_raw = int(amount_hex, 16)
                    amount_usdc = amount_raw / 1e6  # USDC has 6 decimals
                    if amount_usdc >= _WHALE_THRESHOLD_USDC:
                        event = self._build_whale_event(
                            tx_hash=tx_hash,
                            from_addr=from_addr,
                            to_addr=to_addr,
                            asset="USDC",
                            amount=round(amount_usdc, 2),
                            block_num=block_num,
                            block_timestamp=block_timestamp,
                        )
                        await self._publisher.publish(ONCHAIN_STREAM, event)
                        await self._store.put(
                            entity_id=f"onchain:{tx_hash}",
                            feature_set="whale_tx",
                            features=event,
                        )
                        logger.info(
                            "onchain.whale_detected",
                            asset="USDC",
                            amount=amount_usdc,
                            tx=tx_hash[:16],
                        )
                except (ValueError, IndexError):
                    pass

    # ------------------------------------------------------------------
    # Event builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_whale_event(
        tx_hash: str,
        from_addr: str,
        to_addr: str,
        asset: str,
        amount: float,
        block_num: int,
        block_timestamp: int,
    ) -> dict[str, Any]:
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_type": "whale_transfer",
            "tx_hash": tx_hash,
            "from": from_addr,
            "to": to_addr,
            "asset": asset,
            "amount": amount,
            "block_number": block_num,
            "block_timestamp": block_timestamp,
            "chain": "polygon",
        }

    @staticmethod
    def _build_contract_event(
        tx_hash: str,
        from_addr: str,
        contract: str,
        block_num: int,
        block_timestamp: int,
        input_data: str = "",
    ) -> dict[str, Any]:
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_type": "contract_interaction",
            "tx_hash": tx_hash,
            "from": from_addr,
            "contract": contract,
            "function_selector": input_data[:10] if input_data else "",
            "block_number": block_num,
            "block_timestamp": block_timestamp,
            "chain": "polygon",
        }
