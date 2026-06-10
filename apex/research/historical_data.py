"""Collect historical resolved market data from Polymarket and Kalshi APIs.

Data sources:
- Polymarket Gamma API: https://gamma-api.polymarket.com
- Kalshi API: historical events and resolutions

Collects resolved markets, normalizes them, and stores in TimescaleDB
for model training.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# API endpoints
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"
KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2"


class HistoricalDataCollector:
    """Collects and normalizes historical market data from prediction venues.

    All data is normalized to a common schema:
    - market_id: unique identifier
    - venue: source platform
    - question: market question text
    - category: market category
    - outcome: 1 (YES) or 0 (NO)
    - resolution_date: when the market was resolved
    - final_price: last traded price before resolution
    - volume: total volume traded
    - open_date: when the market opened
    - duration_hours: market lifetime
    """

    def __init__(
        self,
        polymarket_api_key: str = "",
        kalshi_api_key: str = "",
        timeout: float = 30.0,
    ) -> None:
        self._poly_key = polymarket_api_key
        self._kalshi_key = kalshi_api_key
        self._timeout = timeout

    async def collect_polymarket(
        self,
        limit: int = 50000,
        batch_size: int = 100,
    ) -> pd.DataFrame:
        """Fetch resolved markets from Polymarket Gamma API.

        The Gamma API provides historical market data including
        resolved outcomes.  Paginates through all resolved markets.

        Parameters
        ----------
        limit : maximum total markets to fetch
        batch_size : markets per API call
        """
        all_markets: list[dict[str, Any]] = []

        async with httpx.AsyncClient(
            base_url=POLYMARKET_GAMMA_URL,
            timeout=httpx.Timeout(self._timeout),
        ) as client:
            offset = 0
            while offset < limit:
                try:
                    resp = await client.get(
                        "/markets",
                        params={
                            "limit": batch_size,
                            "offset": offset,
                            "closed": True,
                            "order": "end_date_iso",
                            "ascending": False,
                        },
                    )
                    resp.raise_for_status()
                    markets = resp.json()

                    if not markets:
                        break

                    for m in markets:
                        # Only include resolved markets with outcomes
                        if m.get("outcome") is None and m.get("resolved_by") is None:
                            continue

                        # Parse outcome from Polymarket format
                        outcome = self._parse_poly_outcome(m)
                        if outcome is None:
                            continue

                        record = {
                            "market_id": m.get("condition_id", m.get("id", "")),
                            "venue": "polymarket",
                            "question": m.get("question", ""),
                            "category": m.get("category", "other"),
                            "outcome": outcome,
                            "resolution_date": m.get("end_date_iso", ""),
                            "final_price": self._safe_float(
                                m.get("outcomePrices", "0.5")
                            ),
                            "volume": self._safe_float(m.get("volume", 0)),
                            "open_date": m.get("start_date_iso", ""),
                            "liquidity": self._safe_float(m.get("liquidity", 0)),
                            "num_traders": self._safe_float(
                                m.get("competitive_traders", 0)
                            ),
                        }
                        all_markets.append(record)

                    offset += batch_size
                    logger.info(
                        "historical.polymarket_batch",
                        offset=offset,
                        total_collected=len(all_markets),
                    )

                    # Rate limiting
                    await asyncio.sleep(0.2)

                except httpx.HTTPStatusError as exc:
                    logger.warning(
                        "historical.polymarket_error",
                        status=exc.response.status_code,
                        offset=offset,
                    )
                    if exc.response.status_code == 429:
                        await asyncio.sleep(5.0)
                    else:
                        break
                except Exception:
                    logger.exception("historical.polymarket_batch_failed")
                    break

        df = pd.DataFrame(all_markets)
        logger.info("historical.polymarket_complete", n_markets=len(df))
        return df

    async def collect_kalshi(
        self,
        limit: int = 10000,
        batch_size: int = 100,
    ) -> pd.DataFrame:
        """Fetch resolved events from Kalshi API.

        Parameters
        ----------
        limit : maximum total events to fetch
        batch_size : events per API call
        """
        all_events: list[dict[str, Any]] = []

        headers = {}
        if self._kalshi_key:
            headers["Authorization"] = f"Bearer {self._kalshi_key}"

        async with httpx.AsyncClient(
            base_url=KALSHI_API_URL,
            timeout=httpx.Timeout(self._timeout),
            headers=headers,
        ) as client:
            cursor: str | None = None
            fetched = 0

            while fetched < limit:
                try:
                    params: dict[str, Any] = {
                        "limit": batch_size,
                        "status": "settled",
                    }
                    if cursor:
                        params["cursor"] = cursor

                    resp = await client.get("/events", params=params)
                    resp.raise_for_status()
                    data = resp.json()

                    events = data.get("events", [])
                    if not events:
                        break

                    for event in events:
                        markets = event.get("markets", [])
                        for m in markets:
                            result = m.get("result", m.get("yes_sub_title", ""))
                            outcome = self._parse_kalshi_outcome(m)
                            if outcome is None:
                                continue

                            record = {
                                "market_id": m.get("ticker", ""),
                                "venue": "kalshi",
                                "question": m.get("title", event.get("title", "")),
                                "category": event.get("category", "other"),
                                "outcome": outcome,
                                "resolution_date": m.get(
                                    "close_time",
                                    m.get("expiration_time", ""),
                                ),
                                "final_price": self._safe_float(
                                    m.get("last_price", 50)
                                ) / 100.0,  # Kalshi prices in cents
                                "volume": self._safe_float(m.get("volume", 0)),
                                "open_date": m.get(
                                    "open_time",
                                    m.get("open_date", ""),
                                ),
                                "liquidity": 0.0,
                                "num_traders": 0.0,
                            }
                            all_events.append(record)

                    cursor = data.get("cursor")
                    fetched += len(events)

                    logger.info(
                        "historical.kalshi_batch",
                        fetched=fetched,
                        total_collected=len(all_events),
                    )
                    await asyncio.sleep(0.3)

                except httpx.HTTPStatusError as exc:
                    logger.warning(
                        "historical.kalshi_error",
                        status=exc.response.status_code,
                    )
                    if exc.response.status_code == 429:
                        await asyncio.sleep(10.0)
                    else:
                        break
                except Exception:
                    logger.exception("historical.kalshi_batch_failed")
                    break

        df = pd.DataFrame(all_events)
        logger.info("historical.kalshi_complete", n_events=len(df))
        return df

    async def build_training_dataset(self) -> pd.DataFrame:
        """Combine all sources into a labeled training dataset.

        Returns a unified DataFrame with normalized columns ready for
        feature engineering and model training.
        """
        logger.info("historical.building_dataset")

        # Collect from all sources concurrently
        poly_task = asyncio.create_task(self.collect_polymarket())
        kalshi_task = asyncio.create_task(self.collect_kalshi())

        poly_df, kalshi_df = await asyncio.gather(poly_task, kalshi_task)

        # Combine
        frames = [df for df in [poly_df, kalshi_df] if not df.empty]
        if not frames:
            raise ValueError("No data collected from any source")

        combined = pd.concat(frames, ignore_index=True)

        # Clean and normalize
        combined = self._normalize_dataset(combined)

        logger.info(
            "historical.dataset_built",
            n_total=len(combined),
            n_polymarket=len(poly_df),
            n_kalshi=len(kalshi_df),
            outcome_distribution=combined["outcome"].value_counts().to_dict(),
        )

        return combined

    def _normalize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize the combined dataset."""
        # Parse dates
        for col in ["resolution_date", "open_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

        # Compute duration
        if "open_date" in df.columns and "resolution_date" in df.columns:
            df["duration_hours"] = (
                (df["resolution_date"] - df["open_date"]).dt.total_seconds() / 3600
            ).fillna(0)
        else:
            df["duration_hours"] = 0.0

        # Ensure outcome is integer
        df["outcome"] = df["outcome"].astype(int)

        # Drop duplicates
        df = df.drop_duplicates(subset=["market_id", "venue"], keep="last")

        # Drop rows with missing critical fields
        df = df.dropna(subset=["market_id", "outcome"])

        # Sort by resolution date
        if "resolution_date" in df.columns:
            df = df.sort_values("resolution_date").reset_index(drop=True)

        return df

    @staticmethod
    def _parse_poly_outcome(market: dict) -> int | None:
        """Parse outcome from Polymarket market data."""
        outcome = market.get("outcome")
        if outcome == "Yes" or outcome == "YES" or outcome == 1:
            return 1
        elif outcome == "No" or outcome == "NO" or outcome == 0:
            return 0

        # Try parsing from resolution data
        resolved = market.get("resolved_by")
        if resolved:
            return 1 if resolved == "yes" else 0

        return None

    @staticmethod
    def _parse_kalshi_outcome(market: dict) -> int | None:
        """Parse outcome from Kalshi market data."""
        result = market.get("result")
        if result == "yes":
            return 1
        elif result == "no":
            return 0

        # Try from yes_sub_title
        sub = market.get("yes_sub_title", "")
        if sub.lower() in ("yes", "over", "above"):
            return 1
        elif sub.lower() in ("no", "under", "below"):
            return 0

        return None

    @staticmethod
    def _safe_float(val: Any) -> float:
        """Safely convert a value to float."""
        if val is None:
            return 0.0
        try:
            if isinstance(val, str):
                # Handle Polymarket's JSON-encoded price arrays
                if val.startswith("["):
                    import json

                    prices = json.loads(val)
                    return float(prices[0]) if prices else 0.0
                return float(val)
            return float(val)
        except (ValueError, TypeError, IndexError):
            return 0.0


async def store_to_database(
    df: pd.DataFrame,
    db_url: str,
) -> int:
    """Store collected data into TimescaleDB markets table.

    Returns the number of rows inserted.
    """
    import asyncpg

    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
    inserted = 0

    try:
        async with pool.acquire() as conn:
            for _, row in df.iterrows():
                try:
                    await conn.execute(
                        """
                        INSERT INTO markets (id, venue, symbol, title, category,
                                           resolution_date, status, outcome, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, 'resolved', $7, $8)
                        ON CONFLICT (id) DO UPDATE SET
                            outcome = EXCLUDED.outcome,
                            status = EXCLUDED.status,
                            updated_at = NOW()
                        """,
                        str(row["market_id"]),
                        str(row["venue"]),
                        str(row["market_id"]),
                        str(row.get("question", "")),
                        str(row.get("category", "other")),
                        row.get("resolution_date"),
                        int(row["outcome"]),
                        json.dumps({
                            "volume": float(row.get("volume", 0)),
                            "final_price": float(row.get("final_price", 0)),
                            "duration_hours": float(row.get("duration_hours", 0)),
                        }),
                    )
                    inserted += 1
                except Exception:
                    logger.debug(
                        "historical.insert_failed",
                        market_id=row.get("market_id"),
                    )
    finally:
        await pool.close()

    import json

    logger.info("historical.stored", n_inserted=inserted)
    return inserted
