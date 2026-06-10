"""Collect historical resolved market data from Polymarket and Kalshi APIs.

Data sources:
- Polymarket Gamma API: https://gamma-api.polymarket.com
- Kalshi API: https://api.elections.kalshi.com/trade-api/v2

Collects resolved markets, normalizes them, and stores in PostgreSQL
for model training.
"""

from __future__ import annotations

import asyncio
import json
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
        resolved outcomes via outcomePrices.  Paginates through all
        closed markets.

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
                            "active": False,
                            "order": "endDateIso",
                            "ascending": False,
                        },
                    )
                    resp.raise_for_status()
                    markets = resp.json()

                    if not markets:
                        break

                    for m in markets:
                        # Parse outcome from outcomePrices (the only reliable signal)
                        outcome = self._parse_poly_outcome(m)
                        if outcome is None:
                            continue

                        # Extract final price from outcomePrices
                        final_price = self._extract_poly_final_price(m)

                        record = {
                            "market_id": m.get("conditionId", m.get("id", "")),
                            "venue": "polymarket",
                            "question": m.get("question", ""),
                            "category": m.get("category", "other"),
                            "outcome": outcome,
                            "resolution_date": m.get(
                                "endDateIso",
                                m.get("closedTime", ""),
                            ),
                            "final_price": final_price,
                            "volume": self._safe_float(m.get("volume", 0)),
                            "open_date": m.get("createdAt", ""),
                            "liquidity": self._safe_float(m.get("liquidity", 0)),
                            "num_traders": self._safe_float(
                                m.get("competitive", 0)
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
        limit: int = 50000,
        batch_size: int = 200,
    ) -> pd.DataFrame:
        """Fetch resolved markets from Kalshi /markets endpoint.

        Uses status=settled to get finalized markets, then also
        status=closed for determined markets.  Paginates with cursor.

        Parameters
        ----------
        limit : maximum total markets to fetch per status
        batch_size : markets per API call (max 200)
        """
        all_markets: list[dict[str, Any]] = []
        seen_tickers: set[str] = set()

        headers = {}
        if self._kalshi_key:
            headers["Authorization"] = f"Bearer {self._kalshi_key}"

        async with httpx.AsyncClient(
            base_url=KALSHI_API_URL,
            timeout=httpx.Timeout(self._timeout),
            headers=headers,
        ) as client:
            # Collect from both "settled" and "closed" statuses
            for status_filter in ("settled", "closed"):
                cursor: str | None = None
                fetched = 0

                while fetched < limit:
                    try:
                        params: dict[str, Any] = {
                            "limit": batch_size,
                            "status": status_filter,
                        }
                        if cursor:
                            params["cursor"] = cursor

                        resp = await client.get("/markets", params=params)
                        resp.raise_for_status()
                        data = resp.json()

                        markets = data.get("markets", [])
                        if not markets:
                            break

                        for m in markets:
                            ticker = m.get("ticker", "")
                            if ticker in seen_tickers:
                                continue

                            outcome = self._parse_kalshi_outcome(m)
                            if outcome is None:
                                continue

                            seen_tickers.add(ticker)

                            # Prices are already in dollars
                            final_price = self._safe_float(
                                m.get("last_price_dollars", 0)
                            )

                            record = {
                                "market_id": ticker,
                                "venue": "kalshi",
                                "question": m.get("title", ""),
                                "category": self._kalshi_category_from_ticker(ticker),
                                "outcome": outcome,
                                "resolution_date": m.get(
                                    "close_time",
                                    m.get("expiration_time", ""),
                                ),
                                "final_price": final_price,
                                "volume": self._safe_float(m.get("volume_fp", 0)),
                                "open_date": m.get("open_time", ""),
                                "liquidity": self._safe_float(
                                    m.get("liquidity_dollars", 0)
                                ),
                                "num_traders": 0.0,
                            }
                            all_markets.append(record)

                        cursor = data.get("cursor")
                        fetched += len(markets)

                        logger.info(
                            "historical.kalshi_batch",
                            status_filter=status_filter,
                            fetched=fetched,
                            total_collected=len(all_markets),
                        )
                        await asyncio.sleep(0.15)

                        if not cursor:
                            break

                    except httpx.HTTPStatusError as exc:
                        logger.warning(
                            "historical.kalshi_error",
                            status=exc.response.status_code,
                            status_filter=status_filter,
                        )
                        if exc.response.status_code == 429:
                            await asyncio.sleep(10.0)
                        else:
                            break
                    except Exception:
                        logger.exception("historical.kalshi_batch_failed")
                        break

        df = pd.DataFrame(all_markets)
        logger.info("historical.kalshi_complete", n_markets=len(df))
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
        """Parse outcome from Polymarket market data.

        Polymarket has NO explicit 'outcome' or 'resolved_by' field.
        Resolution is indicated by outcomePrices: a JSON string like
        '["0.99", "0.01"]'. When the first price (Yes) > 0.5 the
        outcome is YES (1). When the second price (No) > 0.5 the
        outcome is NO (0).
        """
        prices_raw = market.get("outcomePrices")
        if prices_raw:
            try:
                prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                if len(prices) >= 2:
                    p_yes = float(prices[0])
                    p_no = float(prices[1])
                    if p_yes > 0.5:
                        return 1
                    if p_no > 0.5:
                        return 0
            except (json.JSONDecodeError, ValueError, TypeError, IndexError):
                pass

        # Fallback: use lastTradePrice if available
        ltp = market.get("lastTradePrice")
        if ltp is not None:
            try:
                price = float(ltp)
                if price > 0.5:
                    return 1
                elif price < 0.5 and price > 0:
                    return 0
            except (ValueError, TypeError):
                pass

        return None

    @staticmethod
    def _extract_poly_final_price(market: dict) -> float:
        """Extract the final YES price from a Polymarket market."""
        # Try lastTradePrice first (most accurate for final price)
        ltp = market.get("lastTradePrice")
        if ltp is not None:
            try:
                price = float(ltp)
                if price > 0:
                    return price
            except (ValueError, TypeError):
                pass

        # Fall back to outcomePrices
        prices_raw = market.get("outcomePrices")
        if prices_raw:
            try:
                prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
                if prices:
                    return float(prices[0])
            except (json.JSONDecodeError, ValueError, TypeError, IndexError):
                pass

        return 0.0

    @staticmethod
    def _parse_kalshi_outcome(market: dict) -> int | None:
        """Parse outcome from Kalshi market data.

        The 'result' field contains 'yes' or 'no' for binary markets.
        Some markets have 'scalar' results which we skip.
        """
        result = market.get("result", "")
        if result == "yes":
            return 1
        elif result == "no":
            return 0

        # Skip scalar / non-binary results
        if result and result not in ("", "scalar"):
            return None

        # For 'determined' status markets without yes/no result,
        # infer from settlement or last price
        status = market.get("status", "")
        if status == "determined":
            price = 0.0
            try:
                price = float(market.get("last_price_dollars", 0))
            except (ValueError, TypeError):
                pass
            if price >= 0.9:
                return 1
            elif price <= 0.1 and price >= 0:
                return 0

        return None

    @staticmethod
    def _kalshi_category_from_ticker(ticker: str) -> str:
        """Infer a rough category from Kalshi ticker prefix."""
        t = ticker.upper()
        if any(x in t for x in ("SPORT", "NFL", "NBA", "MLB", "NHL", "MMA", "FIFA")):
            return "sports"
        if any(x in t for x in ("CRYPTO", "BTC", "ETH", "XRP")):
            return "crypto"
        if any(x in t for x in ("ECON", "GDP", "CPI", "JOBS", "FED", "RATE")):
            return "economics"
        if any(x in t for x in ("PRES", "ELECT", "VOTE", "SENATE", "HOUSE")):
            return "politics"
        if any(x in t for x in ("WEATHER", "TEMP", "RAIN", "SNOW", "HURRICANE")):
            return "weather"
        if any(x in t for x in ("MVE", "GAME", "MATCH")):
            return "sports"
        if any(x in t for x in ("WTA", "ATP")):
            return "sports"
        return "other"

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


def _to_datetime(val: Any) -> datetime | None:
    """Convert a value to a timezone-aware datetime, or None."""
    if val is None:
        return None
    # Must check pd.isna BEFORE isinstance checks because pd.NaT
    # passes isinstance(val, datetime) but cannot be used as a real datetime.
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, pd.Timestamp):
        dt = val.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    if isinstance(val, datetime):
        if val.tzinfo is None:
            val = val.replace(tzinfo=timezone.utc)
        return val
    if isinstance(val, str) and val.strip():
        try:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
        # Try common date-only format
        try:
            dt = datetime.strptime(val.strip()[:10], "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


async def store_to_database(
    df: pd.DataFrame,
    db_url: str,
) -> int:
    """Store collected data into the resolved_markets table.

    Returns the number of rows inserted.
    """
    import asyncpg

    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
    inserted = 0

    try:
        async with pool.acquire() as conn:
            for _, row in df.iterrows():
                try:
                    res_date = _to_datetime(row.get("resolution_date"))
                    open_date = _to_datetime(row.get("open_date"))

                    # Safely convert numeric fields
                    final_price = float(row.get("final_price", 0) or 0)
                    volume = float(row.get("volume", 0) or 0)
                    duration_hours = float(row.get("duration_hours", 0) or 0)
                    liquidity = float(row.get("liquidity", 0) or 0)
                    num_traders = float(row.get("num_traders", 0) or 0)

                    await conn.execute(
                        """
                        INSERT INTO resolved_markets
                            (id, venue, title, category, outcome,
                             final_price, volume, created_at, resolved_at, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT (id) DO UPDATE SET
                            outcome = EXCLUDED.outcome,
                            final_price = EXCLUDED.final_price,
                            volume = EXCLUDED.volume,
                            resolved_at = EXCLUDED.resolved_at
                        """,
                        str(row["market_id"]),
                        str(row["venue"]),
                        str(row.get("question", "")),
                        str(row.get("category", "other")),
                        int(row["outcome"]),
                        final_price,
                        volume,
                        open_date,
                        res_date,
                        json.dumps({
                            "duration_hours": duration_hours,
                            "liquidity": liquidity,
                            "num_traders": num_traders,
                        }),
                    )
                    inserted += 1
                except Exception as exc:
                    logger.debug(
                        "historical.insert_failed",
                        market_id=row.get("market_id"),
                        error=str(exc),
                    )
    finally:
        await pool.close()

    logger.info("historical.stored", n_inserted=inserted)
    return inserted
