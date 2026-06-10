#!/usr/bin/env python3
"""Collect historical data from Polymarket and Kalshi for model training.

Aggressively collects resolved markets from both platforms, stores to
PostgreSQL, and saves a CSV backup.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import os

import structlog

structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(colors=True),
    ],
)

logger = structlog.get_logger()


async def main():
    from apex.research.historical_data import HistoricalDataCollector, store_to_database

    db_url = os.environ.get(
        "DATABASE_URL", "postgresql://odin-mini@localhost:5432/apex"
    )

    logger.info("collect.starting", db_url=db_url.split("@")[-1])

    collector = HistoricalDataCollector(
        polymarket_api_key=os.environ.get("POLYMARKET_API_KEY", ""),
        kalshi_api_key=os.environ.get("KALSHI_API_KEY", ""),
        timeout=60.0,
    )

    t0 = time.time()

    # Collect Polymarket -- be aggressive, aim for many thousands
    logger.info("collect.polymarket_starting")
    try:
        poly_df = await collector.collect_polymarket(limit=50000, batch_size=100)
        logger.info("collect.polymarket_done", n_markets=len(poly_df))
    except Exception as e:
        logger.error("collect.polymarket_failed", error=str(e))
        import traceback

        traceback.print_exc()
        poly_df = None

    # Collect Kalshi -- both settled and closed
    logger.info("collect.kalshi_starting")
    try:
        kalshi_df = await collector.collect_kalshi(limit=50000, batch_size=200)
        logger.info("collect.kalshi_done", n_markets=len(kalshi_df))
    except Exception as e:
        logger.error("collect.kalshi_failed", error=str(e))
        import traceback

        traceback.print_exc()
        kalshi_df = None

    # Build combined dataset
    import pandas as pd

    frames = []
    if poly_df is not None and not poly_df.empty:
        frames.append(poly_df)
    if kalshi_df is not None and not kalshi_df.empty:
        frames.append(kalshi_df)

    if not frames:
        logger.error("collect.no_data", msg="No markets collected from any source!")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = collector._normalize_dataset(combined)

    elapsed = time.time() - t0

    # Print statistics
    print("\n" + "=" * 70)
    print("  APEX Historical Data Collection -- Results")
    print("=" * 70)

    if poly_df is not None and not poly_df.empty:
        print(f"\n  Polymarket: {len(poly_df):,} resolved markets")
        print(f"    Categories: {poly_df['category'].nunique()}")
        print(f"    Outcome dist: YES={int((poly_df['outcome']==1).sum()):,}  "
              f"NO={int((poly_df['outcome']==0).sum()):,}")
        vol = poly_df["volume"].sum()
        print(f"    Total volume: ${vol:,.0f}")
    else:
        print("\n  Polymarket: 0 markets (FAILED)")

    if kalshi_df is not None and not kalshi_df.empty:
        print(f"\n  Kalshi: {len(kalshi_df):,} resolved markets")
        print(f"    Categories: {kalshi_df['category'].nunique()}")
        print(f"    Outcome dist: YES={int((kalshi_df['outcome']==1).sum()):,}  "
              f"NO={int((kalshi_df['outcome']==0).sum()):,}")
        vol = kalshi_df["volume"].sum()
        print(f"    Total volume: ${vol:,.0f}")
    else:
        print("\n  Kalshi: 0 markets (FAILED)")

    print(f"\n  Combined: {len(combined):,} unique resolved markets")
    print(f"  Outcome distribution:")
    print(f"    YES (1): {int((combined['outcome']==1).sum()):,} "
          f"({(combined['outcome']==1).mean()*100:.1f}%)")
    print(f"    NO  (0): {int((combined['outcome']==0).sum()):,} "
          f"({(combined['outcome']==0).mean()*100:.1f}%)")

    if "category" in combined.columns:
        print(f"\n  Category breakdown:")
        for cat, count in combined["category"].value_counts().head(15).items():
            print(f"    {cat:25s} {count:>6,}")

    if "venue" in combined.columns:
        print(f"\n  Venue breakdown:")
        for venue, count in combined["venue"].value_counts().items():
            print(f"    {venue:25s} {count:>6,}")

    print(f"\n  Collection time: {elapsed:.1f}s")

    # Store to database
    print(f"\n  Storing to PostgreSQL ({db_url.split('@')[-1]})...")
    try:
        n_stored = await store_to_database(combined, db_url)
        print(f"  Stored: {n_stored:,} rows (upserted)")
    except Exception as e:
        logger.error("collect.store_failed", error=str(e))
        import traceback

        traceback.print_exc()
        n_stored = 0

    # Save CSV backup
    csv_path = Path(__file__).parent.parent / "data" / "training_data.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, index=False)
    size_mb = csv_path.stat().st_size / 1e6
    print(f"  CSV backup: {csv_path} ({size_mb:.2f} MB)")

    print("\n" + "=" * 70)
    target = 1000
    status = "PASS" if len(combined) >= target else "BELOW TARGET"
    print(f"  RESULT: {len(combined):,} markets collected [{status}]"
          f" (target: {target:,}+)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
