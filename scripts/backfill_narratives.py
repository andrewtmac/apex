"""Backfill/regenerate AI narratives for all closed trades in paper_state_v2.json.

Run with the bot STOPPED — the trader's save loop would otherwise overwrite the
state file with its in-memory (un-narrated) copy.

    uv run python scripts/backfill_narratives.py [--force]

--force regenerates even trades that already have narratives (use after a
facts-schema change, e.g. the 2026-07-02 edge sign fix).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from trade_narrator import narrate_close, narrate_entry_thesis, narrate_improvement

STATE = os.path.join(os.path.dirname(__file__), "..", "paper_state_v2.json")
CONCURRENCY = 6

FIELDS = (
    ("closeSummary", narrate_close),
    ("entryThesis", narrate_entry_thesis),
    ("improvementNote", narrate_improvement),
)


async def narrate_trade(t: dict, force: bool, sem: asyncio.Semaphore) -> int:
    wrote = 0
    async with sem:
        for field, fn in FIELDS:
            if not force and t.get(field):
                continue
            text = await fn(t)
            if text:
                t[field] = text
                wrote += 1
            else:
                print(f"  ! {t.get('position_id')} {field}: generation failed")
    return wrote


async def main() -> None:
    force = "--force" in sys.argv
    d = json.load(open(STATE))
    trades = d.get("closed_trades", [])
    print(f"{len(trades)} closed trades | force={force}")

    sem = asyncio.Semaphore(CONCURRENCY)
    results = await asyncio.gather(*(narrate_trade(t, force, sem) for t in trades))
    total = sum(results)

    tmp = STATE + ".tmp"
    json.dump(d, open(tmp, "w"), indent=2)
    os.replace(tmp, STATE)

    have = sum(1 for t in trades if all(t.get(f) for f, _ in FIELDS))
    print(f"wrote {total} narratives | {have}/{len(trades)} trades fully narrated")


if __name__ == "__main__":
    asyncio.run(main())
