"""Decision ledger — durable, append-only analytics DB for APEX.

Every signal the bot evaluates (taken OR rejected), every trade close, price
marks, and TT daily scans land in Postgres (database `apex_ledger` on the
always-on native 5432 instance — deliberately NOT the docker TimescaleDB,
which dies whenever Docker Desktop isn't running).

The whole module is fire-and-forget: callers enqueue synchronously (never
block, never raise), a single background task drains the queue. If the DB is
down, rows are dropped with a rate-limited warning — the trading path must
never depend on analytics.

The point of all this: trades alone are a tiny sample. The rejected-signal
stream is 10-50x larger, and once the nightly flywheel job labels each row
with the market's actual settlement, every gate (min-edge floors, same-day
window, focus-list cutoff) becomes measurable instead of vibes.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import structlog

logger = structlog.get_logger("apex.ledger")

DSN = os.environ.get(
    "APEX_LEDGER_DSN",
    "postgresql://postgres:postgres@localhost:5432/apex_ledger",
)
CURRENT_EPOCH = 2          # bump at every epoch reset (see data/epochs/)
QUEUE_MAX = 2000
REJECT_DEDUP_SECONDS = 3600  # same (market,strategy,reason) logged 1x/hour
MARK_INTERVAL_SECONDS = 300  # per-market mark throttle

_queue: asyncio.Queue | None = None
_pool = None
_writer_task = None
_last_warn = 0.0
_reject_seen: dict[tuple, float] = {}
_mark_seen: dict[str, float] = {}


def _warn(msg: str, **kw):
    global _last_warn
    if time.time() - _last_warn > 300:
        _last_warn = time.time()
        logger.warning(msg, **kw)


async def _writer():
    global _pool
    import asyncpg
    while True:
        item = await _queue.get()
        try:
            if _pool is None:
                _pool = await asyncpg.create_pool(DSN, min_size=1, max_size=2)
            sql, args = item
            async with _pool.acquire() as con:
                await con.execute(sql, *args)
        except Exception as e:  # noqa: BLE001 — analytics must never crash the bot
            _warn("ledger.write_failed", error=str(e)[:200])


def start():
    """Start the background writer. Call once from the running event loop."""
    global _queue, _writer_task
    if _writer_task is not None:
        return
    _queue = asyncio.Queue(maxsize=QUEUE_MAX)
    _writer_task = asyncio.get_event_loop().create_task(_writer())
    logger.info("ledger.started", dsn=DSN.rsplit("@", 1)[-1], epoch=CURRENT_EPOCH)


def _enqueue(sql: str, args: tuple):
    if _queue is None:
        return
    try:
        _queue.put_nowait((sql, args))
    except asyncio.QueueFull:
        _warn("ledger.queue_full", dropped=sql.split()[2])


def _num(v) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _dt(v):
    """asyncpg wants datetime objects, the bots carry ISO strings."""
    from datetime import datetime
    if v is None or isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None


def _date(v):
    from datetime import date, datetime
    if v is None or isinstance(v, date) and not isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v)).date()
    except ValueError:
        return None


def log_signal(venue: str, market_id: str, strategy: str, action: str,
               *, direction: str | None = None, price=None, edge=None,
               market_title: str | None = None, reject_reason: str | None = None,
               features: dict[str, Any] | None = None):
    """action: ENTERED | REJECTED. Rejects are deduped per hour per
    (market, strategy, reason) — the cycle loop re-sees the same signal
    every 60-90s and one row per hour carries the same information."""
    if action == "REJECTED":
        key = (market_id, strategy, reject_reason)
        now = time.time()
        if now - _reject_seen.get(key, 0) < REJECT_DEDUP_SECONDS:
            return
        _reject_seen[key] = now
        if len(_reject_seen) > 5000:
            cutoff = now - REJECT_DEDUP_SECONDS
            for k in [k for k, t in _reject_seen.items() if t < cutoff]:
                del _reject_seen[k]
    _enqueue(
        """INSERT INTO signals (venue, market_id, market_title, strategy, direction,
                                price, edge, features, action, reject_reason, epoch)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)""",
        (venue, market_id, (market_title or "")[:300], strategy, direction,
         _num(price), _num(edge),
         json.dumps(features, default=str)[:4000] if features else None,
         action, reject_reason, CURRENT_EPOCH),
    )


def log_trade(record: dict[str, Any], venue: str = "kalshi"):
    """Durable close ledger. `record` is the closed-trade dict the bots
    already build (asdict(pos) + narrator fields)."""
    _enqueue(
        """INSERT INTO trades (venue, position_id, market_id, question, strategy,
                               direction, entry_price, exit_price, shares, cost_basis,
                               pnl, status, edge_at_entry, entry_time, exit_time,
                               close_summary, entry_thesis, improvement_note, epoch, raw)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,
                   $14,$15,$16,$17,$18,$19,$20)
           ON CONFLICT (position_id) DO UPDATE SET
             close_summary = COALESCE(EXCLUDED.close_summary, trades.close_summary),
             entry_thesis = COALESCE(EXCLUDED.entry_thesis, trades.entry_thesis),
             improvement_note = COALESCE(EXCLUDED.improvement_note, trades.improvement_note)""",
        (record.get("venue", venue), record.get("position_id"),
         record.get("market_id") or record.get("symbol"),
         (record.get("question") or "")[:300], record.get("strategy"),
         record.get("direction"), _num(record.get("entry_price")),
         _num(record.get("exit_price")), _num(record.get("shares")),
         _num(record.get("cost_basis")), _num(record.get("realized_pnl")),
         record.get("status"), _num(record.get("edge_at_entry")),
         _dt(record.get("entry_time")), _dt(record.get("exit_time")),
         record.get("closeSummary"), record.get("entryThesis"),
         record.get("improvementNote"), CURRENT_EPOCH,
         json.dumps(record, default=str)[:8000]),
    )


def log_mark(venue: str, market_id: str, price, *, position_id: str | None = None,
             unrealized_pnl=None):
    """Price snapshot for an open position; throttled to one per market
    per MARK_INTERVAL_SECONDS."""
    now = time.time()
    if now - _mark_seen.get(market_id, 0) < MARK_INTERVAL_SECONDS:
        return
    _mark_seen[market_id] = now
    _enqueue(
        "INSERT INTO marks (venue, market_id, position_id, price, unrealized_pnl)"
        " VALUES ($1,$2,$3,$4,$5)",
        (venue, market_id, position_id, _num(price), _num(unrealized_pnl)),
    )


def log_scan(scan_date: str, *, risk_on: bool, universe: int, eligible: int,
             focus: list[str], top: list[dict]):
    """TT daily universe scan — one row per trading day (upsert)."""
    _enqueue(
        """INSERT INTO scans (scan_date, venue, risk_on, universe, eligible, focus, top)
           VALUES ($1,'tastytrade',$2,$3,$4,$5,$6)
           ON CONFLICT (scan_date, venue) DO UPDATE SET
             risk_on=EXCLUDED.risk_on, universe=EXCLUDED.universe,
             eligible=EXCLUDED.eligible, focus=EXCLUDED.focus, top=EXCLUDED.top""",
        (_date(scan_date), risk_on, universe, eligible,
         json.dumps(focus), json.dumps(top, default=str)[:8000]),
    )
