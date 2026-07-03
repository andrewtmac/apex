"""Nightly flywheel job — turns the decision ledger into learning.

Three passes, all DB-only (safe to run while the bot trades):

1. LABEL: every unlabeled Kalshi signal row whose market has settled gets
   outcome (yes/no), cf_win (would our side have won?) and cf_pnl
   (counterfactual P&L per contract at the signal's price, fees modeled).
   This is what makes REJECTED rows learnable — every gate becomes
   measurable against what actually happened.
2. STATS: 7-day cohort table — per strategy × action × reject_reason:
   count, win rate, realized P&L (entered) / counterfactual P&L (rejected).
   Plus edge-bucket calibration: claimed edge vs. settled win rate.
3. LESSON: the stats go to MiMo for a short synthesis, stored in `lessons`.

Runs inside apex_v2 (nightly_loop, 07:10 UTC — after the weather window
closes and most same-day markets settle) or standalone:

    uv run python scripts/flywheel_job.py
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

import httpx
import structlog

from ledger import DSN

logger = structlog.get_logger("apex.flywheel")

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
LABEL_BATCH = 300          # markets checked per run (politeness cap)
RUN_AT_UTC_HOUR = 7        # 07:10 UTC nightly


def _kalshi_fee(price: float) -> float:
    """Per-contract fee, both sides approximated at the same price."""
    p = min(max(price, 0.01), 0.99)
    return 2 * math.ceil(7 * p * (1 - p)) / 100


async def label_outcomes(pool) -> int:
    """Settle unlabeled signal rows against Kalshi market results."""
    rows = await pool.fetch(
        """SELECT DISTINCT market_id FROM signals
           WHERE venue = 'kalshi' AND labeled_at IS NULL
             AND ts < now() - interval '2 hours'
           LIMIT $1""", LABEL_BATCH)
    labeled = 0
    async with httpx.AsyncClient(timeout=15) as http:
        for r in rows:
            mid = r["market_id"]
            try:
                resp = await http.get(f"{KALSHI_API}/markets/{mid}")
                if resp.status_code != 200:
                    continue
                m = resp.json().get("market", {})
                result = m.get("result")          # "yes" | "no" | "" while open
                if m.get("status") != "settled" or result not in ("yes", "no"):
                    continue
                await pool.execute(
                    """UPDATE signals SET
                         outcome = $2,
                         cf_win = (direction = 'BUY') = ($2 = 'yes'),
                         cf_pnl = CASE
                           WHEN (direction = 'BUY') = ($2 = 'yes')
                           THEN (CASE WHEN direction = 'BUY' THEN 1 - price ELSE price END) - $3
                           ELSE -(CASE WHEN direction = 'BUY' THEN price ELSE 1 - price END) - $3
                         END,
                         labeled_at = now()
                       WHERE market_id = $1 AND labeled_at IS NULL""",
                    mid, result, _kalshi_fee(0.5))
                labeled += 1
                await asyncio.sleep(0.1)
            except Exception as e:  # noqa: BLE001
                logger.debug("flywheel.label_error", market=mid, error=str(e)[:100])
    return labeled


async def cohort_stats(pool) -> dict:
    """7-day decision table + edge calibration, JSON-serializable."""
    cohorts = await pool.fetch(
        """SELECT strategy, action, COALESCE(reject_reason, '-') AS reason,
                  count(*) AS n,
                  count(*) FILTER (WHERE cf_win) AS cf_wins,
                  count(*) FILTER (WHERE labeled_at IS NOT NULL) AS labeled,
                  round(avg(cf_pnl)::numeric, 4) AS avg_cf_pnl
           FROM signals WHERE ts > now() - interval '7 days' AND venue = 'kalshi'
           GROUP BY 1, 2, 3 ORDER BY n DESC""")
    calib = await pool.fetch(
        """SELECT width_bucket(abs(edge), 0, 0.6, 6) AS bucket,
                  count(*) AS n, count(*) FILTER (WHERE cf_win) AS wins
           FROM signals
           WHERE ts > now() - interval '14 days' AND labeled_at IS NOT NULL
           GROUP BY 1 ORDER BY 1""")
    trades = await pool.fetch(
        """SELECT venue, strategy, status, count(*) AS n,
                  round(sum(pnl)::numeric, 2) AS pnl,
                  count(*) FILTER (WHERE pnl >= 0) AS wins
           FROM trades WHERE ts > now() - interval '7 days'
           GROUP BY 1, 2, 3 ORDER BY n DESC""")
    return {
        "cohorts": [dict(r) for r in cohorts],
        "calibration": [
            {"edge_range": f"{(b['bucket'] - 1) * 0.1:.1f}-{b['bucket'] * 0.1:.1f}",
             "n": b["n"], "win_rate": round(b["wins"] / b["n"], 3) if b["n"] else None}
            for b in calib],
        "trade_outcomes": [dict(r) for r in trades],
    }


async def synthesize_lesson(stats: dict) -> str | None:
    """MiMo synthesis of the week's decision data."""
    try:
        from trade_narrator import _generate
        return await _generate(
            "You are a trading bot's weekly quant reviewer. Below are 7-day "
            "decision cohorts (every signal taken AND rejected, with "
            "counterfactual outcomes where markets settled), edge-bucket "
            "calibration, and trade outcomes. In 4-6 plain sentences: name "
            "the single biggest leak (a gate rejecting winners, or letting "
            "losers through), whether claimed edge is calibrated to realized "
            "win rate, and the one parameter change with the best expected "
            "value. Be specific with numbers. No preamble, no headers.",
            json.dumps(stats, default=str)[:6000],
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("flywheel.lesson_failed", error=str(e)[:150])
        return None


async def run_once() -> dict:
    import asyncpg
    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=2)
    try:
        labeled = await label_outcomes(pool)
        stats = await cohort_stats(pool)
        lesson = await synthesize_lesson(stats)
        await pool.execute(
            "INSERT INTO lessons (period, stats, lesson, model) VALUES ($1,$2,$3,$4)",
            "7d", json.dumps(stats, default=str), lesson, "mimo-v2.5")
        logger.info("flywheel.run_complete", labeled=labeled,
                    cohorts=len(stats["cohorts"]), lesson=bool(lesson))
        return {"labeled": labeled, "stats": stats, "lesson": lesson}
    finally:
        await pool.close()


async def nightly_loop():
    """Scheduled inside apex_v2's gather — runs once per UTC day at 07:10."""
    last_run = ""
    while True:
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        if now.hour == RUN_AT_UTC_HOUR and now.minute >= 10 and last_run != today:
            last_run = today
            try:
                await run_once()
            except Exception as e:  # noqa: BLE001 — never take down the bot
                logger.warning("flywheel.run_failed", error=str(e)[:200])
        await asyncio.sleep(300)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    out = asyncio.run(run_once())
    print(f"labeled {out['labeled']} markets")
    print(json.dumps(out["stats"], indent=1, default=str)[:3000])
    if out["lesson"]:
        print("\nLESSON:\n" + out["lesson"])
