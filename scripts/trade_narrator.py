"""Trade narrator — the narration flywheel (mirrors epik-trade's trade_narrator.ts).

At close time: a short "why it closed" summary, written fire-and-forget onto the
closed trade record. On demand (Trade Detail Panel): entry thesis + improvement
analysis, generated once and cached on the same record. Per-trade conclusions
accumulate as the substrate for higher-level learning ("flywheel").

Model: Xiaomi MiMo (operator-selected, 2026-07-02) via its OpenAI-compatible
endpoint. MiMo returns reasoning in a separate `reasoning_content` field, so
`content` is clean narrative text.

Every call is best-effort: narration failure must never affect the trading path.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger("apex.narrator")

MODEL = "mimo-v2.5"


def _mimo_config() -> tuple[str, str] | None:
    key = os.environ.get("XIAOMI_MIMO_API_KEY")
    base = os.environ.get("XIAOMI_MIMO_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1")
    if not key:
        return None
    return base.rstrip("/"), key


def _facts_block(t: dict[str, Any]) -> str:
    """Render the trade facts the model narrates from."""
    hold_min = None
    try:
        from datetime import datetime

        opened = datetime.fromisoformat(str(t.get("entry_time")))
        closed = datetime.fromisoformat(str(t.get("exit_time")))
        hold_min = (closed - opened).total_seconds() / 60
    except Exception:
        pass

    pnl = t.get("realized_pnl") or 0
    lines = [
        f"Venue: {t.get('venue', 'kalshi')}",
        f"Market: {str(t.get('question') or t.get('market_id', '')).replace('**', '')}",
        f"Direction: {t.get('direction')} | Strategy: {t.get('strategy')}",
        f"Entry price: {t.get('entry_price')} | Exit price: {t.get('exit_price') if t.get('exit_price') is not None else 'resolution'}",
        f"Size: {t.get('shares')} shares @ cost ${(t.get('cost_basis') or 0):.2f}",
        f"P&L: {'+' if pnl >= 0 else ''}${pnl:.2f} ({'WIN' if pnl >= 0 else 'LOSS'})",
        f"Model edge claimed at entry: {t.get('edge_at_entry')}",
        f"Mechanical close trigger: {t.get('status')}",
        f"Stop loss: {t.get('stop_loss')} | Take profit: {t.get('take_profit')}",
    ]
    if hold_min is not None:
        lines.append(f"Held: {hold_min:.0f} minutes")
    return "\n".join(lines)


# MiMo is a reasoning model: its thinking counts against max_tokens (reasoning
# runs 700-2500 tokens on these prompts). 2048 leaves room for clean completion.
async def _generate(system: str, facts: str, max_tokens: int = 2048) -> str | None:
    cfg = _mimo_config()
    if cfg is None:
        logger.warning("narrator disabled: XIAOMI_MIMO_API_KEY not set")
        return None
    base, key = cfg
    try:
        async with httpx.AsyncClient(timeout=60) as http:
            resp = await http.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": MODEL,
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": facts},
                    ],
                },
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"].get("content") or ""
            # Reasoning arrives in reasoning_content, but strip any leaked
            # think-tags defensively.
            text = re.sub(r"<think>[\s\S]*?</think>", "", text)
            text = re.sub(r"<think>[\s\S]*$", "", text)
            return text.strip() or None
    except Exception as exc:  # noqa: BLE001 — narration must never break trading
        logger.warning("narrator generate failed (non-fatal): %s", str(exc)[:200])
        return None


async def narrate_close(trade: dict[str, Any]) -> str | None:
    """1-2 sentence close-reason summary. Caller writes it onto the trade record."""
    return await _generate(
        "You are a trading bot's journal. In 1-2 plain sentences, explain why this "
        "trade closed when it did — name the trigger (stop loss, take profit, market "
        "resolution, stale-position close) and whether the outcome matched the "
        "strategy's intent. No preamble, no headers.",
        _facts_block(trade),
    )


async def narrate_entry_thesis(trade: dict[str, Any]) -> str | None:
    """2-3 sentence reconstruction of why the trade was taken."""
    return await _generate(
        "You are a trading bot's journal. In 2-3 plain sentences, reconstruct why "
        "this trade was taken and the thesis for how it was expected to win, based "
        "on the strategy, entry price, and claimed edge. Be concrete about what edge "
        "the strategy claims. No preamble.",
        _facts_block(trade),
    )


async def narrate_improvement(trade: dict[str, Any]) -> str | None:
    """2-3 sentence do-better-next-time analysis, feeding future retraining."""
    return await _generate(
        "You are a trading bot's post-trade reviewer feeding future retraining. In "
        "2-3 plain sentences, state what could be done better next time this kind of "
        "trade sets up — entry timing, sizing, exit rule, or additional data that "
        "would have helped. Be specific and actionable, not generic. No preamble.",
        _facts_block(trade),
    )
