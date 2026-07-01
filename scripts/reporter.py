#!/usr/bin/env python3
"""Hourly Telegram Reporter for APEX V2.

Sends hourly trading status updates, P&L, and learnings to Telegram.
Supports multiple recipients (Andrew + Scott).
"""

import os
from datetime import datetime, timezone

import httpx
import structlog

logger = structlog.get_logger()


class TelegramReporter:
    """Sends trading updates to Telegram (multiple recipients)."""

    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        # Primary chat (Andrew)
        self.chat_ids: list[str] = []
        andrew_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if andrew_id:
            self.chat_ids.append(andrew_id)
        # Scott
        scott_id = os.getenv("TELEGRAM_SCOTT_CHAT_ID", "")
        if scott_id:
            self.chat_ids.append(scott_id)
        self._last_update_hour: int = -1

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_ids)

    async def send(self, text: str) -> bool:
        """Broadcast to all configured chat IDs."""
        if not self.configured:
            logger.debug("telegram.not_configured")
            return False
        results = []
        for chat_id in self.chat_ids:
            results.append(await self._send_to(chat_id, text))
        return any(results)

    async def send_to(self, chat_id: str, text: str) -> bool:
        """Send to a specific chat ID."""
        if not self.token:
            return False
        return await self._send_to(chat_id, text)

    async def _send_to(self, chat_id: str, text: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"https://api.telegram.org/bot{self.token}/sendMessage",
                    data={"chat_id": chat_id, "text": text},
                )
                if resp.status_code == 200:
                    logger.debug("telegram.sent", chat_id=chat_id)
                    return True
                else:
                    logger.warning("telegram.send_failed", status=resp.status_code, chat_id=chat_id)
                    return False
        except Exception as e:
            logger.warning("telegram.error", error=str(e), chat_id=chat_id)
            return False

    def should_send_hourly(self) -> bool:
        now = datetime.now(timezone.utc)
        if now.hour == self._last_update_hour:
            return False
        self._last_update_hour = now.hour
        return True

    def build_hourly_update(self, state: dict) -> str:
        eq = state.get("equity", 0)
        initial = state.get("initial_bankroll", 1000)
        roi = ((eq - initial) / initial * 100) if initial > 0 else 0

        bankroll = state.get("bankroll", 0)
        deployed = state.get("deployed", 0)
        unrealized = state.get("unrealized_pnl", 0)
        realized = state.get("realized_pnl", 0)
        wins = state.get("wins", 0)
        losses = state.get("losses", 0)
        total_trades = wins + losses
        wr = f"{wins/total_trades*100:.0f}%" if total_trades > 0 else "N/A"

        breaker = state.get("breaker", "GREEN")
        dd = state.get("drawdown_pct", 0)
        cycle = state.get("cycle", 0)
        signals = state.get("signals_generated", 0)
        regime = state.get("regime", "NORMAL")

        positions = state.get("positions", [])
        strategy_weights = state.get("strategy_weights", {})
        learnings = state.get("learnings_summary", "")

        eq_emoji = "📈" if roi >= 0 else "📉"
        roi_sign = "+" if roi >= 0 else ""

        breaker_emoji = {
            "GREEN": "🟢", "YELLOW": "🟡", "ORANGE": "🟠",
            "RED": "🔴", "BLACK": "⚫"
        }.get(breaker, "⚪")

        lines = [
            f"{eq_emoji} APEX Hourly Update",
            f"{datetime.now(timezone.utc).strftime('%H:%M UTC')}",
            "",
            f"Portfolio",
            f"  Equity: ${eq:,.0f} ({roi_sign}{roi:.1f}%)",
            f"  Cash: ${bankroll:,.0f} | Deployed: ${deployed:,.0f}",
            f"  P&L: ${realized:+.0f} real / ${unrealized:+.0f} unreal",
            "",
            f"Stats",
            f"  Record: {wins}W-{losses}L ({wr})",
            f"  Signals: {signals:,} | Cycle: {cycle}",
            f"  Regime: {regime}",
        ]

        if strategy_weights:
            lines.append("")
            lines.append("Strategies")
            for name, w in sorted(strategy_weights.items(),
                                   key=lambda x: x[1].get("weight", 0), reverse=True):
                weight_pct = w.get("weight", 0) * 100
                s_wins = w.get("wins", 0)
                s_losses = w.get("losses", 0)
                s_pnl = w.get("total_pnl", 0)
                lines.append(
                    f"  {name}: {weight_pct:.0f}% | "
                    f"{s_wins}W-{s_losses}L | ${s_pnl:+.0f}"
                )

        if positions:
            lines.append("")
            lines.append(f"Positions ({len(positions)})")
            sorted_pos = sorted(
                positions, key=lambda p: p.get("unrealized_pnl", 0), reverse=True,
            )
            for p in sorted_pos[:5]:
                direction = p.get("direction", "?")
                question = p.get("question", "")[:30]
                pnl = p.get("unrealized_pnl", 0)
                emoji = "+" if pnl >= 0 else ""
                lines.append(f"  {direction} {question} ${emoji}{pnl:.0f}")
            if len(sorted_pos) > 5:
                lines.append(f"  ... +{len(sorted_pos) - 5} more")

        lines.append("")
        lines.append(f"Risk: {breaker_emoji} {breaker} | DD: {dd:.1f}%")

        if learnings:
            lines.append("")
            lines.append("Learnings")
            lines.append(f"  {learnings[:150]}")

        return "\n".join(lines)

    def build_milestone_alert(self, event: str, details: dict) -> str:
        eq = details.get("equity", 0)
        initial = details.get("initial_bankroll", 1000)
        roi = ((eq - initial) / initial * 100) if initial > 0 else 0

        if event == "NEW_HIGH":
            return f"New All-Time High!\nEquity: ${eq:,.0f} (+{roi:.1f}%)"
        elif event == "DRAWDOWN_WARNING":
            dd = details.get("drawdown_pct", 0)
            return f"Drawdown Warning: {dd:.1f}%\nEquity: ${eq:,.0f}"
        elif event == "BIG_WIN":
            pnl = details.get("trade_pnl", 0)
            return f"Big Win! ${pnl:+.0f}\nEquity: ${eq:,.0f} (+{roi:.1f}%)"
        elif event == "BIG_LOSS":
            pnl = details.get("trade_pnl", 0)
            return f"Big Loss ${pnl:+.0f}\nEquity: ${eq:,.0f} ({roi:.1f}%)"
        elif event == "CIRCUIT_BREAKER":
            return f"Circuit Breaker: {details.get('breaker', '?')}\nDD: {details.get('drawdown_pct', 0):.1f}%"
        return f"Alert: {event}"

    async def send_milestone(self, event: str, details: dict) -> bool:
        text = self.build_milestone_alert(event, details)
        return await self.send(text)
