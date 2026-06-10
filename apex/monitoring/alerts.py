"""Telegram alerting for APEX.

Alerts on:
- Circuit breaker state changes
- Trade executions (above threshold)
- Data source failures
- Model retrain results
- Daily P&L summary
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org"

# Priority emoji mapping
_PRIORITY_ICONS = {
    "critical": "!!",
    "warning": "!",
    "info": "",
    "success": "",
}


class AlertManager:
    """Sends structured alerts to Telegram.

    Formats messages with consistent structure and priority levels.
    Supports rate limiting to avoid flood during rapid state changes.

    Parameters
    ----------
    bot_token : Telegram bot API token
    chat_id : target chat or group ID
    rate_limit_seconds : minimum interval between alerts (per category)
    enabled : master switch for alerting
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        rate_limit_seconds: float = 5.0,
        enabled: bool = True,
    ) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.rate_limit_seconds = rate_limit_seconds
        self.enabled = enabled and bool(bot_token) and bool(chat_id)
        self._last_alert_times: dict[str, float] = {}

    def _should_send(self, category: str) -> bool:
        """Check rate limit for a given alert category."""
        import time

        now = time.time()
        last = self._last_alert_times.get(category, 0)
        if now - last < self.rate_limit_seconds:
            return False
        self._last_alert_times[category] = now
        return True

    async def send_alert(
        self,
        message: str,
        priority: str = "info",
        category: str = "general",
        parse_mode: str = "HTML",
    ) -> bool:
        """Send a Telegram alert message.

        Parameters
        ----------
        message : alert text (supports HTML formatting)
        priority : "critical", "warning", "info", "success"
        category : rate-limiting category
        parse_mode : "HTML" or "MarkdownV2"

        Returns
        -------
        True if message was sent successfully
        """
        if not self.enabled:
            logger.debug("alerts.disabled", message=message[:80])
            return False

        if not self._should_send(category):
            logger.debug("alerts.rate_limited", category=category)
            return False

        icon = _PRIORITY_ICONS.get(priority, "")
        prefix = f"[APEX {icon}]" if icon else "[APEX]"
        full_message = f"{prefix} {message}"

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                resp = await client.post(
                    f"{TELEGRAM_API_BASE}/bot{self.bot_token}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": full_message,
                        "parse_mode": parse_mode,
                        "disable_web_page_preview": True,
                    },
                )
                resp.raise_for_status()
                logger.info(
                    "alerts.sent",
                    priority=priority,
                    category=category,
                    length=len(full_message),
                )
                return True

        except Exception:
            logger.exception("alerts.send_failed")
            return False

    async def daily_summary(
        self,
        portfolio: dict[str, Any],
        performance: dict[str, Any],
    ) -> None:
        """Send daily P&L summary.

        Parameters
        ----------
        portfolio : current portfolio state dict
        performance : performance metrics dict
        """
        equity = portfolio.get("total_equity", 0)
        daily_pnl = performance.get("daily_pnl", 0)
        daily_pnl_pct = performance.get("daily_pnl_pct", 0)
        total_return = performance.get("total_return_pct", 0)

        n_trades = performance.get("trades_today", 0)
        win_rate = performance.get("win_rate", 0)
        drawdown = portfolio.get("drawdown_pct", 0)

        regime = portfolio.get("regime", "NORMAL")
        breaker = portfolio.get("breaker_level", "GREEN")

        pnl_sign = "+" if daily_pnl >= 0 else ""

        positions_detail = ""
        positions = portfolio.get("positions", {})
        if positions:
            pos_lines = []
            for venue, count in positions.items():
                pos_lines.append(f"  {venue}: {count}")
            positions_detail = "\n".join(pos_lines)
        else:
            positions_detail = "  None"

        message = (
            f"<b>Daily Summary</b>\n"
            f"{'='*28}\n\n"
            f"<b>Equity:</b> ${equity:,.2f}\n"
            f"<b>Daily P&L:</b> {pnl_sign}${daily_pnl:,.2f} ({pnl_sign}{daily_pnl_pct:.2f}%)\n"
            f"<b>Total Return:</b> {total_return:+.2f}%\n\n"
            f"<b>Trades:</b> {n_trades}\n"
            f"<b>Win Rate:</b> {win_rate:.1%}\n"
            f"<b>Max Drawdown:</b> {drawdown:.2%}\n\n"
            f"<b>Regime:</b> {regime}\n"
            f"<b>Breaker:</b> {breaker}\n\n"
            f"<b>Positions:</b>\n{positions_detail}"
        )

        priority = "success" if daily_pnl >= 0 else "warning"
        await self.send_alert(message, priority=priority, category="daily_summary")

    async def breaker_alert(
        self,
        old_level: str,
        new_level: str,
        reason: str = "",
    ) -> None:
        """Alert on circuit breaker state change.

        Parameters
        ----------
        old_level : previous breaker level (GREEN, YELLOW, RED)
        new_level : new breaker level
        reason : why the breaker changed
        """
        severity_map = {
            "GREEN": 0,
            "YELLOW": 1,
            "RED": 2,
        }

        is_escalation = severity_map.get(new_level, 0) > severity_map.get(
            old_level, 0
        )
        priority = "critical" if is_escalation and new_level == "RED" else "warning"

        direction = "ESCALATED" if is_escalation else "DE-ESCALATED"

        message = (
            f"<b>Circuit Breaker {direction}</b>\n\n"
            f"{old_level} -> <b>{new_level}</b>\n"
        )
        if reason:
            message += f"\nReason: {reason}"

        await self.send_alert(
            message, priority=priority, category="breaker"
        )

    async def trade_alert(
        self,
        trade: dict[str, Any],
        threshold_usd: float = 50.0,
    ) -> None:
        """Alert on trade execution above threshold.

        Parameters
        ----------
        trade : trade details dict
        threshold_usd : minimum trade size to alert on
        """
        cost = abs(trade.get("cost", 0))
        if cost < threshold_usd:
            return

        direction = trade.get("direction", "?")
        venue = trade.get("venue", "?")
        market = trade.get("market_id", "?")[:30]
        price = trade.get("price", 0)
        edge = trade.get("edge", 0)

        message = (
            f"<b>Trade Executed</b>\n\n"
            f"<b>{direction}</b> on {venue}\n"
            f"Market: {market}\n"
            f"Price: {price:.4f}\n"
            f"Size: ${cost:,.2f}\n"
            f"Edge: {edge:+.4f}"
        )

        await self.send_alert(message, priority="info", category="trade")

    async def model_retrain_alert(
        self,
        model_name: str,
        result: dict[str, Any],
    ) -> None:
        """Alert on model retrain completion.

        Parameters
        ----------
        model_name : name of the retrained model
        result : training result dict with status and metrics
        """
        status = result.get("status", "unknown")
        version = result.get("version_id", "?")
        metrics = result.get("metrics", {})

        # Format top metrics
        metric_lines = []
        for key in ["accuracy", "brier_score", "sharpe_ratio", "ic", "f1_macro"]:
            if key in metrics:
                metric_lines.append(f"  {key}: {metrics[key]:.4f}")

        metrics_text = "\n".join(metric_lines) if metric_lines else "  (none)"

        priority = "success" if status == "deployed" else "warning"

        message = (
            f"<b>Model Retrained: {model_name}</b>\n\n"
            f"Status: <b>{status.upper()}</b>\n"
            f"Version: {version}\n\n"
            f"<b>Metrics:</b>\n{metrics_text}"
        )

        await self.send_alert(
            message, priority=priority, category=f"retrain_{model_name}"
        )

    async def data_failure_alert(
        self,
        source: str,
        error: str,
        stale_seconds: float | None = None,
    ) -> None:
        """Alert on data source failure.

        Parameters
        ----------
        source : data source name
        error : error description
        stale_seconds : how long the data has been stale
        """
        message = f"<b>Data Source Failure: {source}</b>\n\n{error}"
        if stale_seconds is not None:
            minutes = stale_seconds / 60
            message += f"\nStale for: {minutes:.1f} minutes"

        await self.send_alert(
            message, priority="warning", category=f"data_{source}"
        )

    async def risk_alert(
        self,
        metric: str,
        value: float,
        threshold: float,
        message: str = "",
    ) -> None:
        """Alert on risk metric breach.

        Parameters
        ----------
        metric : risk metric name
        value : current value
        threshold : threshold that was breached
        message : additional context
        """
        alert_text = (
            f"<b>Risk Alert: {metric}</b>\n\n"
            f"Current: {value:.4f}\n"
            f"Threshold: {threshold:.4f}\n"
        )
        if message:
            alert_text += f"\n{message}"

        await self.send_alert(
            alert_text, priority="critical", category=f"risk_{metric}"
        )
