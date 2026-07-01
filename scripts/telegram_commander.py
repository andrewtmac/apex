#!/usr/bin/env python3
"""Telegram Commander for APEX V2.

Listens for incoming Telegram messages. Only two authorized users (Andrew, Scott)
can send commands. The ONLY accepted command type is "UI ..." — any message
prefixed with "UI" triggers a Claude-powered dashboard HTML modification.

All other message types are rejected with a polite explanation.
"""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import structlog

logger = structlog.get_logger()

DASHBOARD_HTML = Path(__file__).parent / "dashboard_v2.html"

# Authorized user chat IDs
AUTHORIZED_USERS: dict[str, str] = {}  # populated in __init__


def _load_authorized() -> dict[str, str]:
    users = {}
    andrew = os.getenv("TELEGRAM_CHAT_ID", "")
    scott = os.getenv("TELEGRAM_SCOTT_CHAT_ID", "")
    if andrew:
        users[andrew] = "Andrew"
    if scott:
        users[scott] = "Scott"
    return users


class TelegramCommander:
    """Listens for UI commands from authorized Telegram users."""

    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.authorized = _load_authorized()
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._offset = 0
        self._running = False

    @property
    def configured(self) -> bool:
        return bool(self.token and self.authorized and self.anthropic_key)

    async def run(self):
        """Main polling loop — checks for new Telegram messages every 3s."""
        if not self.configured:
            logger.info("commander.not_configured", msg="Skipping Telegram command listener")
            return

        self._running = True
        logger.info("commander.started", users=list(self.authorized.values()))

        # Flush old updates so we don't process stale messages
        await self._flush_old_updates()

        while self._running:
            try:
                await self._poll_once()
            except Exception as e:
                logger.warning("commander.poll_error", error=str(e))
            await asyncio.sleep(3)

    async def _flush_old_updates(self):
        """Skip any messages that arrived before the bot started."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"https://api.telegram.org/bot{self.token}/getUpdates",
                    params={"offset": -1, "timeout": 0},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    updates = data.get("result", [])
                    if updates:
                        self._offset = updates[-1]["update_id"] + 1
                        logger.info("commander.flushed", count=len(updates))
        except Exception:
            pass

    async def _poll_once(self):
        """Fetch and process one batch of updates."""
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"https://api.telegram.org/bot{self.token}/getUpdates",
                params={"offset": self._offset, "timeout": 10},
            )
            if resp.status_code != 200:
                return

            data = resp.json()
            for update in data.get("result", []):
                self._offset = update["update_id"] + 1
                await self._handle_update(client, update)

    async def _handle_update(self, client: httpx.AsyncClient, update: dict):
        """Route a single Telegram update."""
        msg = update.get("message")
        if not msg:
            return

        text = (msg.get("text") or "").strip()
        chat = msg.get("chat", {})
        chat_id = str(chat.get("id", ""))
        user = msg.get("from", {})
        username = user.get("username", user.get("first_name", "unknown"))

        # 1. Authorization check
        if chat_id not in self.authorized:
            logger.debug("commander.unauthorized", chat_id=chat_id, username=username)
            return

        sender_name = self.authorized[chat_id]
        logger.info("commander.message_received", sender=sender_name, text=text[:80])

        # 2. Only "UI" commands are accepted
        if not text.upper().startswith("UI"):
            await self._reply(
                chat_id,
                "Only UI changes are supported.\n\n"
                'Prefix your message with "UI" to modify dashboards.\n'
                "Example: UI change the equity card color to blue",
            )
            return

        # 3. Extract the UI request (everything after "UI")
        ui_request = text[2:].strip()
        if not ui_request:
            await self._reply(
                chat_id,
                "Please include your UI change after the UI prefix.\n"
                "Example: UI add a daily P&L row to the summary section",
            )
            return

        # 4. Process the UI change via Claude
        await self._reply(chat_id, f"Processing UI change: {ui_request[:60]}...")
        success, summary = await self._apply_ui_change(ui_request, sender_name)

        if success:
            await self._reply(chat_id, f"✅ Dashboard updated!\n\n{summary}")
        else:
            await self._reply(chat_id, f"❌ Failed to apply change:\n{summary}")

    async def _apply_ui_change(self, request: str, requester: str) -> tuple[bool, str]:
        """Send current HTML + request to Claude, get back modified HTML."""
        try:
            current_html = DASHBOARD_HTML.read_text()
        except Exception as e:
            return False, f"Could not read dashboard HTML: {e}"

        system_prompt = (
            "You are a dashboard UI editor. You receive the current HTML of a trading "
            "dashboard and a user request for a UI change. You must:\n"
            "1. Apply ONLY the requested UI change — do not modify data logic or API endpoints\n"
            "2. Preserve all existing functionality\n"
            "3. Return the COMPLETE modified HTML file (not a diff)\n"
            "4. Keep the same visual style (dark theme, Inter font, CSS variables)\n"
            "5. If the request is ambiguous, make a reasonable interpretation\n"
            "6. Do NOT add external dependencies — use inline CSS/JS only\n\n"
            "Return ONLY the HTML. No explanation, no markdown fences."
        )

        user_msg = (
            f"Requested by: {requester}\n"
            f"Change requested: {request}\n\n"
            f"Current dashboard HTML:\n{current_html}"
        )

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.anthropic_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 16000,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": user_msg}],
                    },
                )

            if resp.status_code != 200:
                return False, f"Anthropic API error: {resp.status_code} {resp.text[:200]}"

            data = resp.json()
            # Extract text from response
            content_blocks = data.get("content", [])
            html_text = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    html_text += block["text"]

            # Strip markdown fences if present
            html_text = html_text.strip()
            if html_text.startswith("```"):
                # Remove opening fence (possibly ```html)
                first_newline = html_text.index("\n")
                html_text = html_text[first_newline + 1:]
            if html_text.endswith("```"):
                html_text = html_text[:-3].rstrip()

            # Validate it looks like HTML
            if "<html" not in html_text.lower() and "<!doctype" not in html_text.lower():
                return False, "Claude did not return valid HTML. Aborting to protect dashboard."

            # Backup current version
            backup = DASHBOARD_HTML.with_suffix(".html.bak")
            backup.write_text(current_html)

            # Write new version
            DASHBOARD_HTML.write_text(html_text)

            logger.info(
                "commander.ui_updated",
                requester=requester,
                request=request[:80],
                size=len(html_text),
            )

            usage = data.get("usage", {})
            tokens_in = usage.get("input_tokens", 0)
            tokens_out = usage.get("output_tokens", 0)
            return True, f"Applied change ({tokens_in + tokens_out} tokens). Backup saved."

        except Exception as e:
            return False, f"Error calling Claude: {e}"

    async def _reply(self, chat_id: str, text: str):
        """Send a reply to a specific chat."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                await client.post(
                    f"https://api.telegram.org/bot{self.token}/sendMessage",
                    data={"chat_id": chat_id, "text": text},
                )
        except Exception as e:
            logger.warning("commander.reply_failed", chat_id=chat_id, error=str(e))
