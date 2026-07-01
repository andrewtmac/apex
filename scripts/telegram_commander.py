#!/usr/bin/env python3
"""Telegram Commander for APEX V2.

Listens for incoming Telegram messages. Only two authorized users (Andrew, Scott)
can send commands. The ONLY accepted command type is "UI ..." — any message
prefixed with "UI" triggers a Claude-powered dashboard HTML modification.

Command format:
  UI                              — show help / available targets
  UI APEX <change>                — modify APEX main dashboard
  UI EPIK <change>                — modify epik-trade main dashboard (index)
  UI EPIK <page> <change>         — modify specific epik-trade page
                                     pages: overview, investor, proposals, leadership

All other message types are rejected with a polite explanation.
"""

import asyncio
import os
from pathlib import Path

import httpx
import structlog

logger = structlog.get_logger()

# --- Dashboard registry ---
# Each entry: (display_name, description, file_path)
# Relative paths resolve from the apex/scripts directory.

APEX_DIR = Path(__file__).parent.parent  # /CODING/apex
EPIK_DIR = Path("/Users/odin-mini/CODING/epik-trade/polymarket-autobot")

DASHBOARDS: dict[str, dict[str, dict]] = {
    "APEX": {
        "main": {
            "name": "APEX Dashboard",
            "path": APEX_DIR / "scripts" / "dashboard_v2.html",
            "url_hint": "http://100.64.161.91:8080",
        },
    },
    "EPIK": {
        "main": {
            "name": "Epik Trade Dashboard",
            "path": EPIK_DIR / "public" / "index.html",
            "url_hint": "http://100.64.161.91:3001",
        },
        "overview": {
            "name": "Epik Overview",
            "path": EPIK_DIR / "public" / "overview.html",
            "url_hint": "http://100.64.161.91:3001/overview",
        },
        "investor": {
            "name": "Epik Investor View",
            "path": EPIK_DIR / "public" / "investor.html",
            "url_hint": "http://100.64.161.91:3001/investor",
        },
        "proposals": {
            "name": "Epik Proposals",
            "path": EPIK_DIR / "public" / "proposals.html",
            "url_hint": "http://100.64.161.91:3001/proposals",
        },
        "leadership": {
            "name": "Epik Leadership",
            "path": EPIK_DIR / "public" / "leadership.html",
            "url_hint": "http://100.64.161.91:3001/leadership",
        },
    },
}

# Aliases so users can type shorthand
PAGE_ALIASES = {
    "index": "main",
    "home": "main",
    "dash": "main",
    "dashboard": "main",
}

# Authorized user chat IDs (populated from .env)
AUTHORIZED_USERS: dict[str, str] = {}


def _load_authorized() -> dict[str, str]:
    users = {}
    andrew = os.getenv("TELEGRAM_CHAT_ID", "")
    scott = os.getenv("TELEGRAM_SCOTT_CHAT_ID", "")
    if andrew:
        users[andrew] = "Andrew"
    if scott:
        users[scott] = "Scott"
    return users


def _build_help() -> str:
    lines = ["APEX UI Commander — Available targets:\n"]
    for bot, pages in DASHBOARDS.items():
        for page_key, info in pages.items():
            label = f"  UI {bot}" if page_key == "main" else f"  UI {bot} {page_key}"
            lines.append(f"{label}\n    {info['name']} ({info['url_hint']})")
    lines.append('\nExamples:')
    lines.append('  UI APEX change equity card color to blue')
    lines.append('  UI EPIK overview add a P&L chart')
    lines.append('  UI EPIK investor make the table sortable')
    return "\n".join(lines)


def _resolve_target(bot: str, page: str | None) -> tuple[bool, str, dict | None]:
    """Resolve bot + page to a dashboard entry. Returns (ok, error_msg, entry)."""
    bot = bot.upper()
    if bot not in DASHBOARDS:
        return False, f"Unknown bot '{bot}'. Available: {', '.join(DASHBOARDS.keys())}", None

    pages = DASHBOARDS[bot]
    if not page or page.lower() in ("main", "index", "home", "dash", "dashboard"):
        page_key = "main"
    else:
        page_key = PAGE_ALIASES.get(page.lower(), page.lower())

    if page_key not in pages:
        avail = ", ".join(pages.keys())
        return False, f"Unknown page '{page}' for {bot}. Available: {avail}", None

    return True, "", pages[page_key]


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
                "Send UI help for available targets.",
            )
            return

        # 3. Parse: UI [BOT] [PAGE] <change>
        rest = text[2:].strip()
        if not rest or rest.lower() == "help":
            await self._reply(chat_id, _build_help())
            return

        tokens = rest.split(None, 2)  # max 3 parts
        bot_name = tokens[0].upper() if tokens else ""

        # Check if first token is a known bot name
        if bot_name in DASHBOARDS:
            page = tokens[1].lower() if len(tokens) >= 3 else None
            change = tokens[2] if len(tokens) >= 3 else (tokens[1] if len(tokens) == 2 else "")
            # Edge case: "UI APEX <change>" (no page, just bot + change)
            if len(tokens) == 2:
                page = None
                change = tokens[1]
        else:
            # No bot specified — assume APEX for backward compat
            bot_name = "APEX"
            page = None
            change = rest

        # If change is empty after parsing, they just typed a bot name
        if not change.strip():
            await self._reply(chat_id, f"Please include your UI change for {bot_name}.\n\n" + _build_help())
            return

        # 4. Resolve target
        ok, err, target = _resolve_target(bot_name, page)
        if not ok or target is None:
            await self._reply(chat_id, err)
            return

        # 5. Check file exists
        target_path: Path = target["path"]
        if not target_path.exists():
            await self._reply(chat_id, f"Dashboard file not found: {target_path}\nIs the bot running?")
            return

        # 6. Immediate ack, then process
        await self._reply(chat_id, "Received, working on it")
        success, summary = await self._apply_ui_change(target, change, sender_name)

        if success:
            await self._reply(
                chat_id,
                f"✅ {target['name']} updated!\n\n{summary}\n\nView: {target['url_hint']}",
            )
        else:
            await self._reply(chat_id, f"❌ Failed to update {target['name']}:\n{summary}")

    async def _apply_ui_change(
        self, target: dict, request: str, requester: str
    ) -> tuple[bool, str]:
        """Send current HTML + request to Claude, get back modified HTML."""
        target_path: Path = target["path"]
        try:
            current_html = target_path.read_text()
        except Exception as e:
            return False, f"Could not read {target_path.name}: {e}"

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
            f"Dashboard: {target['name']}\n"
            f"Requested by: {requester}\n"
            f"Change requested: {request}\n\n"
            f"Current HTML:\n{current_html}"
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
            content_blocks = data.get("content", [])
            html_text = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    html_text += block["text"]

            # Strip markdown fences if present
            html_text = html_text.strip()
            if html_text.startswith("```"):
                first_newline = html_text.index("\n")
                html_text = html_text[first_newline + 1:]
            if html_text.endswith("```"):
                html_text = html_text[:-3].rstrip()

            # Validate it looks like HTML
            if "<html" not in html_text.lower() and "<!doctype" not in html_text.lower():
                return False, "Claude did not return valid HTML. Aborting to protect dashboard."

            # Backup current version
            backup = target_path.with_suffix(".html.bak")
            backup.write_text(current_html)

            # Write new version
            target_path.write_text(html_text)

            logger.info(
                "commander.ui_updated",
                dashboard=target["name"],
                requester=requester,
                request=request[:80],
                size=len(html_text),
            )

            usage = data.get("usage", {})
            tokens_in = usage.get("input_tokens", 0)
            tokens_out = usage.get("output_tokens", 0)
            return True, f"Applied ({tokens_in + tokens_out} tokens). Backup saved."

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
