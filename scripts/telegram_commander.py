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
import json
import os
import time
from datetime import datetime, timezone
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
    lines = ["APEX Commander — Available commands:\n"]
    lines.append("  UI <change>")
    lines.append("    Modify dashboard HTML/CSS")
    lines.append("  UI APEX <change>  |  UI EPIK <page> <change>")
    lines.append("    Target specific dashboard")
    lines.append("")
    lines.append("  CODE <request>")
    lines.append("    Modify monitoring/stats/display code (NOT trading logic)")
    lines.append("")
    lines.append("Examples:")
    lines.append("  UI APEX change equity card color to blue")
    lines.append("  UI EPIK investor make the table sortable")
    lines.append("  CODE recalculate all P&L stats on epik dashboards")
    lines.append("  CODE add a 7-day rolling win rate to the apex dashboard")
    lines.append("")
    lines.append("Protected (CODE will NOT modify):")
    lines.append("  Trading logic, risk management, circuit breakers,")
    lines.append("  position sizing, order execution, signal generation")
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
        self.mimo_key = os.getenv("MIMO_API_KEY", "")
        self._offset = 0
        self._running = False

    @property
    def configured(self) -> bool:
        return bool(self.token and self.authorized and self.mimo_key)

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

        # 2. Route by command type
        upper = text.upper()
        if upper.startswith("UI"):
            await self._handle_ui_command(chat_id, text, sender_name)
        elif upper.startswith("CODE"):
            await self._handle_code_command(chat_id, text[4:].strip(), sender_name)
        else:
            await self._reply(
                chat_id,
                "Available commands:\n\n"
                '"UI <change>" — modify dashboard HTML/CSS\n'
                '"CODE <request>" — modify monitoring/stats code\n\n'
                "Send UI help or CODE help for details.",
            )

    async def _handle_ui_command(self, chat_id: str, text: str, sender_name: str):
        """Handle UI ... commands."""
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

        # 6. Immediate ack, then queue for watcher
        await self._reply(chat_id, "Received, working on it")
        await self._queue_ui_request(target, change, sender_name, chat_id)

    async def _queue_ui_request(
        self, target: dict, request: str, requester: str, chat_id: str
    ):
        """Write UI request to queue file for the watcher to process."""
        queue_dir = Path(__file__).parent.parent / "data" / "ui_requests" / "pending"
        queue_dir.mkdir(parents=True, exist_ok=True)

        ts = int(time.time() * 1000)
        req_file = queue_dir / f"{ts}.json"
        req_file.write_text(json.dumps({
            "target_path": str(target["path"]),
            "dashboard_name": target["name"],
            "url_hint": target.get("url_hint", ""),
            "request": request,
            "requester": requester,
            "chat_id": chat_id,
            "queued_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))

        logger.info("commander.queued", dashboard=target["name"], file=req_file.name)

    async def _handle_code_command(self, chat_id: str, request: str, sender_name: str):
        """Handle CODE ... commands for monitoring/stats code changes."""
        if not request or request.lower() == "help":
            await self._reply(chat_id, _build_help())
            return

        await self._reply(chat_id, "Received, working on it")
        await self._queue_code_request(request, sender_name, chat_id)

    async def _queue_code_request(self, request: str, requester: str, chat_id: str):
        """Write CODE request to queue file for the watcher to process."""
        queue_dir = Path(__file__).parent.parent / "data" / "code_requests" / "pending"
        queue_dir.mkdir(parents=True, exist_ok=True)

        ts = int(time.time() * 1000)
        req_file = queue_dir / f"{ts}.json"
        req_file.write_text(json.dumps({
            "type": "code_change",
            "request": request,
            "requester": requester,
            "chat_id": chat_id,
            "queued_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))

        logger.info("commander.code_queued", request=request[:60], file=req_file.name)

    async def _apply_ui_change(
        self, target: dict, request: str, requester: str
    ) -> tuple[bool, str]:
        """Send current HTML + request to MiMo, get back modified HTML."""
        target_path: Path = target["path"]
        try:
            current_html = target_path.read_text()
        except Exception as e:
            return False, f"Could not read {target_path.name}: {e}"

        # For large files, strip the <script> block to reduce token load.
        # MiMo only needs CSS + HTML structure for UI changes.
        # Use the LAST <script> tag to preserve small inline scripts (theme toggle etc).
        html_for_llm = current_html
        script_content = ""
        lower = current_html.lower()
        script_tag_start = lower.rfind("<script")
        script_tag_end = current_html.rfind("</script>")
        if script_tag_start > 0 and script_tag_end > script_tag_start:
            # Find the closing > of the <script> tag
            tag_close = current_html.index(">", script_tag_start) + 1
            script_content = current_html[script_tag_start:script_tag_end + len("</script>")]
            html_for_llm = (
                current_html[:tag_close]
                + "\n<!-- [SCRIPT SECTION REMOVED FOR SIZE — will be restored after edit] -->\n"
                + current_html[script_tag_end + len("</script>"):]
            )

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
            f"Current HTML:\n{html_for_llm}"
        )

        mimo_key = os.getenv("MIMO_API_KEY", "")
        mimo_base = os.getenv("MIMO_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1")
        mimo_model = os.getenv("MIMO_MODEL", "mimo-v2.5-pro")

        if not mimo_key:
            return False, "MIMO_API_KEY not set in .env"

        timeout_s = 180 if len(current_html) > 20000 else 90

        try:
            logger.info(
                "commander.mimo_call",
                dashboard=target["name"],
                html_chars=len(html_for_llm),
                timeout=timeout_s,
            )
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                resp = await client.post(
                    f"{mimo_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {mimo_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": mimo_model,
                        "reasoning_effort": "medium",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                    },
                )

            if resp.status_code != 200:
                return False, f"MiMo API error {resp.status_code}: {resp.text[:300]}"

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return False, "MiMo returned no choices"

            html_text = choices[0].get("message", {}).get("content", "")

            # Strip markdown fences if present
            html_text = html_text.strip()
            if html_text.startswith("```"):
                first_newline = html_text.index("\n")
                html_text = html_text[first_newline + 1:]
            if html_text.endswith("```"):
                html_text = html_text[:-3].rstrip()

            # Validate it looks like HTML
            if "<html" not in html_text.lower() and "<!doctype" not in html_text.lower():
                return False, "MiMo did not return valid HTML. Aborting to protect dashboard."

            # Restore the original script section if it was stripped
            if script_content:
                marker = "<!-- [SCRIPT SECTION REMOVED FOR SIZE — will be restored after edit] -->"
                if marker in html_text:
                    html_text = html_text.replace(marker, script_content)
                else:
                    # MiMo may have rewritten the marker — try to restore before </body>
                    body_close = html_text.lower().rfind("</body>")
                    if body_close > 0:
                        html_text = html_text[:body_close] + script_content + "\n" + html_text[body_close:]

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
            total_tokens = usage.get("total_tokens", 0)
            return True, f"Applied ({total_tokens} tokens). Backup saved."

        except httpx.ReadTimeout:
            return False, f"MiMo timed out after {timeout_s}s. File may be too large."
        except httpx.ConnectError as e:
            return False, f"Cannot reach MiMo API: {e}"
        except Exception as e:
            return False, f"Error ({type(e).__name__}): {e}"

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
