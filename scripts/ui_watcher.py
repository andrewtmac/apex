#!/usr/bin/env python3
"""APEX UI Request Watcher — Hermes-side processor.

Watches for UI change requests written by the Telegram commander,
processes them via MiMo, and sends results back via Telegram.
Runs as a Hermes background terminal process (visible in CLI).

Usage: python ui_watcher.py [--once] [--watch-dir DIR]
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

WATCH_DIR = Path(__file__).parent.parent / "data" / "ui_requests"
PENDING = WATCH_DIR / "pending"
DONE = WATCH_DIR / "done"
FAILED = WATCH_DIR / "failed"

MIMO_BASE = "https://token-plan-sgp.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.5-pro"


def send_telegram(token: str, chat_id: str, text: str):
    try:
        httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text},
            timeout=15,
        )
    except Exception:
        pass


def process_request(req_path: Path, tg_token: str) -> dict:
    """Process a single UI request file."""
    req = json.loads(req_path.read_text())
    target_path = Path(req["target_path"])
    request_text = req["request"]
    requester = req["requester"]
    dashboard_name = req["dashboard_name"]
    chat_id = req.get("chat_id", "")
    url_hint = req.get("url_hint", "")

    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] Processing: {dashboard_name} — {request_text[:60]}")

    current_html = target_path.read_text()

    # Strip <script> for large files
    html_for_llm = current_html
    script_content = ""
    script_tag_start = current_html.lower().find("<script")
    script_tag_end = current_html.rfind("</script>")
    if script_tag_start > 0 and script_tag_end > script_tag_start:
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
        f"Dashboard: {dashboard_name}\n"
        f"Requested by: {requester}\n"
        f"Change requested: {request_text}\n\n"
        f"Current HTML:\n{html_for_llm}"
    )

    mimo_key = os.getenv("MIMO_API_KEY", "")
    if not mimo_key:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("MIMO_API_KEY="):
                    mimo_key = line.split("=", 1)[1].strip()
                    break

    if not mimo_key:
        return {"success": False, "error": "MIMO_API_KEY not found"}

    timeout_s = 180 if len(current_html) > 20000 else 90

    try:
        print(f"  Calling MiMo ({len(html_for_llm)} chars, {timeout_s}s timeout)...")
        resp = httpx.post(
            f"{MIMO_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {mimo_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MIMO_MODEL,
                "reasoning_effort": "medium",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            },
            timeout=timeout_s,
        )

        if resp.status_code != 200:
            return {"success": False, "error": f"MiMo API {resp.status_code}: {resp.text[:300]}"}

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return {"success": False, "error": "MiMo returned no choices"}

        html_text = choices[0].get("message", {}).get("content", "").strip()

        # Strip markdown fences
        if html_text.startswith("```"):
            first_newline = html_text.index("\n")
            html_text = html_text[first_newline + 1:]
        if html_text.endswith("```"):
            html_text = html_text[:-3].rstrip()

        if "<html" not in html_text.lower() and "<!doctype" not in html_text.lower():
            return {"success": False, "error": "MiMo did not return valid HTML"}

        # Restore script section
        if script_content:
            marker = "<!-- [SCRIPT SECTION REMOVED FOR SIZE — will be restored after edit] -->"
            if marker in html_text:
                html_text = html_text.replace(marker, script_content)
            else:
                body_close = html_text.lower().rfind("</body>")
                if body_close > 0:
                    html_text = html_text[:body_close] + script_content + "\n" + html_text[body_close:]

        # Backup + write
        backup = target_path.with_suffix(".html.bak")
        backup.write_text(current_html)
        target_path.write_text(html_text)

        usage = data.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)
        print(f"  ✅ Done! {total_tokens} tokens, {len(html_text)} chars written")

        # Send Telegram success
        if chat_id and tg_token:
            msg = f"✅ {dashboard_name} updated!\n\nApplied ({total_tokens} tokens). Backup saved."
            if url_hint:
                msg += f"\n\nView: {url_hint}"
            send_telegram(tg_token, chat_id, msg)

        return {"success": True, "tokens": total_tokens, "size": len(html_text)}

    except httpx.ReadTimeout:
        err = f"MiMo timed out after {timeout_s}s. File may be too large."
        if chat_id and tg_token:
            send_telegram(tg_token, chat_id, f"❌ Failed to update {dashboard_name}:\n{err}")
        return {"success": False, "error": err}
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        if chat_id and tg_token:
            send_telegram(tg_token, chat_id, f"❌ Failed to update {dashboard_name}:\n{err}")
        return {"success": False, "error": err}


def main():
    once = "--once" in sys.argv
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    PENDING.mkdir(parents=True, exist_ok=True)
    DONE.mkdir(parents=True, exist_ok=True)
    FAILED.mkdir(parents=True, exist_ok=True)

    # Load Telegram token for replies
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not tg_token:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("TELEGRAM_BOT_TOKEN="):
                    tg_token = line.split("=", 1)[1].strip()
                    break

    print(f"UI Watcher started — watching {PENDING}")
    print(f"Telegram replies: {'enabled' if tg_token else 'DISABLED (no token)'}")
    print(f"Mode: {'one-shot' if once else 'continuous (polling every 5s)'}")
    print()

    while True:
        requests = sorted(PENDING.glob("*.json"))
        for req_path in requests:
            try:
                result = process_request(req_path, tg_token)
                result_path = (DONE if result["success"] else FAILED) / req_path.name
                result_path.write_text(json.dumps(result, indent=2))
                req_path.unlink()
            except Exception as e:
                print(f"  ❌ Error processing {req_path.name}: {e}")
                try:
                    req_data = json.loads(req_path.read_text())
                    cid = req_data.get("chat_id", "")
                    if cid and tg_token:
                        send_telegram(tg_token, cid, f"❌ Error: {e}")
                except Exception:
                    pass
                (FAILED / req_path.name).write_text(json.dumps({"success": False, "error": str(e)}))
                req_path.unlink()

        if once:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
