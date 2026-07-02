#!/usr/bin/env python3
"""APEX Code Request Watcher — processes CODE commands via MiMo.

Watches for code change requests from the Telegram commander,
generates changes via MiMo with guardrails, applies and type-checks.

Protected (NEVER modifies):
  - Trading logic, signal generation, entry/exit logic
  - Risk management, circuit breakers, position sizing
  - Core engine, execution, order management
  - Strategy logic, deliberation

Allowed:
  - monitoring/* (dashboard, pnl_tracker, cycle_stats, health_check, alerts)
  - public/* (HTML, CSS, JS for dashboards)
  - reporter.py, learner.py (apex display/stats)
  - dashboard_v2.html, dashboard*.js, dashboard*.css
  - utils/math.ts (display calculations only)

Usage: python code_watcher.py
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

WATCH_DIR = Path(__file__).parent.parent / "data" / "code_requests"
PENDING = WATCH_DIR / "pending"
DONE = WATCH_DIR / "done"
FAILED = WATCH_DIR / "failed"

EPIK_ROOT = Path("/Users/odin-mini/CODING/epik-trade/polymarket-autobot")
APEX_ROOT = Path("/Users/odin-mini/CODING/apex")

MIMO_BASE = "https://token-plan-sgp.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.5-pro"

# ── Guardrails ──────────────────────────────────────────────────────────

# Paths that CODE is NEVER allowed to modify (relative to project root)
PROTECTED_PATTERNS = [
    # epik-trade: trading logic
    "src/core/engine.ts",
    "src/core/engine_clarb.ts",
    "src/core/config.ts",
    "src/core/circuit_breaker.ts",
    "src/core/exit_params.ts",
    "src/core/scheduler.ts",
    "src/core/strategy_graduation.ts",
    "src/execution/*",
    "src/strategy/*",
    "src/deliberation/*",
    "src/analysis/llm_analyst.ts",
    "src/analysis/probability_engine.ts",
    "src/analysis/calibration.ts",
    "src/analysis/category_scoring.ts",
    "src/analysis/multi_model_consensus.ts",
    "src/analysis/cohort_analyzer.ts",
    "src/analysis/regime_detector.ts",
    "src/analysis/backtest.ts",
    "src/backtest/*",
    "src/data/*",
    # apex: trading logic
    "apex/config.py",
    "apex/strategies/*",
    "apex/risk/*",
    "apex/ensemble/*",
    "apex/models/*",
    "apex/data/*",
    "apex/node.py",
    "apex/cli.py",
    # apex scripts: trading agents
    "scripts/apex_v2.py",
    "scripts/weather_agent.py",
    "scripts/crypto_agent.py",
    "scripts/macro_agent.py",
    "scripts/sports_agent.py",
    "scripts/events_agent.py",
    "scripts/run_paper.py",
    "scripts/train_models.py",
    "scripts/collect_data.py",
]

# Paths that CODE IS allowed to modify
ALLOWED_PATTERNS = [
    # epik-trade monitoring & display
    "src/monitoring/*",
    "public/*",
    # apex monitoring & display
    "scripts/reporter.py",
    "scripts/learner.py",
    "scripts/dashboard_v2.html",
    "scripts/dashboard.html",
]

# Project structures to send as context
EPIK_STRUCTURE = """epik-trade/polymarket-autobot/
  src/
    monitoring/
      dashboard.ts      — Express dashboard server (serves HTML, API endpoints)
      pnl_tracker.ts    — P&L calculation, equity tracking, period stats
      cycle_stats.ts    — Trading cycle statistics
      health_check.ts   — System health monitoring
      alerts.ts         — Alert generation and delivery
    analysis/
      self_learner.ts   — Self-learning adaptation (has stats display logic)
    utils/
      math.ts           — Math utilities
    core/
      engine.ts         — [PROTECTED] Main trading engine
      config.ts         — [PROTECTED] Configuration
      circuit_breaker.ts — [PROTECTED] Circuit breaker logic
    strategy/
      (all files)       — [PROTECTED] Trading strategy logic
    execution/
      (all files)       — [PROTECTED] Order execution
  public/
    index.html          — Main dashboard page
    overview.html       — Overview page
    investor.html       — Investor view
    proposals.html      — Proposals page
    leadership.html     — Leadership page
    dashboard.js        — Dashboard JS
    dashboard-v2.js     — Dashboard V2 JS
    dashboard.css       — Dashboard CSS
    dashboard-v2.css    — Dashboard V2 CSS
    leadership.js       — Leadership page JS
    leadership.css      — Leadership page CSS
"""

APEX_STRUCTURE = """apex/
  scripts/
    apex_v2.py          — [PROTECTED] Main trading loop
    weather_agent.py    — [PROTECTED] Weather strategy
    crypto_agent.py     — [PROTECTED] Crypto strategy
    macro_agent.py      — [PROTECTED] Macro strategy
    sports_agent.py     — [PROTECTED] Sports strategy
    events_agent.py     — [PROTECTED] Events strategy
    reporter.py         — Telegram reporter (stats display)
    learner.py          — Strategy learning & stats
    dashboard_v2.html   — Dashboard HTML
    telegram_commander.py — [PROTECTED] Command handler
    ui_watcher.py       — [PROTECTED] UI processing
  apex/
    risk/               — [PROTECTED] Risk management
    ensemble/           — [PROTECTED] Signal ensemble
    models/             — [PROTECTED] ML models
    strategies/         — [PROTECTED] Strategy implementations
"""

SYSTEM_PROMPT = """You are a code editor for trading bot dashboards and monitoring systems.

You receive a code change request and must output the modified files.

CRITICAL RULES:
1. You may ONLY modify files in monitoring/, public/, reporter.py, learner.py, or dashboard files
2. You must NEVER modify trading logic, risk management, circuit breakers, position sizing, order execution, signal generation, or strategy code
3. Return COMPLETE file contents (not diffs)
4. Preserve all existing functionality
5. Keep consistent code style with the existing codebase
6. Do NOT add external dependencies unless absolutely necessary

OUTPUT FORMAT — for each file you modify, output EXACTLY this format:

===FILE: path/to/file.ext===
<complete file content here>
===END===

If you need to create a new file:
===NEWFILE: path/to/file.ext===
<file content here>
===END===

Do NOT output any explanation outside of these blocks. Only output file blocks.
If the request requires modifying protected files, output:
===ERROR: <explanation of why this can't be done>===
"""


def _is_protected(rel_path: str) -> bool:
    """Check if a relative path matches any protected pattern."""
    for pattern in PROTECTED_PATTERNS:
        if pattern.endswith("/*"):
            prefix = pattern[:-2]
            if rel_path.startswith(prefix + "/") or rel_path == prefix:
                return True
        elif rel_path == pattern or rel_path.endswith("/" + pattern):
            return True
    return False


def _load_file_context(request: str) -> str:
    """Build file context for MiMo based on the request."""
    context_parts = []

    # Always include project structure
    context_parts.append("=== PROJECT STRUCTURE ===")
    context_parts.append(EPIK_STRUCTURE)
    context_parts.append(APEX_STRUCTURE)
    context_parts.append("")

    # Try to identify relevant files and include their contents
    request_lower = request.lower()
    relevant_files = []

    # epik-trade monitoring files
    epik_monitoring = EPIK_ROOT / "src" / "monitoring"
    if epik_monitoring.exists():
        for f in epik_monitoring.glob("*.ts"):
            relevant_files.append(("epik", f, f"src/monitoring/{f.name}"))

    # epik-trade public files (HTML/JS only, not CSS)
    epik_public = EPIK_ROOT / "public"
    if epik_public.exists():
        for f in epik_public.glob("*.html"):
            relevant_files.append(("epik", f, f"public/{f.name}"))
        for f in epik_public.glob("*.js"):
            relevant_files.append(("epik", f, f"public/{f.name}"))

    # apex monitoring files
    apex_scripts = APEX_ROOT / "scripts"
    for name in ["reporter.py", "learner.py", "dashboard_v2.html"]:
        f = apex_scripts / name
        if f.exists():
            relevant_files.append(("apex", f, f"scripts/{name}"))

    # Filter to files that are relevant to the request (rough heuristic)
    # Always include all monitoring/dashboard files — they're small enough
    for project, fpath, rel in relevant_files:
        try:
            content = fpath.read_text()
            # Skip very large files (>30KB) unless specifically mentioned
            if len(content) > 30000 and fpath.name.replace(".", " ") not in request_lower:
                context_parts.append(f"=== {project}/{rel} ({len(content)} chars, skipped — too large) ===")
                continue
            context_parts.append(f"=== {project}/{rel} ===")
            context_parts.append(content)
            context_parts.append("")
        except Exception:
            continue

    return "\n".join(context_parts)


def _parse_mimo_response(response: str) -> list[dict]:
    """Parse MiMo's output into file changes."""
    changes = []

    # Check for error
    error_match = re.search(r"===ERROR:\s*(.+?)===", response, re.DOTALL)
    if error_match:
        return [{"error": error_match.group(1).strip()}]

    # Parse FILE blocks (replace existing)
    for match in re.finditer(r"===FILE:\s*(.+?)===\n(.*?)===END===", response, re.DOTALL):
        changes.append({
            "path": match.group(1).strip(),
            "content": match.group(2),
            "action": "replace",
        })

    # Parse NEWFILE blocks
    for match in re.finditer(r"===NEWFILE:\s*(.+?)===\n(.*?)===END===", response, re.DOTALL):
        changes.append({
            "path": match.group(1).strip(),
            "content": match.group(2),
            "action": "create",
        })

    return changes


def _resolve_file_path(rel_path: str) -> tuple[Path, str] | None:
    """Resolve a relative path to an absolute path and project root.
    Returns (absolute_path, project_name) or None if invalid."""
    # Normalize
    rel_path = rel_path.strip().lstrip("/")

    # Try epik-trade
    epik_path = EPIK_ROOT / rel_path
    if epik_path.parent.exists():
        return epik_path, "epik"

    # Try apex
    apex_path = APEX_ROOT / rel_path
    if apex_path.parent.exists():
        return apex_path, "apex"

    return None


def send_telegram(token: str, chat_id: str, text: str):
    try:
        httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text},
            timeout=15,
        )
    except Exception:
        pass


def _type_check_epik() -> tuple[bool, str]:
    """Run TypeScript type check on epik-trade."""
    try:
        result = subprocess.run(
            ["npx", "tsc", "--noEmit"],
            cwd=EPIK_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return True, "tsc --noEmit passed"
        return False, f"tsc errors:\n{result.stdout[:500]}"
    except Exception as e:
        return False, f"tsc failed: {e}"


def _type_check_apex(files: list[str]) -> tuple[bool, str]:
    """Run Python syntax check on modified apex files."""
    for f in files:
        if f.endswith(".py"):
            try:
                result = subprocess.run(
                    ["python3", "-m", "py_compile", f],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if result.returncode != 0:
                    return False, f"py_compile error in {f}: {result.stderr[:300]}"
            except Exception as e:
                return False, f"py_compile failed for {f}: {e}"
    return True, "py_compile passed"


def process_code_request(req_path: Path, tg_token: str) -> dict:
    """Process a single CODE request."""
    req = json.loads(req_path.read_text())
    request_text = req["request"]
    requester = req["requester"]
    chat_id = req.get("chat_id", "")

    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"\n[{ts}] CODE request from {requester}: {request_text[:80]}")

    # 1. Build context
    print("  Building context...")
    context = _load_file_context(request_text)
    print(f"  Context: {len(context)} chars")

    # 2. Call MiMo
    mimo_key = os.getenv("MIMO_API_KEY", "")
    if not mimo_key:
        env_path = APEX_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("MIMO_API_KEY="):
                    mimo_key = line.split("=", 1)[1].strip()
                    break

    if not mimo_key:
        return {"success": False, "error": "MIMO_API_KEY not found"}

    user_msg = (
        f"Request from: {requester}\n"
        f"Request: {request_text}\n\n"
        f"Relevant source files:\n{context}"
    )

    timeout_s = 180
    try:
        print(f"  Calling MiMo ({len(user_msg)} chars)...")
        resp = httpx.post(
            f"{MIMO_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {mimo_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MIMO_MODEL,
                "reasoning_effort": "high",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
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

        mimo_output = choices[0].get("message", {}).get("content", "")

    except httpx.ReadTimeout:
        return {"success": False, "error": f"MiMo timed out after {timeout_s}s"}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}

    # 3. Parse response
    print("  Parsing response...")
    changes = _parse_mimo_response(mimo_output)

    if not changes:
        return {"success": False, "error": "MiMo returned no file changes"}

    if changes and "error" in changes[0]:
        return {"success": False, "error": f"MiMo refused: {changes[0]['error']}"}

    # 4. Validate guardrails
    print(f"  Validating {len(changes)} file(s)...")
    for change in changes:
        rel_path = change["path"]

        # Check if protected
        # Determine which project this belongs to
        if rel_path.startswith("epik/") or rel_path.startswith("src/") or rel_path.startswith("public/"):
            check_path = rel_path.removeprefix("epik/")
        elif rel_path.startswith("apex/") or rel_path.startswith("scripts/"):
            check_path = rel_path.removeprefix("apex/")
        else:
            check_path = rel_path

        if _is_protected(check_path):
            return {
                "success": False,
                "error": f"BLOCKED: {rel_path} is a protected file (trading logic). "
                         f"CODE cannot modify trading strategies, risk, or execution code.",
            }

    # 5. Apply changes
    print("  Applying changes...")
    applied = []
    modified_epik_files = []
    modified_apex_files = []

    for change in changes:
        resolved = _resolve_file_path(change["path"])
        if not resolved:
            print(f"  ⚠️  Cannot resolve: {change['path']}")
            continue

        abs_path, project = resolved

        if change["action"] == "replace" and not abs_path.exists():
            print(f"  ⚠️  File not found (use NEWFILE to create): {change['path']}")
            continue

        # Backup existing
        if abs_path.exists():
            backup = abs_path.with_suffix(abs_path.suffix + ".bak")
            backup.write_text(abs_path.read_text())

        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(change["content"])
        applied.append(change["path"])

        if project == "epik":
            modified_epik_files.append(str(abs_path))
        else:
            modified_apex_files.append(str(abs_path))

        print(f"  ✅ {change['action']}: {change['path']}")

    if not applied:
        return {"success": False, "error": "No files could be applied"}

    # 6. Type check
    print("  Running type checks...")
    all_ok = True
    check_results = []

    if modified_epik_files:
        ok, msg = _type_check_epik()
        check_results.append(f"epik tsc: {msg}")
        if not ok:
            all_ok = False

    if modified_apex_files:
        ok, msg = _type_check_apex(modified_apex_files)
        check_results.append(f"apex py: {msg}")
        if not ok:
            all_ok = False

    # 7. Rollback if type check failed
    if not all_ok:
        print("  ❌ Type check failed — rolling back!")
        for change in changes:
            resolved = _resolve_file_path(change["path"])
            if not resolved:
                continue
            abs_path, _ = resolved
            backup = abs_path.with_suffix(abs_path.suffix + ".bak")
            if backup.exists():
                abs_path.write_text(backup.read_text())
                print(f"  ↩️  Rolled back: {change['path']}")

        error_detail = "\n".join(check_results)
        if chat_id and tg_token:
            send_telegram(tg_token, chat_id, f"❌ Code change rolled back (type check failed):\n\n{error_detail}")
        return {"success": False, "error": f"Type check failed, rolled back:\n{error_detail}"}

    # 8. Success
    usage = data.get("usage", {})
    total_tokens = usage.get("total_tokens", 0)
    print(f"  ✅ Done! {len(applied)} file(s) changed, {total_tokens} tokens")

    if chat_id and tg_token:
        file_list = "\n".join(f"  • {f}" for f in applied)
        checks = "\n".join(check_results)
        msg = (
            f"✅ Code change applied!\n\n"
            f"Files modified:\n{file_list}\n\n"
            f"Type checks: {checks}\n\n"
            f"({total_tokens} tokens used)"
        )
        send_telegram(tg_token, chat_id, msg)

    return {"success": True, "files": applied, "tokens": total_tokens}


def main():
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    PENDING.mkdir(parents=True, exist_ok=True)
    DONE.mkdir(parents=True, exist_ok=True)
    FAILED.mkdir(parents=True, exist_ok=True)

    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not tg_token:
        env_path = APEX_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("TELEGRAM_BOT_TOKEN="):
                    tg_token = line.split("=", 1)[1].strip()
                    break

    print(f"Code Watcher started — watching {PENDING}")
    print(f"Telegram replies: {'enabled' if tg_token else 'DISABLED'}")
    print(f"Protected patterns: {len(PROTECTED_PATTERNS)}")
    print()

    while True:
        requests = sorted(PENDING.glob("*.json"))
        for req_path in requests:
            try:
                result = process_code_request(req_path, tg_token)
                result_path = (DONE if result["success"] else FAILED) / req_path.name
                result_path.write_text(json.dumps(result, indent=2, default=str))
                req_path.unlink()
            except Exception as e:
                print(f"  ❌ Error: {e}")
                try:
                    req_data = json.loads(req_path.read_text())
                    cid = req_data.get("chat_id", "")
                    if cid and tg_token:
                        send_telegram(tg_token, cid, f"❌ Code change error: {e}")
                except Exception:
                    pass
                (FAILED / req_path.name).write_text(json.dumps({"success": False, "error": str(e)}))
                req_path.unlink()

        time.sleep(5)


if __name__ == "__main__":
    main()
