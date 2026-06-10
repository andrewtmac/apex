"""FastAPI web dashboard for APEX monitoring.

Endpoints:
- GET / - Dashboard HTML page
- GET /api/health - System health
- GET /api/portfolio - Portfolio state
- GET /api/performance - Strategy performance
- GET /api/models - Model status
- GET /api/risk - Risk metrics
- GET /api/data - Data pipeline health
- WebSocket /ws - Real-time updates
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="APEX Dashboard",
    description="Adaptive Prediction EXchange monitoring dashboard",
    version="0.1.0",
)

# Global state holders (injected on startup)
_db_url: str = ""
_redis_url: str = ""
_model_registry = None
_alert_manager = None


def configure_dashboard(
    db_url: str,
    redis_url: str,
    model_registry: Any = None,
    alert_manager: Any = None,
) -> None:
    """Inject dependencies into the dashboard module."""
    global _db_url, _redis_url, _model_registry, _alert_manager
    _db_url = db_url
    _redis_url = redis_url
    _model_registry = model_registry
    _alert_manager = alert_manager


# ---------------------------------------------------------------------------
# WebSocket manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.active_connections.remove(conn)


ws_manager = ConnectionManager()


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page with live metrics."""
    return HTMLResponse(content=_DASHBOARD_HTML)


@app.get("/api/health")
async def health():
    """System health check."""
    from apex.monitoring.data_health import DataHealthMonitor

    monitor = DataHealthMonitor(redis_url=_redis_url, db_url=_db_url)
    try:
        await monitor.connect()
        result = await monitor.check_all()
        return JSONResponse(content=result)
    except Exception as exc:
        return JSONResponse(
            content={"status": "error", "error": str(exc)},
            status_code=500,
        )
    finally:
        await monitor.close()


@app.get("/api/portfolio")
async def portfolio():
    """Current portfolio state."""
    import asyncpg

    try:
        pool = await asyncpg.create_pool(_db_url, min_size=1, max_size=3, command_timeout=10)
        try:
            row = await pool.fetchrow(
                "SELECT * FROM portfolio_snapshots ORDER BY time DESC LIMIT 1"
            )
            if row:
                return JSONResponse(content={
                    "time": row["time"].isoformat(),
                    "total_equity": float(row["total_equity"]),
                    "poly_equity": float(row["poly_equity"]),
                    "kalshi_equity": float(row["kalshi_equity"]),
                    "tt_equity": float(row["tt_equity"]),
                    "open_positions": int(row["open_positions"]),
                    "deployed_pct": float(row["deployed_pct"]),
                    "drawdown_pct": float(row["drawdown_pct"]),
                    "cvar_95": float(row["cvar_95"]) if row["cvar_95"] else None,
                    "regime": row["regime"],
                    "breaker_level": row["breaker_level"],
                })
            return JSONResponse(content={"status": "no_data"})
        finally:
            await pool.close()
    except Exception as exc:
        return JSONResponse(
            content={"status": "error", "error": str(exc)},
            status_code=500,
        )


@app.get("/api/performance")
async def performance():
    """Strategy performance metrics."""
    from apex.monitoring.performance import PerformanceTracker

    tracker = PerformanceTracker(db_url=_db_url)
    try:
        await tracker.connect()
        report = await tracker.daily_report()
        return JSONResponse(content=report)
    except Exception as exc:
        return JSONResponse(
            content={"status": "error", "error": str(exc)},
            status_code=500,
        )
    finally:
        await tracker.close()


@app.get("/api/models")
async def models():
    """Model status and versions."""
    if _model_registry is None:
        return JSONResponse(content={"status": "registry_not_configured"})

    model_names = [
        "xgboost_prob", "lgbm_return", "tft_quantile",
        "lstm_regime", "ppo_position_manager", "finbert_sentiment",
    ]

    result: dict[str, Any] = {}
    for name in model_names:
        try:
            versions = _model_registry.list_versions(name)
            prod = [v for v in versions if v.get("is_production")]
            result[name] = {
                "total_versions": len(versions),
                "production_version": prod[-1]["version_id"] if prod else None,
                "production_metrics": prod[-1].get("metrics", {}) if prod else {},
                "latest_version": versions[-1]["version_id"] if versions else None,
                "latest_timestamp": versions[-1].get("timestamp") if versions else None,
            }
        except FileNotFoundError:
            result[name] = {"total_versions": 0, "production_version": None}

    return JSONResponse(content=result)


@app.get("/api/risk")
async def risk():
    """Risk metrics."""
    import asyncpg

    try:
        pool = await asyncpg.create_pool(_db_url, min_size=1, max_size=3, command_timeout=10)
        try:
            # Latest snapshot
            latest = await pool.fetchrow(
                "SELECT * FROM portfolio_snapshots ORDER BY time DESC LIMIT 1"
            )
            # 30-day history for drawdown curve
            history = await pool.fetch(
                """
                SELECT time_bucket('1 hour', time) AS hour,
                       last(drawdown_pct, time) AS drawdown,
                       last(cvar_95, time) AS cvar,
                       last(regime, time) AS regime,
                       last(breaker_level, time) AS breaker
                FROM portfolio_snapshots
                WHERE time >= NOW() - INTERVAL '7 days'
                GROUP BY hour
                ORDER BY hour
                """
            )

            return JSONResponse(content={
                "current": {
                    "drawdown_pct": float(latest["drawdown_pct"]) if latest else 0,
                    "cvar_95": float(latest["cvar_95"]) if latest and latest["cvar_95"] else None,
                    "regime": latest["regime"] if latest else "NORMAL",
                    "breaker_level": latest["breaker_level"] if latest else "GREEN",
                    "deployed_pct": float(latest["deployed_pct"]) if latest else 0,
                },
                "history": [
                    {
                        "time": r["hour"].isoformat(),
                        "drawdown": float(r["drawdown"]) if r["drawdown"] else 0,
                        "cvar": float(r["cvar"]) if r["cvar"] else None,
                        "regime": r["regime"],
                        "breaker": r["breaker"],
                    }
                    for r in history
                ],
            })
        finally:
            await pool.close()
    except Exception as exc:
        return JSONResponse(
            content={"status": "error", "error": str(exc)},
            status_code=500,
        )


@app.get("/api/data")
async def data_health():
    """Data pipeline health."""
    from apex.monitoring.data_health import DataHealthMonitor

    monitor = DataHealthMonitor(redis_url=_redis_url, db_url=_db_url)
    try:
        await monitor.connect()
        result = await monitor.check_all()
        return JSONResponse(content=result)
    except Exception as exc:
        return JSONResponse(
            content={"status": "error", "error": str(exc)},
            status_code=500,
        )
    finally:
        await monitor.close()


@app.get("/api/trades")
async def recent_trades(limit: int = 50):
    """Recent trade history."""
    import asyncpg

    try:
        pool = await asyncpg.create_pool(_db_url, min_size=1, max_size=3, command_timeout=10)
        try:
            rows = await pool.fetch(
                """
                SELECT id, time, market_id, venue, strategy,
                       direction, side, price, quantity, cost, fee
                FROM trades
                ORDER BY time DESC
                LIMIT $1
                """,
                limit,
            )
            trades = [
                {
                    "id": r["id"],
                    "time": r["time"].isoformat(),
                    "market_id": r["market_id"][:30],
                    "venue": r["venue"],
                    "strategy": r["strategy"],
                    "direction": r["direction"],
                    "side": r["side"],
                    "price": float(r["price"]),
                    "quantity": float(r["quantity"]),
                    "cost": float(r["cost"]),
                    "fee": float(r["fee"]) if r["fee"] else 0,
                }
                for r in rows
            ]
            return JSONResponse(content={"trades": trades})
        finally:
            await pool.close()
    except Exception as exc:
        return JSONResponse(
            content={"status": "error", "error": str(exc)},
            status_code=500,
        )


@app.get("/api/equity-curve")
async def equity_curve(days: int = 30):
    """Equity curve data for charting."""
    import asyncpg

    try:
        pool = await asyncpg.create_pool(_db_url, min_size=1, max_size=3, command_timeout=10)
        try:
            rows = await pool.fetch(
                """
                SELECT time_bucket('1 hour', time) AS hour,
                       last(total_equity, time) AS equity,
                       last(poly_equity, time) AS poly,
                       last(kalshi_equity, time) AS kalshi,
                       last(tt_equity, time) AS tt
                FROM portfolio_snapshots
                WHERE time >= NOW() - make_interval(days => $1)
                GROUP BY hour
                ORDER BY hour
                """,
                days,
            )
            return JSONResponse(content={
                "curve": [
                    {
                        "time": r["hour"].isoformat(),
                        "total": float(r["equity"]),
                        "polymarket": float(r["poly"]) if r["poly"] else 0,
                        "kalshi": float(r["kalshi"]) if r["kalshi"] else 0,
                        "tastytrade": float(r["tt"]) if r["tt"] else 0,
                    }
                    for r in rows
                ]
            })
        finally:
            await pool.close()
    except Exception as exc:
        return JSONResponse(
            content={"status": "error", "error": str(exc)},
            status_code=500,
        )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time WebSocket updates.

    Pushes portfolio snapshots, trade notifications, and health
    status updates every 5 seconds.
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            # Push updates periodically
            try:
                update = {
                    "type": "heartbeat",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                # Try to include latest portfolio data
                import asyncpg

                pool = await asyncpg.create_pool(
                    _db_url, min_size=1, max_size=2, command_timeout=5
                )
                try:
                    row = await pool.fetchrow(
                        "SELECT total_equity, drawdown_pct, regime, breaker_level "
                        "FROM portfolio_snapshots ORDER BY time DESC LIMIT 1"
                    )
                    if row:
                        update["portfolio"] = {
                            "total_equity": float(row["total_equity"]),
                            "drawdown_pct": float(row["drawdown_pct"]),
                            "regime": row["regime"],
                            "breaker_level": row["breaker_level"],
                        }
                finally:
                    await pool.close()

                await websocket.send_json(update)
            except Exception:
                pass

            await asyncio.sleep(5)

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Embedded dashboard HTML/CSS/JS
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>APEX Dashboard</title>
<style>
  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #c9d1d9;
    --text-dim: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --red: #f85149;
    --yellow: #d29922;
    --orange: #db6d28;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    font-size: 14px;
    line-height: 1.5;
    padding: 16px;
  }
  h1 { font-size: 20px; font-weight: 600; margin-bottom: 16px; color: var(--accent); }
  h2 { font-size: 15px; font-weight: 600; margin-bottom: 12px; color: var(--text); }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 16px;
    margin-bottom: 16px;
  }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
  }
  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }
  .stat-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
  }
  .stat-label { color: var(--text-dim); }
  .stat-value { font-weight: 500; font-variant-numeric: tabular-nums; }
  .positive { color: var(--green); }
  .negative { color: var(--red); }
  .neutral { color: var(--text-dim); }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
  }
  .badge-green { background: rgba(63,185,80,0.15); color: var(--green); }
  .badge-yellow { background: rgba(210,153,34,0.15); color: var(--yellow); }
  .badge-red { background: rgba(248,81,73,0.15); color: var(--red); }
  .badge-blue { background: rgba(88,166,255,0.15); color: var(--accent); }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  th {
    text-align: left;
    padding: 6px 8px;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    font-weight: 500;
  }
  td {
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
  }
  tr:hover td { background: rgba(88,166,255,0.05); }
  .chart-container {
    width: 100%;
    height: 200px;
    position: relative;
  }
  canvas { width: 100% !important; height: 100% !important; }
  .status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
  }
  .dot-green { background: var(--green); }
  .dot-yellow { background: var(--yellow); }
  .dot-red { background: var(--red); }
  .dot-gray { background: var(--text-dim); }
  #last-update { color: var(--text-dim); font-size: 12px; }
  .full-width { grid-column: 1 / -1; }
  .header-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }
</style>
</head>
<body>

<div class="header-bar">
  <h1>APEX Dashboard</h1>
  <span id="last-update">Connecting...</span>
</div>

<!-- Portfolio Overview -->
<div class="grid">
  <div class="card">
    <div class="card-header">
      <h2>Portfolio</h2>
      <span id="breaker-badge" class="badge badge-green">GREEN</span>
    </div>
    <div id="portfolio-stats">
      <div class="stat-row"><span class="stat-label">Total Equity</span><span class="stat-value" id="total-equity">--</span></div>
      <div class="stat-row"><span class="stat-label">Daily P&L</span><span class="stat-value" id="daily-pnl">--</span></div>
      <div class="stat-row"><span class="stat-label">Deployed</span><span class="stat-value" id="deployed-pct">--</span></div>
      <div class="stat-row"><span class="stat-label">Drawdown</span><span class="stat-value" id="drawdown-pct">--</span></div>
      <div class="stat-row"><span class="stat-label">CVaR 95</span><span class="stat-value" id="cvar-95">--</span></div>
      <div class="stat-row"><span class="stat-label">Regime</span><span class="stat-value" id="regime">--</span></div>
      <div class="stat-row"><span class="stat-label">Open Positions</span><span class="stat-value" id="open-positions">--</span></div>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><h2>Venue Breakdown</h2></div>
    <div id="venue-stats">
      <div class="stat-row"><span class="stat-label">Polymarket</span><span class="stat-value" id="poly-equity">--</span></div>
      <div class="stat-row"><span class="stat-label">Kalshi</span><span class="stat-value" id="kalshi-equity">--</span></div>
      <div class="stat-row"><span class="stat-label">TastyTrade</span><span class="stat-value" id="tt-equity">--</span></div>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><h2>Risk Metrics</h2></div>
    <div id="risk-stats">
      <div class="stat-row"><span class="stat-label">Max Drawdown (7d)</span><span class="stat-value" id="max-dd-7d">--</span></div>
      <div class="stat-row"><span class="stat-label">Regime History</span><span class="stat-value" id="regime-history">--</span></div>
      <div class="stat-row"><span class="stat-label">Breaker Changes (24h)</span><span class="stat-value" id="breaker-changes">--</span></div>
    </div>
  </div>
</div>

<!-- Equity Curve -->
<div class="grid">
  <div class="card full-width">
    <div class="card-header"><h2>Equity Curve (30 Days)</h2></div>
    <div class="chart-container">
      <canvas id="equity-chart"></canvas>
    </div>
  </div>
</div>

<!-- Strategy Performance & Data Health -->
<div class="grid">
  <div class="card">
    <div class="card-header"><h2>Strategy Performance</h2></div>
    <table>
      <thead><tr><th>Strategy</th><th>Trades</th><th>Win%</th><th>PnL</th><th>Sharpe</th></tr></thead>
      <tbody id="strategy-table"><tr><td colspan="5" class="neutral">Loading...</td></tr></tbody>
    </table>
  </div>

  <div class="card">
    <div class="card-header"><h2>Data Pipeline</h2></div>
    <table>
      <thead><tr><th>Source</th><th>Status</th><th>Lag</th><th>Rate/h</th></tr></thead>
      <tbody id="data-table"><tr><td colspan="4" class="neutral">Loading...</td></tr></tbody>
    </table>
  </div>
</div>

<!-- Models & Recent Trades -->
<div class="grid">
  <div class="card">
    <div class="card-header"><h2>Model Status</h2></div>
    <table>
      <thead><tr><th>Model</th><th>Version</th><th>Key Metric</th></tr></thead>
      <tbody id="model-table"><tr><td colspan="3" class="neutral">Loading...</td></tr></tbody>
    </table>
  </div>

  <div class="card">
    <div class="card-header"><h2>Recent Trades</h2></div>
    <table>
      <thead><tr><th>Time</th><th>Venue</th><th>Dir</th><th>Price</th><th>Size</th></tr></thead>
      <tbody id="trades-table"><tr><td colspan="5" class="neutral">Loading...</td></tr></tbody>
    </table>
  </div>
</div>

<script>
// Minimal chart library (no external dependencies)
class SimpleChart {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.data = [];
  }
  setData(labels, values) {
    this.data = { labels, values };
    this.draw();
  }
  draw() {
    const { labels, values } = this.data;
    if (!values || values.length === 0) return;

    const canvas = this.canvas;
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * (window.devicePixelRatio || 1);
    canvas.height = rect.height * (window.devicePixelRatio || 1);
    const ctx = this.ctx;
    ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);

    const w = rect.width;
    const h = rect.height;
    const pad = { top: 10, right: 10, bottom: 30, left: 60 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    const minV = Math.min(...values);
    const maxV = Math.max(...values);
    const range = maxV - minV || 1;

    // Grid lines
    ctx.strokeStyle = '#30363d';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + plotH * (1 - i / 4);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();

      const val = minV + range * i / 4;
      ctx.fillStyle = '#8b949e';
      ctx.font = '11px -apple-system, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText('$' + val.toFixed(0), pad.left - 6, y + 4);
    }

    // Line
    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {
      const x = pad.left + (i / (values.length - 1)) * plotW;
      const y = pad.top + plotH * (1 - (values[i] - minV) / range);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Fill under line
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.closePath();
    const gradient = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
    gradient.addColorStop(0, 'rgba(88,166,255,0.15)');
    gradient.addColorStop(1, 'rgba(88,166,255,0)');
    ctx.fillStyle = gradient;
    ctx.fill();

    // X-axis labels (first, mid, last)
    ctx.fillStyle = '#8b949e';
    ctx.font = '11px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    if (labels.length > 0) {
      ctx.fillText(labels[0], pad.left, h - 8);
      if (labels.length > 2) {
        const midIdx = Math.floor(labels.length / 2);
        ctx.fillText(labels[midIdx], pad.left + plotW / 2, h - 8);
      }
      ctx.fillText(labels[labels.length - 1], pad.left + plotW, h - 8);
    }
  }
}

const equityChart = new SimpleChart('equity-chart');

// Format helpers
function fmtUsd(v) { return '$' + (v || 0).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2}); }
function fmtPct(v) { return ((v || 0) * 100).toFixed(2) + '%'; }
function pnlClass(v) { return v >= 0 ? 'positive' : 'negative'; }
function statusDot(s) {
  const cls = s === 'healthy' ? 'dot-green' : s === 'degraded' ? 'dot-yellow' : s === 'critical' ? 'dot-red' : 'dot-gray';
  return '<span class="status-dot ' + cls + '"></span>';
}
function breakerBadge(level) {
  const cls = level === 'GREEN' ? 'badge-green' : level === 'YELLOW' ? 'badge-yellow' : 'badge-red';
  return '<span class="badge ' + cls + '">' + level + '</span>';
}

// Fetch helpers
async function fetchJson(url) {
  try {
    const r = await fetch(url);
    return await r.json();
  } catch(e) {
    console.error('Fetch failed:', url, e);
    return null;
  }
}

// Update functions
async function updatePortfolio() {
  const d = await fetchJson('/api/portfolio');
  if (!d || d.status === 'error') return;
  document.getElementById('total-equity').textContent = fmtUsd(d.total_equity);
  document.getElementById('deployed-pct').textContent = fmtPct(d.deployed_pct);
  document.getElementById('drawdown-pct').textContent = fmtPct(d.drawdown_pct);
  document.getElementById('drawdown-pct').className = 'stat-value ' + (d.drawdown_pct > 0.1 ? 'negative' : 'neutral');
  document.getElementById('cvar-95').textContent = d.cvar_95 !== null ? fmtUsd(d.cvar_95) : '--';
  document.getElementById('regime').textContent = d.regime || '--';
  document.getElementById('open-positions').textContent = d.open_positions || 0;
  document.getElementById('poly-equity').textContent = fmtUsd(d.poly_equity);
  document.getElementById('kalshi-equity').textContent = fmtUsd(d.kalshi_equity);
  document.getElementById('tt-equity').textContent = fmtUsd(d.tt_equity);
  document.getElementById('breaker-badge').innerHTML = breakerBadge(d.breaker_level || 'GREEN');
}

async function updatePerformance() {
  const d = await fetchJson('/api/performance');
  if (!d || d.status === 'error') return;

  // Daily PnL
  const pnl = d.portfolio?.daily_pnl || 0;
  const pnlEl = document.getElementById('daily-pnl');
  pnlEl.textContent = (pnl >= 0 ? '+' : '') + fmtUsd(pnl);
  pnlEl.className = 'stat-value ' + pnlClass(pnl);

  // Strategy table
  const tbody = document.getElementById('strategy-table');
  const strats = d.strategies || {};
  if (Object.keys(strats).length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="neutral">No strategies</td></tr>';
    return;
  }
  let html = '';
  for (const [name, s] of Object.entries(strats)) {
    html += '<tr>';
    html += '<td>' + name + '</td>';
    html += '<td>' + (s.trades || 0) + '</td>';
    html += '<td>' + ((s.win_rate || 0) * 100).toFixed(1) + '%</td>';
    html += '<td class="' + pnlClass(s.net_pnl) + '">' + fmtUsd(s.net_pnl) + '</td>';
    html += '<td>' + (s.sharpe_ratio || 0).toFixed(2) + '</td>';
    html += '</tr>';
  }
  tbody.innerHTML = html;
}

async function updateEquityCurve() {
  const d = await fetchJson('/api/equity-curve?days=30');
  if (!d || !d.curve || d.curve.length === 0) return;
  const labels = d.curve.map(p => p.time.split('T')[0]);
  const values = d.curve.map(p => p.total);
  equityChart.setData(labels, values);
}

async function updateDataHealth() {
  const d = await fetchJson('/api/data');
  if (!d || d.status === 'error') return;

  const tbody = document.getElementById('data-table');
  const sources = d.sources || {};
  if (Object.keys(sources).length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" class="neutral">No sources</td></tr>';
    return;
  }
  let html = '';
  for (const [name, s] of Object.entries(sources)) {
    const shortName = name.replace('apex:', '').replace(':', '/');
    const lag = s.staleness_seconds !== null ? s.staleness_seconds.toFixed(0) + 's' : '--';
    const rate = s.throughput_per_hour || '--';
    html += '<tr>';
    html += '<td>' + statusDot(s.status) + shortName + '</td>';
    html += '<td>' + s.status + '</td>';
    html += '<td>' + lag + '</td>';
    html += '<td>' + rate + '</td>';
    html += '</tr>';
  }
  tbody.innerHTML = html;
}

async function updateModels() {
  const d = await fetchJson('/api/models');
  if (!d || d.status) return;

  const tbody = document.getElementById('model-table');
  let html = '';
  for (const [name, m] of Object.entries(d)) {
    const version = m.production_version || '--';
    const metrics = m.production_metrics || {};
    let keyMetric = '--';
    if (metrics.accuracy) keyMetric = 'acc: ' + metrics.accuracy.toFixed(3);
    else if (metrics.brier_score) keyMetric = 'brier: ' + metrics.brier_score.toFixed(3);
    else if (metrics.sharpe_ratio) keyMetric = 'sharpe: ' + metrics.sharpe_ratio.toFixed(2);
    else if (metrics.ic) keyMetric = 'IC: ' + metrics.ic.toFixed(3);
    else if (metrics.f1_macro) keyMetric = 'F1: ' + metrics.f1_macro.toFixed(3);

    html += '<tr>';
    html += '<td>' + name + '</td>';
    html += '<td><span class="badge badge-blue">' + version + '</span></td>';
    html += '<td>' + keyMetric + '</td>';
    html += '</tr>';
  }
  tbody.innerHTML = html;
}

async function updateTrades() {
  const d = await fetchJson('/api/trades?limit=20');
  if (!d || !d.trades) return;

  const tbody = document.getElementById('trades-table');
  if (d.trades.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="neutral">No trades</td></tr>';
    return;
  }
  let html = '';
  for (const t of d.trades) {
    const time = new Date(t.time).toLocaleTimeString();
    const dirClass = t.direction === 'BUY' ? 'positive' : 'negative';
    html += '<tr>';
    html += '<td>' + time + '</td>';
    html += '<td>' + t.venue + '</td>';
    html += '<td class="' + dirClass + '">' + t.direction + '</td>';
    html += '<td>' + t.price.toFixed(4) + '</td>';
    html += '<td>' + fmtUsd(t.cost) + '</td>';
    html += '</tr>';
  }
  tbody.innerHTML = html;
}

async function updateRisk() {
  const d = await fetchJson('/api/risk');
  if (!d || d.status === 'error') return;

  const current = d.current || {};
  const history = d.history || [];

  if (history.length > 0) {
    const maxDd = Math.max(...history.map(h => h.drawdown || 0));
    document.getElementById('max-dd-7d').textContent = fmtPct(maxDd);
    document.getElementById('max-dd-7d').className = 'stat-value ' + (maxDd > 0.15 ? 'negative' : 'neutral');

    // Regime history
    const regimes = [...new Set(history.map(h => h.regime))];
    document.getElementById('regime-history').textContent = regimes.join(' > ');

    // Breaker changes
    let changes = 0;
    for (let i = 1; i < history.length; i++) {
      if (history[i].breaker !== history[i-1].breaker) changes++;
    }
    document.getElementById('breaker-changes').textContent = changes;
  }
}

// WebSocket for real-time updates
function connectWs() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(proto + '//' + location.host + '/ws');
  ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    document.getElementById('last-update').textContent = 'Last update: ' + new Date().toLocaleTimeString();
    if (data.portfolio) {
      document.getElementById('total-equity').textContent = fmtUsd(data.portfolio.total_equity);
    }
  };
  ws.onclose = function() {
    setTimeout(connectWs, 5000);
  };
  ws.onerror = function() {
    ws.close();
  };
}

// Initial load and periodic refresh
async function refresh() {
  await Promise.all([
    updatePortfolio(),
    updatePerformance(),
    updateEquityCurve(),
    updateDataHealth(),
    updateModels(),
    updateTrades(),
    updateRisk(),
  ]);
  document.getElementById('last-update').textContent = 'Updated: ' + new Date().toLocaleTimeString();
}

refresh();
setInterval(refresh, 15000);
connectWs();
window.addEventListener('resize', () => equityChart.draw());
</script>
</body>
</html>"""
