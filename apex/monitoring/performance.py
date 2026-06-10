"""Strategy and model performance tracking.

Computes and stores:
- Per-strategy: trades, wins, losses, PnL, Sharpe, max drawdown, Brier score
- Per-model: accuracy, calibration, feature importance drift
- Portfolio-level: equity curve, risk metrics, regime history
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class PerformanceTracker:
    """Tracks and reports strategy and model performance.

    Reads from TimescaleDB trades, signals, and portfolio_snapshots
    tables to compute comprehensive performance metrics.

    Parameters
    ----------
    db_url : TimescaleDB connection string
    """

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            self.db_url, min_size=2, max_size=10, command_timeout=30
        )
        logger.info("performance_tracker.connected")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def _ensure_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            await self.connect()
        assert self._pool is not None
        return self._pool

    async def record_trade(self, trade: dict[str, Any]) -> None:
        """Record a completed trade in the database.

        Parameters
        ----------
        trade : dict with keys:
            id, market_id, venue, strategy, direction, side,
            price, quantity, cost, fee, signal_id, metadata
        """
        pool = await self._ensure_pool()

        await pool.execute(
            """
            INSERT INTO trades (id, time, market_id, venue, strategy,
                              direction, side, price, quantity, cost,
                              fee, signal_id, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (id) DO NOTHING
            """,
            trade["id"],
            trade.get("time", datetime.now(timezone.utc)),
            trade["market_id"],
            trade["venue"],
            trade["strategy"],
            trade["direction"],
            trade.get("side", "ENTRY"),
            float(trade["price"]),
            float(trade["quantity"]),
            float(trade["cost"]),
            float(trade.get("fee", 0)),
            trade.get("signal_id"),
            json.dumps(trade.get("metadata", {})),
        )

        logger.debug(
            "performance.trade_recorded",
            trade_id=trade["id"],
            venue=trade["venue"],
        )

    async def compute_strategy_stats(
        self,
        strategy: str,
        days: int = 30,
    ) -> dict[str, Any]:
        """Compute performance statistics for a specific strategy.

        Parameters
        ----------
        strategy : strategy name
        days : lookback period

        Returns
        -------
        dict with: trades, wins, losses, gross_pnl, net_pnl, sharpe,
                   max_drawdown, win_rate, avg_edge, brier_score, avg_hold_hours
        """
        pool = await self._ensure_pool()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        # Fetch trades for this strategy
        trades = await pool.fetch(
            """
            SELECT t.time, t.direction, t.side, t.price, t.quantity,
                   t.cost, t.fee, t.market_id, t.venue,
                   s.edge, s.ensemble_score
            FROM trades t
            LEFT JOIN signals s ON s.id = t.signal_id
            WHERE t.strategy = $1
              AND t.time >= $2
            ORDER BY t.time ASC
            """,
            strategy,
            cutoff,
        )

        if not trades:
            return {
                "strategy": strategy,
                "period_days": days,
                "trades": 0,
                "status": "no_trades",
            }

        # Group trades by market to compute P&L per position
        positions: dict[str, list[dict]] = {}
        for t in trades:
            mid = t["market_id"]
            if mid not in positions:
                positions[mid] = []
            positions[mid].append(dict(t))

        # Compute P&L per closed position
        pnl_list: list[float] = []
        edges: list[float] = []
        hold_times: list[float] = []

        for market_id, market_trades in positions.items():
            entries = [t for t in market_trades if t["side"] == "ENTRY"]
            exits = [t for t in market_trades if t["side"] == "EXIT"]

            if entries and exits:
                entry_cost = sum(t["cost"] for t in entries)
                exit_value = sum(t["cost"] for t in exits)
                entry_fees = sum(t["fee"] or 0 for t in entries)
                exit_fees = sum(t["fee"] or 0 for t in exits)

                gross = exit_value - entry_cost
                net = gross - entry_fees - exit_fees
                pnl_list.append(net)

                # Hold time
                entry_time = entries[0]["time"]
                exit_time = exits[-1]["time"]
                hold_hours = (exit_time - entry_time).total_seconds() / 3600
                hold_times.append(hold_hours)

            for t in market_trades:
                if t.get("edge") is not None:
                    edges.append(float(t["edge"]))

        # Aggregate metrics
        pnl_arr = np.array(pnl_list) if pnl_list else np.array([0.0])
        wins = int(np.sum(pnl_arr > 0))
        losses = int(np.sum(pnl_arr <= 0))

        # Sharpe ratio (daily)
        daily_returns: list[float] = []
        if pnl_list:
            # Group P&L by day
            from collections import defaultdict

            daily_pnl: dict[str, float] = defaultdict(float)
            for t in trades:
                day_key = t["time"].strftime("%Y-%m-%d")
                daily_pnl[day_key] += float(t["cost"]) * (
                    1 if t["side"] == "EXIT" else -1
                )
            daily_returns = list(daily_pnl.values())

        dr_arr = np.array(daily_returns) if daily_returns else np.array([0.0])
        mean_ret = float(np.mean(dr_arr))
        std_ret = float(np.std(dr_arr)) + 1e-8
        sharpe = mean_ret / std_ret * np.sqrt(252)

        # Max drawdown on cumulative P&L
        cum_pnl = np.cumsum(pnl_arr)
        peak = np.maximum.accumulate(cum_pnl)
        drawdown = peak - cum_pnl
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        # Brier score (if we have edge data and outcomes)
        brier = await self._compute_strategy_brier(pool, strategy, cutoff)

        result = {
            "strategy": strategy,
            "period_days": days,
            "trades": len(trades),
            "closed_positions": len(pnl_list),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / max(wins + losses, 1),
            "gross_pnl": float(np.sum(pnl_arr)),
            "net_pnl": float(np.sum(pnl_arr)),
            "avg_pnl": float(np.mean(pnl_arr)),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": max_dd,
            "avg_edge": float(np.mean(edges)) if edges else 0.0,
            "avg_hold_hours": float(np.mean(hold_times)) if hold_times else 0.0,
            "brier_score": brier,
        }

        logger.info("performance.strategy_stats", **result)
        return result

    async def _compute_strategy_brier(
        self,
        pool: asyncpg.Pool,
        strategy: str,
        cutoff: datetime,
    ) -> float | None:
        """Compute Brier score for a strategy's probability predictions."""
        rows = await pool.fetch(
            """
            SELECT s.edge + p.mid AS predicted_prob, m.outcome
            FROM signals s
            JOIN price_ticks p ON p.symbol = s.market_id
                AND p.time = (SELECT MAX(time) FROM price_ticks
                              WHERE symbol = s.market_id AND time <= s.time)
            JOIN markets m ON m.id = s.market_id
            WHERE s.strategy = $1
              AND s.time >= $2
              AND m.outcome IS NOT NULL
              AND s.accepted = TRUE
            """,
            strategy,
            cutoff,
        )

        if len(rows) < 5:
            return None

        probs = np.array([float(r["predicted_prob"]) for r in rows])
        outcomes = np.array([int(r["outcome"]) for r in rows])

        probs = np.clip(probs, 0, 1)
        brier = float(np.mean((probs - outcomes) ** 2))
        return brier

    async def compute_model_metrics(
        self,
        model_name: str,
        days: int = 30,
    ) -> dict[str, Any]:
        """Compute performance metrics for a specific model.

        Parameters
        ----------
        model_name : name of the model
        days : lookback period

        Returns
        -------
        dict with accuracy, calibration, recent_performance, drift indicators
        """
        pool = await self._ensure_pool()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        rows = await pool.fetch(
            """
            SELECT time, metric_name, metric_value
            FROM model_performance
            WHERE model_name = $1
              AND time >= $2
            ORDER BY time ASC
            """,
            model_name,
            cutoff,
        )

        if not rows:
            return {
                "model_name": model_name,
                "period_days": days,
                "status": "no_data",
            }

        # Group metrics by name
        metric_series: dict[str, list[tuple[datetime, float]]] = {}
        for r in rows:
            name = r["metric_name"]
            if name not in metric_series:
                metric_series[name] = []
            metric_series[name].append((r["time"], float(r["metric_value"])))

        result: dict[str, Any] = {
            "model_name": model_name,
            "period_days": days,
        }

        for metric_name, series in metric_series.items():
            values = [v for _, v in series]
            result[f"{metric_name}_latest"] = values[-1]
            result[f"{metric_name}_mean"] = float(np.mean(values))
            result[f"{metric_name}_std"] = float(np.std(values))

            # Trend: positive slope = improving
            if len(values) >= 3:
                x = np.arange(len(values), dtype=float)
                from scipy.stats import linregress

                slope, _, _, _, _ = linregress(x, values)
                result[f"{metric_name}_trend"] = float(slope)

        logger.info(
            "performance.model_metrics",
            model=model_name,
            n_metrics=len(metric_series),
        )
        return result

    async def record_portfolio_snapshot(
        self,
        snapshot: dict[str, Any],
    ) -> None:
        """Record a portfolio state snapshot."""
        pool = await self._ensure_pool()

        await pool.execute(
            """
            INSERT INTO portfolio_snapshots
                (time, total_equity, poly_equity, kalshi_equity, tt_equity,
                 open_positions, deployed_pct, drawdown_pct, cvar_95,
                 regime, breaker_level)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            snapshot.get("time", datetime.now(timezone.utc)),
            float(snapshot.get("total_equity", 0)),
            float(snapshot.get("poly_equity", 0)),
            float(snapshot.get("kalshi_equity", 0)),
            float(snapshot.get("tt_equity", 0)),
            int(snapshot.get("open_positions", 0)),
            float(snapshot.get("deployed_pct", 0)),
            float(snapshot.get("drawdown_pct", 0)),
            snapshot.get("cvar_95"),
            snapshot.get("regime", "NORMAL"),
            snapshot.get("breaker_level", "GREEN"),
        )

    async def daily_report(self) -> dict[str, Any]:
        """Generate a comprehensive daily performance report.

        Returns
        -------
        dict with portfolio summary, per-strategy stats, model health,
        and risk metrics.
        """
        pool = await self._ensure_pool()
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # 1. Portfolio snapshot
        latest_snapshot = await pool.fetchrow(
            """
            SELECT * FROM portfolio_snapshots
            ORDER BY time DESC LIMIT 1
            """
        )

        yesterday_snapshot = await pool.fetchrow(
            """
            SELECT * FROM portfolio_snapshots
            WHERE time < $1
            ORDER BY time DESC LIMIT 1
            """,
            today_start,
        )

        portfolio: dict[str, Any] = {}
        if latest_snapshot:
            portfolio = dict(latest_snapshot)
            if yesterday_snapshot:
                portfolio["daily_pnl"] = (
                    float(latest_snapshot["total_equity"])
                    - float(yesterday_snapshot["total_equity"])
                )
                prev_eq = float(yesterday_snapshot["total_equity"])
                portfolio["daily_pnl_pct"] = (
                    portfolio["daily_pnl"] / prev_eq * 100 if prev_eq > 0 else 0
                )

        # 2. Today's trades
        trades_today = await pool.fetchval(
            """
            SELECT COUNT(*) FROM trades WHERE time >= $1
            """,
            today_start,
        )

        # 3. Per-strategy breakdown
        strategies = await pool.fetch(
            """
            SELECT DISTINCT strategy FROM trades
            WHERE time >= $1
            """,
            now - timedelta(days=30),
        )

        strategy_stats: dict[str, Any] = {}
        for row in strategies:
            stats = await self.compute_strategy_stats(row["strategy"], days=30)
            strategy_stats[row["strategy"]] = stats

        # 4. Equity curve (last 30 days)
        equity_rows = await pool.fetch(
            """
            SELECT time_bucket('1 day', time) AS day,
                   last(total_equity, time) AS equity
            FROM portfolio_snapshots
            WHERE time >= $1
            GROUP BY day
            ORDER BY day
            """,
            now - timedelta(days=30),
        )

        equity_curve = [
            {"date": r["day"].isoformat(), "equity": float(r["equity"])}
            for r in equity_rows
        ]

        report = {
            "generated_at": now.isoformat(),
            "portfolio": portfolio,
            "trades_today": trades_today or 0,
            "strategies": strategy_stats,
            "equity_curve": equity_curve,
        }

        logger.info(
            "performance.daily_report",
            equity=portfolio.get("total_equity"),
            trades_today=trades_today,
            n_strategies=len(strategy_stats),
        )
        return report
