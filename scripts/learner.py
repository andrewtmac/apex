#!/usr/bin/env python3
"""Self-Improvement / Learning Agent for APEX V2.

Tracks strategy performance, adjusts weights, detects regimes,
and continuously improves the trading system.
"""

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

logger = structlog.get_logger()

WEIGHTS_FILE = Path(__file__).parent.parent / "data" / "strategy_weights.json"
LEARNINGS_FILE = Path(__file__).parent.parent / "data" / "learnings.json"

DEFAULT_WEIGHTS = {
    "weather": {
        "alpha": 5.0, "beta": 2.0, "weight": 0.50,
        "min_edge": 0.04, "max_size_pct": 0.12,
        "stop_loss": -0.25, "take_profit": 0.20,
        "total_trades": 0, "wins": 0, "losses": 0,
        "total_pnl": 0.0, "sharpe": 0.0,
    },
    "crypto": {
        "alpha": 3.0, "beta": 2.0, "weight": 0.30,
        "min_edge": 0.05, "max_size_pct": 0.08,
        "stop_loss": -0.20, "take_profit": 0.15,
        "total_trades": 0, "wins": 0, "losses": 0,
        "total_pnl": 0.0, "sharpe": 0.0,
    },
    "macro": {
        "alpha": 2.0, "beta": 2.0, "weight": 0.15,
        "min_edge": 0.06, "max_size_pct": 0.05,
        "stop_loss": -0.30, "take_profit": 0.25,
        "total_trades": 0, "wins": 0, "losses": 0,
        "total_pnl": 0.0, "sharpe": 0.0,
    },
    "sports": {
        "alpha": 2.0, "beta": 3.0, "weight": 0.05,
        "min_edge": 0.05, "max_size_pct": 0.03,
        "stop_loss": -0.30, "take_profit": 0.20,
        "total_trades": 0, "wins": 0, "losses": 0,
        "total_pnl": 0.0, "sharpe": 0.0,
    },
}


class LearnerAgent:
    """Tracks performance, adjusts strategy weights, detects market regimes."""

    def __init__(self):
        self.weights: dict = {}
        self.learnings: list[dict] = []
        self._load_weights()
        self._load_learnings()

    def _load_weights(self):
        try:
            if WEIGHTS_FILE.exists():
                with open(WEIGHTS_FILE) as f:
                    self.weights = json.load(f)
            else:
                self.weights = dict(DEFAULT_WEIGHTS)
                self._save_weights()
        except Exception:
            self.weights = dict(DEFAULT_WEIGHTS)

    def _save_weights(self):
        try:
            WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(WEIGHTS_FILE, "w") as f:
                json.dump(self.weights, f, indent=2)
        except Exception as e:
            logger.warning("learner.weights_save_failed", error=str(e))

    def _load_learnings(self):
        try:
            if LEARNINGS_FILE.exists():
                with open(LEARNINGS_FILE) as f:
                    self.learnings = json.load(f)
        except Exception:
            self.learnings = []

    def _save_learnings(self):
        try:
            LEARNINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(LEARNINGS_FILE, "w") as f:
                json.dump(self.learnings[-500:], f, indent=2)
        except Exception as e:
            logger.warning("learner.learnings_save_failed", error=str(e))

    def record_trade(self, trade: dict):
        strategy = trade.get("strategy", "unknown")
        pnl = trade.get("realized_pnl", 0)

        if strategy not in self.weights:
            self.weights[strategy] = dict(DEFAULT_WEIGHTS.get("sports", {}))

        w = self.weights[strategy]
        w["total_trades"] += 1
        w["total_pnl"] += pnl

        if pnl >= 0:
            w["wins"] += 1
            w["alpha"] += 1
        else:
            w["losses"] += 1
            w["beta"] += 1

        if w["total_trades"] >= 10:
            win_rate = w["wins"] / w["total_trades"]
            avg_win = w["total_pnl"] / max(w["wins"], 1) if w["wins"] > 0 else 0
            avg_loss = w["total_pnl"] / max(w["losses"], 1) if w["losses"] > 0 else 0
            if avg_loss != 0:
                w["sharpe"] = round(
                    (win_rate * avg_win) / max(abs(avg_loss), 0.01) * np.sqrt(252), 2
                )

        self._save_weights()

    def adjust_weights(self):
        samples = {}
        for name, w in self.weights.items():
            samples[name] = np.random.beta(w["alpha"], w["beta"])

        total = sum(samples.values())
        if total <= 0:
            return

        for name in self.weights:
            self.weights[name]["weight"] = round(samples[name] / total, 3)

        self._save_weights()
        logger.info(
            "learner.weights_updated",
            weights={k: v["weight"] for k, v in self.weights.items()},
        )

    def get_strategy_config(self, strategy: str) -> dict:
        return self.weights.get(strategy, DEFAULT_WEIGHTS.get("sports", {}))

    def get_min_edge(self, strategy: str) -> float:
        return self.weights.get(strategy, {}).get("min_edge", 0.05)

    def get_max_size_pct(self, strategy: str) -> float:
        return self.weights.get(strategy, {}).get("max_size_pct", 0.05)

    def detect_regime(self, equity_history: list[dict], recent_trades: list[dict]) -> str:
        if len(equity_history) < 20:
            return "NORMAL"

        recent_eq = [e["equity"] for e in equity_history[-48:]]
        if len(recent_eq) < 10:
            return "NORMAL"

        returns = np.diff(recent_eq) / np.array(recent_eq[:-1])
        vol = np.std(returns) * 100
        trend = (recent_eq[-1] - recent_eq[0]) / recent_eq[0] * 100
        peak = max(recent_eq)
        current = recent_eq[-1]
        dd = (peak - current) / peak * 100

        if dd > 10:
            return "DRAWING_DOWN"
        elif vol > 2:
            return "HIGH_VOL"
        elif vol < 0.3:
            return "LOW_VOL"
        elif abs(trend) > 5:
            return "TRENDING"
        return "NORMAL"

    def generate_insight(self, trade: dict) -> Optional[dict]:
        strategy = trade.get("strategy", "unknown")
        pnl = trade.get("realized_pnl", 0)
        edge = trade.get("edge_at_entry", 0)
        question = trade.get("question", "")

        insight = None
        w = self.weights.get(strategy, {})

        if edge > 0 and pnl < 0:
            insight = {
                "type": "edge_miss", "strategy": strategy,
                "message": f"Positive edge ({edge:+.1%}) but lost ${abs(pnl):.0f} on {question[:40]}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        if pnl > 0 and pnl > w.get("total_pnl", 0) / max(w.get("wins", 1), 1) * 2:
            insight = {
                "type": "big_win", "strategy": strategy,
                "message": f"Big win: ${pnl:+.0f} on {question[:40]} (edge {edge:+.1%})",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        if insight:
            self.learnings.append(insight)
            self._save_learnings()
        return insight

    def build_daily_summary(self) -> str:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        recent = [l for l in self.learnings if l.get("timestamp", "") > cutoff]

        if not recent:
            return "No new learnings in the last 24h."

        lines = ["Learnings (24h)"]
        for l in recent[-10:]:
            lines.append(f"  {l['type']}: {l['message'][:80]}")

        lines.append("")
        lines.append("Strategy Performance")
        for name, w in sorted(self.weights.items(), key=lambda x: x[1].get("weight", 0), reverse=True):
            wr = w["wins"] / w["total_trades"] * 100 if w["total_trades"] > 0 else 0
            lines.append(
                f"  {name}: {w['weight']:.0%} weight, "
                f"{w['wins']}W-{w['losses']}L ({wr:.0f}%), "
                f"P&L ${w['total_pnl']:+.0f}"
            )
        return "\n".join(lines)

    async def daily_update(self):
        self.adjust_weights()
        summary = self.build_daily_summary()
        logger.info("learner.daily_update", summary=summary[:200])
        return summary
