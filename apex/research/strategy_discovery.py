"""Scan for new market categories where APEX has no models.

Process:
1. Scan Polymarket/Kalshi for market categories
2. Check if we have enough resolved markets (>100) to train
3. If yes, train XGBoost and backtest
4. If Sharpe > 1.0 OOS, deploy to paper for 2 weeks
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"
KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Minimum requirements for deploying a new category
MIN_RESOLVED_MARKETS = 100
MIN_OOS_SHARPE = 1.0
PAPER_TRIAL_DAYS = 14


class StrategyDiscovery:
    """Discovers and evaluates new market categories for APEX.

    Continuously scans prediction market venues for categories that
    have sufficient resolved data for training.  When a viable category
    is found, trains a quick XGBoost model and runs a walk-forward
    backtest.  Categories passing the Sharpe threshold are flagged
    for paper trading deployment.

    Parameters
    ----------
    db_url : TimescaleDB connection string
    min_markets : minimum resolved markets to consider a category
    min_sharpe : minimum out-of-sample Sharpe ratio
    timeout : HTTP request timeout
    """

    def __init__(
        self,
        db_url: str = "",
        min_markets: int = MIN_RESOLVED_MARKETS,
        min_sharpe: float = MIN_OOS_SHARPE,
        timeout: float = 30.0,
    ) -> None:
        self.db_url = db_url
        self.min_markets = min_markets
        self.min_sharpe = min_sharpe
        self._timeout = timeout

    async def scan_categories(self) -> list[dict[str, Any]]:
        """Scan venues for market categories and their readiness.

        Returns a list of category dicts with:
        - category: name
        - venue: source platform
        - n_total: total markets in category
        - n_resolved: resolved markets
        - n_active: currently active markets
        - ready: bool, whether we have enough data to train
        """
        poly_cats = await self._scan_polymarket_categories()
        kalshi_cats = await self._scan_kalshi_categories()

        all_categories = poly_cats + kalshi_cats

        # Check which categories we already trade
        existing = await self._get_existing_categories()

        for cat in all_categories:
            cat["already_trading"] = cat["category"] in existing
            cat["ready"] = (
                cat["n_resolved"] >= self.min_markets
                and not cat["already_trading"]
            )

        # Sort by readiness and resolved count
        all_categories.sort(
            key=lambda c: (c["ready"], c["n_resolved"]), reverse=True
        )

        logger.info(
            "strategy_discovery.scan_complete",
            n_categories=len(all_categories),
            n_ready=sum(1 for c in all_categories if c["ready"]),
        )

        return all_categories

    async def _scan_polymarket_categories(self) -> list[dict[str, Any]]:
        """Scan Polymarket for category distribution."""
        categories: dict[str, dict[str, int]] = {}

        async with httpx.AsyncClient(
            base_url=POLYMARKET_GAMMA_URL,
            timeout=httpx.Timeout(self._timeout),
        ) as client:
            for closed in [True, False]:
                try:
                    resp = await client.get(
                        "/markets",
                        params={
                            "limit": 500,
                            "closed": closed,
                        },
                    )
                    resp.raise_for_status()
                    markets = resp.json()

                    for m in markets if isinstance(markets, list) else markets.get("data", []):
                        cat = m.get("category", "other") or "other"
                        if cat not in categories:
                            categories[cat] = {
                                "n_total": 0,
                                "n_resolved": 0,
                                "n_active": 0,
                            }
                        categories[cat]["n_total"] += 1
                        if m.get("outcome") is not None or closed:
                            categories[cat]["n_resolved"] += 1
                        if not closed:
                            categories[cat]["n_active"] += 1

                except Exception:
                    logger.warning("strategy_discovery.polymarket_scan_error")
                    continue

        return [
            {
                "category": cat,
                "venue": "polymarket",
                **counts,
            }
            for cat, counts in categories.items()
        ]

    async def _scan_kalshi_categories(self) -> list[dict[str, Any]]:
        """Scan Kalshi for category distribution."""
        categories: dict[str, dict[str, int]] = {}

        async with httpx.AsyncClient(
            base_url=KALSHI_API_URL,
            timeout=httpx.Timeout(self._timeout),
        ) as client:
            for status in ["settled", "open"]:
                try:
                    resp = await client.get(
                        "/events",
                        params={"limit": 200, "status": status},
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    for event in data.get("events", []):
                        cat = event.get("category", "other") or "other"
                        if cat not in categories:
                            categories[cat] = {
                                "n_total": 0,
                                "n_resolved": 0,
                                "n_active": 0,
                            }
                        n_markets = len(event.get("markets", []))
                        categories[cat]["n_total"] += n_markets
                        if status == "settled":
                            categories[cat]["n_resolved"] += n_markets
                        else:
                            categories[cat]["n_active"] += n_markets

                except Exception:
                    logger.warning("strategy_discovery.kalshi_scan_error")
                    continue

        return [
            {
                "category": cat,
                "venue": "kalshi",
                **counts,
            }
            for cat, counts in categories.items()
        ]

    async def _get_existing_categories(self) -> set[str]:
        """Get categories we already have strategies for."""
        if not self.db_url:
            return set()

        import asyncpg

        try:
            pool = await asyncpg.create_pool(
                self.db_url, min_size=1, max_size=2, command_timeout=10
            )
            try:
                rows = await pool.fetch(
                    """
                    SELECT DISTINCT category
                    FROM markets
                    WHERE status = 'active'
                    """
                )
                return {r["category"] for r in rows if r["category"]}
            finally:
                await pool.close()
        except Exception:
            return set()

    async def evaluate_category(
        self,
        category: str,
        data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Train and evaluate a model on a specific category.

        Parameters
        ----------
        category : category name to evaluate
        data : DataFrame with features and outcome column for this category

        Returns
        -------
        dict with:
        - category: str
        - n_samples: int
        - oos_sharpe: float
        - oos_accuracy: float
        - oos_brier: float
        - recommendation: "deploy_paper" | "needs_more_data" | "insufficient_edge"
        """
        logger.info(
            "strategy_discovery.evaluate",
            category=category,
            n_samples=len(data),
        )

        if len(data) < self.min_markets:
            return {
                "category": category,
                "n_samples": len(data),
                "recommendation": "needs_more_data",
                "reason": f"Only {len(data)} markets (need {self.min_markets})",
            }

        # Identify feature and target columns
        target_col = "outcome"
        feature_cols = [
            c for c in data.columns
            if c not in (target_col, "market_id", "venue", "question",
                        "category", "resolution_date", "open_date")
            and data[c].dtype in (np.float64, np.float32, float, int)
        ]

        if not feature_cols:
            return {
                "category": category,
                "n_samples": len(data),
                "recommendation": "needs_more_data",
                "reason": "No numeric feature columns found",
            }

        # Walk-forward backtest
        from apex.models.xgboost_prob import XGBoostProbabilityModel
        from apex.research.walk_forward import WalkForwardValidator

        validator = WalkForwardValidator(
            train_days=30,
            test_days=7,
            n_splits=5,
            min_train_samples=30,
        )

        def model_factory():
            return XGBoostProbabilityModel(
                params={"n_estimators": 200, "max_depth": 4}
            )

        try:
            results = validator.validate(
                model_factory=model_factory,
                data=data,
                feature_columns=feature_cols,
                target_column=target_col,
                date_column="resolution_date" if "resolution_date" in data.columns else "date",
            )
        except Exception as exc:
            return {
                "category": category,
                "n_samples": len(data),
                "recommendation": "insufficient_edge",
                "reason": f"Walk-forward failed: {exc}",
            }

        # Extract key metrics
        oos_sharpe = results.get("ic_mean", 0.0) * np.sqrt(252)  # annualized IC as proxy
        oos_accuracy = results.get("accuracy_mean", 0.5)
        oos_brier = results.get("brier_score_mean", 0.25)

        # Determine recommendation
        if oos_sharpe >= self.min_sharpe and oos_accuracy > 0.55:
            recommendation = "deploy_paper"
        elif oos_sharpe >= self.min_sharpe * 0.7:
            recommendation = "monitor"
        else:
            recommendation = "insufficient_edge"

        result = {
            "category": category,
            "n_samples": len(data),
            "oos_sharpe": round(oos_sharpe, 3),
            "oos_accuracy": round(oos_accuracy, 4),
            "oos_brier": round(oos_brier, 4),
            "n_splits": results.get("n_splits", 0),
            "recommendation": recommendation,
            "walk_forward_details": results,
        }

        logger.info(
            "strategy_discovery.evaluated",
            category=category,
            recommendation=recommendation,
            oos_sharpe=result["oos_sharpe"],
        )

        return result

    async def discover_and_evaluate(self) -> list[dict[str, Any]]:
        """Full discovery pipeline: scan + evaluate ready categories.

        Returns evaluated results for all ready categories.
        """
        categories = await self.scan_categories()
        ready = [c for c in categories if c.get("ready", False)]

        if not ready:
            logger.info("strategy_discovery.no_ready_categories")
            return []

        # Collect data and evaluate each ready category
        from apex.research.historical_data import HistoricalDataCollector

        collector = HistoricalDataCollector()
        full_data = await collector.build_training_dataset()

        results: list[dict[str, Any]] = []
        for cat_info in ready:
            cat_name = cat_info["category"]
            cat_data = full_data[full_data["category"] == cat_name]

            if len(cat_data) < 30:
                results.append({
                    "category": cat_name,
                    "venue": cat_info["venue"],
                    "recommendation": "needs_more_data",
                    "n_samples": len(cat_data),
                })
                continue

            evaluation = await self.evaluate_category(cat_name, cat_data)
            evaluation["venue"] = cat_info["venue"]
            results.append(evaluation)

        # Sort by recommendation priority
        priority = {"deploy_paper": 0, "monitor": 1, "insufficient_edge": 2, "needs_more_data": 3}
        results.sort(key=lambda r: priority.get(r.get("recommendation", ""), 99))

        logger.info(
            "strategy_discovery.pipeline_complete",
            n_evaluated=len(results),
            n_deployable=sum(
                1 for r in results if r.get("recommendation") == "deploy_paper"
            ),
        )

        return results
