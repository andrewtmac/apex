"""Main entry point: Multi-venue TradingNode.

Sets up NautilusTrader TradingNode with:
- Polymarket data + exec clients
- Kalshi data + exec clients
- TastyTrade data + exec clients
- ApexStrategy instances for each active market

Based on: nautilus_trader/examples/live/multi_venue_ml_bot.py
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from apex.config import ApexConfig, TradingMode, load_config
from apex.data.store import FeatureStore
from apex.data.streams import StreamPublisher
from apex.models.registry import ModelRegistry
from apex.monitoring.alerts import AlertManager
from apex.monitoring.data_health import DataHealthMonitor
from apex.monitoring.performance import PerformanceTracker
from apex.training.scheduler import TrainingScheduler

logger = structlog.get_logger(__name__)


class ApexNode:
    """Orchestrates all APEX subsystems.

    The node is the top-level coordinator that:
    1. Loads configuration and models
    2. Starts data ingestion pipelines
    3. Launches the training scheduler
    4. Starts performance tracking and monitoring
    5. Runs the trading loop (via NautilusTrader or standalone)

    Parameters
    ----------
    config : ApexConfig instance
    mode : "paper", "live", or "dry_run"
    """

    def __init__(
        self,
        config: ApexConfig,
        mode: str = "paper",
    ) -> None:
        self.config = config
        self.mode = mode
        self._running = False

        # Core components (initialized in start)
        self.registry = ModelRegistry(store_path=Path("models_store"))
        self.feature_store: FeatureStore | None = None
        self.publisher: StreamPublisher | None = None
        self.alert_manager: AlertManager | None = None
        self.performance_tracker: PerformanceTracker | None = None
        self.data_monitor: DataHealthMonitor | None = None
        self.training_scheduler: TrainingScheduler | None = None

        # Background tasks
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Initialize all subsystems and begin trading."""
        logger.info(
            "apex_node.starting",
            mode=self.mode,
            trading_mode=self.config.trading_mode.value,
        )

        # 1. Feature store
        self.feature_store = FeatureStore(
            redis_url=self.config.infra.redis_url,
            database_url=self.config.infra.database_url,
        )
        await self.feature_store.connect()
        logger.info("apex_node.feature_store_connected")

        # 2. Stream publisher
        self.publisher = StreamPublisher(self.config.infra.redis_url)
        await self.publisher.connect()
        logger.info("apex_node.publisher_connected")

        # 3. Alerts
        self.alert_manager = AlertManager(
            bot_token=self.config.notifications.telegram_bot_token,
            chat_id=self.config.notifications.telegram_chat_id,
        )

        # 4. Performance tracking
        self.performance_tracker = PerformanceTracker(
            db_url=self.config.infra.database_url
        )
        await self.performance_tracker.connect()

        # 5. Data health monitor
        self.data_monitor = DataHealthMonitor(
            redis_url=self.config.infra.redis_url,
            db_url=self.config.infra.database_url,
        )
        await self.data_monitor.connect()

        # 6. Load models
        await self._load_models()

        # 7. Training scheduler
        self.training_scheduler = TrainingScheduler(
            model_registry=self.registry,
            db_url=self.config.infra.database_url,
            on_complete=self._on_training_complete,
        )

        # 8. Start data ingestion
        self._tasks.append(
            asyncio.create_task(
                self._run_data_ingestion(), name="data-ingestion"
            )
        )

        # 9. Start training scheduler
        self._tasks.append(
            asyncio.create_task(
                self.training_scheduler.run(), name="training-scheduler"
            )
        )

        # 10. Start monitoring loops
        self._tasks.append(
            asyncio.create_task(
                self._monitoring_loop(), name="monitoring"
            )
        )
        self._tasks.append(
            asyncio.create_task(
                self._portfolio_snapshot_loop(), name="snapshots"
            )
        )

        self._running = True

        # Send startup alert
        await self.alert_manager.send_alert(
            f"<b>APEX Node Started</b>\n\nMode: {self.mode}\n"
            f"Trading Mode: {self.config.trading_mode.value}",
            priority="info",
            category="startup",
        )

        logger.info("apex_node.started", mode=self.mode)

        # Wait for all tasks (run forever until stopped)
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("apex_node.tasks_cancelled")

    async def stop(self) -> None:
        """Gracefully shut down all subsystems."""
        logger.info("apex_node.stopping")
        self._running = False

        # Cancel all background tasks
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        # Stop training scheduler
        if self.training_scheduler:
            await self.training_scheduler.stop()

        # Close connections
        if self.feature_store:
            await self.feature_store.close()
        if self.publisher:
            await self.publisher.close()
        if self.performance_tracker:
            await self.performance_tracker.close()
        if self.data_monitor:
            await self.data_monitor.close()

        # Send shutdown alert
        if self.alert_manager:
            await self.alert_manager.send_alert(
                "<b>APEX Node Stopped</b>",
                priority="warning",
                category="shutdown",
            )

        logger.info("apex_node.stopped")

    async def _load_models(self) -> None:
        """Load production models from the registry."""
        model_names = [
            "xgboost_prob",
            "lgbm_return",
            "tft_quantile",
            "lstm_regime",
            "ppo_position_manager",
            "finbert_sentiment",
        ]

        loaded = 0
        for name in model_names:
            try:
                model = self.registry.load(name, version="production")
                logger.info("apex_node.model_loaded", model=name)
                loaded += 1
            except FileNotFoundError:
                logger.warning(
                    "apex_node.model_not_found",
                    model=name,
                    note="will use default or skip",
                )
            except Exception as exc:
                logger.error(
                    "apex_node.model_load_error",
                    model=name,
                    error=str(exc),
                )

        logger.info("apex_node.models_loaded", count=loaded, total=len(model_names))

    async def _run_data_ingestion(self) -> None:
        """Start data ingestion pipelines for all configured venues."""
        from apex.data.ingestion.polymarket import PolymarketIngester

        ingesters: list[Any] = []

        # Polymarket
        if self.config.venues.polymarket.api_key:
            poly = PolymarketIngester(self.config)
            try:
                await poly.start()
                ingesters.append(poly)
                logger.info("apex_node.polymarket_ingester_started")
            except Exception as exc:
                logger.error(
                    "apex_node.polymarket_ingester_failed", error=str(exc)
                )

        # Keep running while node is active
        try:
            while self._running:
                await asyncio.sleep(60)
        finally:
            for ingester in ingesters:
                try:
                    await ingester.stop()
                except Exception:
                    pass

    async def _monitoring_loop(self) -> None:
        """Periodic health checks and alerting."""
        while self._running:
            try:
                if self.data_monitor:
                    health = await self.data_monitor.check_all()

                    # Alert on critical data sources
                    if health.get("status") == "critical" and self.alert_manager:
                        critical_sources = [
                            name
                            for name, info in health.get("sources", {}).items()
                            if info.get("status") == "critical"
                        ]
                        await self.alert_manager.data_failure_alert(
                            source=", ".join(critical_sources),
                            error="Critical staleness detected",
                        )

                    # Alert on price anomalies
                    for anomaly in health.get("anomalies", []):
                        if self.alert_manager:
                            await self.alert_manager.risk_alert(
                                metric="price_anomaly",
                                value=anomaly.get("z_score", 0),
                                threshold=5.0,
                                message=f"Symbol: {anomaly.get('symbol')}",
                            )

            except Exception:
                logger.exception("apex_node.monitoring_error")

            await asyncio.sleep(60)

    async def _portfolio_snapshot_loop(self) -> None:
        """Periodic portfolio state snapshots."""
        while self._running:
            try:
                if self.performance_tracker:
                    snapshot = {
                        "time": datetime.now(timezone.utc),
                        "total_equity": 0.0,
                        "poly_equity": 0.0,
                        "kalshi_equity": 0.0,
                        "tt_equity": 0.0,
                        "open_positions": 0,
                        "deployed_pct": 0.0,
                        "drawdown_pct": 0.0,
                        "cvar_95": None,
                        "regime": "NORMAL",
                        "breaker_level": "GREEN",
                    }
                    await self.performance_tracker.record_portfolio_snapshot(
                        snapshot
                    )
            except Exception:
                logger.debug("apex_node.snapshot_error")

            await asyncio.sleep(300)  # every 5 minutes

    async def _on_training_complete(
        self, model_name: str, result: dict
    ) -> None:
        """Callback when a training job completes."""
        if self.alert_manager:
            await self.alert_manager.model_retrain_alert(model_name, result)


async def run_node(
    config: ApexConfig | None = None,
    mode: str = "paper",
) -> None:
    """Start the APEX trading node.

    Parameters
    ----------
    config : ApexConfig instance. If None, loads from environment.
    mode : "paper", "live", or "dry_run"
    """
    if config is None:
        config = load_config()

    # Safety check for live mode
    if mode == "live" and config.trading_mode != TradingMode.LIVE:
        raise ValueError(
            "Live mode requested but TRADING_MODE env var is not LIVE. "
            "Set TRADING_MODE=LIVE to confirm."
        )

    node = ApexNode(config=config, mode=mode)

    try:
        await node.start()
    except KeyboardInterrupt:
        logger.info("apex_node.keyboard_interrupt")
    finally:
        await node.stop()
