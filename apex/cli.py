"""APEX Command Line Interface.

Commands:
  apex train [model]    - Train a specific model or all models
  apex backtest         - Run walk-forward backtest
  apex paper            - Start paper trading
  apex live             - Start live trading
  apex status           - Show system status
  apex dashboard        - Start web dashboard
  apex collect          - Collect historical data
  apex discover         - Run alpha discovery pipeline
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
import structlog

logger = structlog.get_logger(__name__)


def _setup_logging(level: str = "info") -> None:
    """Configure structlog for CLI output."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog, level.upper(), structlog.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _run_async(coro):
    """Run an async coroutine from sync CLI context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(coro)
    except RuntimeError:
        pass
    return asyncio.run(coro)


@click.group()
@click.option("--log-level", default="info", help="Logging level")
def main(log_level: str):
    """APEX: Adaptive Prediction EXchange

    ML-driven multi-venue prediction market trading system.
    """
    _setup_logging(log_level)


@main.command()
@click.argument("model", required=False)
@click.option("--lookback", default=90, help="Training lookback days")
def train(model: str | None, lookback: int):
    """Train models.

    If MODEL is specified, train only that model.
    Otherwise, train all models sequentially.

    Available models:
      xgboost_prob, lgbm_return, tft_quantile,
      lstm_regime, ppo_position_manager, finbert_sentiment
    """
    from apex.config import load_config
    from apex.models.registry import ModelRegistry

    config = load_config()
    registry = ModelRegistry(store_path=Path("models_store"))

    async def _train():
        if model:
            return await _train_single(model, config, registry, lookback)
        else:
            return await _train_all(config, registry, lookback)

    result = _run_async(_train())

    if isinstance(result, dict):
        status = result.get("status", "unknown")
        click.echo(f"\nTraining complete: {status}")
        if "metrics" in result:
            for k, v in result["metrics"].items():
                if isinstance(v, float):
                    click.echo(f"  {k}: {v:.6f}")
    elif isinstance(result, list):
        click.echo(f"\nTrained {len(result)} models")
        for r in result:
            name = r.get("model", "?")
            status = r.get("status", "?")
            click.echo(f"  {name}: {status}")


async def _train_single(model: str, config, registry, lookback: int) -> dict:
    """Train a single model by name."""
    db_url = config.infra.database_url

    trainers = {
        "xgboost_prob": lambda: __import__(
            "apex.training.train_xgb", fromlist=["train_xgboost"]
        ).train_xgboost(db_url, registry, lookback),
        "lgbm_return": lambda: __import__(
            "apex.training.train_lgbm", fromlist=["train_lightgbm"]
        ).train_lightgbm(db_url, registry, lookback),
        "tft_quantile": lambda: __import__(
            "apex.training.train_tft", fromlist=["train_tft"]
        ).train_tft(db_url, registry, min(lookback, 30)),
        "lstm_regime": lambda: __import__(
            "apex.training.train_regime", fromlist=["train_regime_detector"]
        ).train_regime_detector(db_url, registry, min(lookback, 60)),
        "ppo_position_manager": lambda: __import__(
            "apex.training.train_ppo", fromlist=["train_ppo_position_manager"]
        ).train_ppo_position_manager(db_url, registry),
        "finbert_sentiment": lambda: __import__(
            "apex.training.train_finbert", fromlist=["fine_tune_finbert"]
        ).fine_tune_finbert(db_url, registry),
    }

    if model not in trainers:
        click.echo(f"Unknown model: {model}")
        click.echo(f"Available: {', '.join(trainers.keys())}")
        sys.exit(1)

    click.echo(f"Training {model}...")
    return await trainers[model]()


async def _train_all(config, registry, lookback: int) -> list[dict]:
    """Train all models sequentially."""
    from apex.training.scheduler import TrainingScheduler

    scheduler = TrainingScheduler(
        model_registry=registry, db_url=config.infra.database_url
    )
    results_dict = await scheduler.run_all()
    return [
        {**v, "model": k} if v else {"model": k, "status": "error"}
        for k, v in results_dict.items()
    ]


@main.command()
@click.option("--train-days", default=60, help="Training window size")
@click.option("--test-days", default=7, help="Test window size")
@click.option("--splits", default=10, help="Number of walk-forward splits")
def backtest(train_days: int, test_days: int, splits: int):
    """Run walk-forward backtest."""
    click.echo(
        f"Running walk-forward backtest: "
        f"{splits} splits, {train_days}d train / {test_days}d test"
    )

    from apex.config import load_config

    config = load_config()

    async def _backtest():
        from apex.research.walk_forward import WalkForwardValidator

        validator = WalkForwardValidator(
            train_days=train_days,
            test_days=test_days,
            n_splits=splits,
        )

        # Load data from DB
        import asyncpg
        import pandas as pd

        pool = await asyncpg.create_pool(config.infra.database_url)
        try:
            rows = await pool.fetch(
                """
                SELECT m.id, m.outcome, m.updated_at AS date,
                       fs.features
                FROM markets m
                JOIN LATERAL (
                    SELECT features FROM feature_store
                    WHERE entity_id = m.id AND feature_set = 'price_features'
                    ORDER BY time DESC LIMIT 1
                ) fs ON TRUE
                WHERE m.status = 'resolved' AND m.outcome IS NOT NULL
                ORDER BY m.updated_at
                """
            )
        finally:
            await pool.close()

        if not rows:
            click.echo("No resolved markets found for backtesting")
            return

        import json

        records = []
        for r in rows:
            feat = json.loads(r["features"]) if isinstance(r["features"], str) else r["features"]
            feat["outcome"] = int(r["outcome"])
            feat["date"] = r["date"]
            records.append(feat)

        data = pd.DataFrame(records)
        feature_cols = [c for c in data.columns if c not in ("outcome", "date")]

        from apex.models.xgboost_prob import XGBoostProbabilityModel

        results = validator.validate(
            model_factory=lambda: XGBoostProbabilityModel(
                params={"n_estimators": 200}
            ),
            data=data,
            feature_columns=feature_cols,
            target_column="outcome",
            date_column="date",
        )

        click.echo("\nWalk-Forward Results:")
        click.echo(f"  Splits: {results.get('n_splits', 0)}")
        click.echo(f"  Test samples: {results.get('total_test_samples', 0)}")

        for metric in ["accuracy", "brier_score", "ic", "rmse"]:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            if mean_key in results:
                click.echo(
                    f"  {metric}: {results[mean_key]:.4f} "
                    f"(+/- {results.get(std_key, 0):.4f})"
                )

    _run_async(_backtest())


@main.command()
def paper():
    """Start paper trading."""
    click.echo("Starting APEX in paper trading mode...")

    from apex.config import load_config
    from apex.node import run_node

    config = load_config()
    _run_async(run_node(config, mode="paper"))


@main.command()
@click.confirmation_option(
    prompt="Are you sure you want to start LIVE trading?"
)
def live():
    """Start live trading (requires confirmation)."""
    click.echo("Starting APEX in LIVE trading mode...")
    click.echo("WARNING: Real money will be at risk.")

    from apex.config import load_config
    from apex.node import run_node

    config = load_config()

    if config.trading_mode.value != "LIVE":
        click.echo(
            "ERROR: TRADING_MODE environment variable must be set to LIVE"
        )
        sys.exit(1)

    _run_async(run_node(config, mode="live"))


@main.command()
def status():
    """Show system status."""
    from apex.config import load_config

    config = load_config()

    async def _status():
        click.echo("APEX System Status")
        click.echo("=" * 40)
        click.echo(f"  Mode: {config.trading_mode.value}")
        click.echo(f"  DB: {config.infra.database_url[:40]}...")
        click.echo(f"  Redis: {config.infra.redis_url}")
        click.echo(
            f"  Telegram: {'enabled' if config.notifications.telegram_enabled else 'disabled'}"
        )

        # Model registry status
        from apex.models.registry import ModelRegistry

        registry = ModelRegistry(store_path=Path("models_store"))
        model_names = [
            "xgboost_prob", "lgbm_return", "tft_quantile",
            "lstm_regime", "ppo_position_manager", "finbert_sentiment",
        ]

        click.echo("\nModels:")
        for name in model_names:
            try:
                versions = registry.list_versions(name)
                prod = [v for v in versions if v.get("is_production")]
                prod_v = prod[-1]["version_id"] if prod else "none"
                click.echo(
                    f"  {name}: {len(versions)} versions, "
                    f"production={prod_v}"
                )
            except Exception:
                click.echo(f"  {name}: not registered")

        # Data health
        click.echo("\nData Sources:")
        try:
            from apex.monitoring.data_health import DataHealthMonitor

            monitor = DataHealthMonitor(
                redis_url=config.infra.redis_url,
                db_url=config.infra.database_url,
            )
            await monitor.connect()
            health = await monitor.check_all()
            await monitor.close()

            for name, info in health.get("sources", {}).items():
                short = name.replace("apex:", "")
                status_str = info.get("status", "unknown")
                lag = info.get("staleness_seconds")
                lag_str = f"{lag:.0f}s" if lag is not None else "--"
                click.echo(f"  {short}: {status_str} (lag: {lag_str})")

            click.echo(f"\n  Overall: {health.get('status', 'unknown')}")
        except Exception as exc:
            click.echo(f"  Error checking data health: {exc}")

        # Training schedule
        click.echo("\nTraining Schedule:")
        from apex.training.scheduler import TrainingScheduler

        scheduler = TrainingScheduler(
            model_registry=registry,
            db_url=config.infra.database_url,
        )
        for name, info in scheduler.status().items():
            sched = f"{info['schedule_type']}({info['schedule_value']})"
            last = info.get("last_run") or "never"
            click.echo(f"  {name}: {sched}, last={last}")

    _run_async(_status())


@main.command()
@click.option("--host", default="0.0.0.0", help="Dashboard host")
@click.option("--port", default=8080, help="Dashboard port")
def dashboard(host: str, port: int):
    """Start web dashboard."""
    import uvicorn

    from apex.config import load_config
    from apex.monitoring.dashboard import app, configure_dashboard

    config = load_config()
    configure_dashboard(
        db_url=config.infra.database_url,
        redis_url=config.infra.redis_url,
    )

    click.echo(f"Starting APEX Dashboard at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


@main.command()
@click.option("--poly-limit", default=50000, help="Polymarket market limit")
@click.option("--kalshi-limit", default=10000, help="Kalshi event limit")
@click.option("--store/--no-store", default=True, help="Store to database")
def collect(poly_limit: int, kalshi_limit: int, store: bool):
    """Collect historical data from prediction markets."""
    click.echo("Collecting historical market data...")

    from apex.config import load_config

    config = load_config()

    async def _collect():
        from apex.research.historical_data import (
            HistoricalDataCollector,
            store_to_database,
        )

        collector = HistoricalDataCollector(
            polymarket_api_key=config.venues.polymarket.api_key,
            kalshi_api_key=config.venues.kalshi.api_key,
        )

        dataset = await collector.build_training_dataset()

        click.echo(f"\nCollected {len(dataset)} markets:")
        click.echo(f"  Venues: {dataset['venue'].value_counts().to_dict()}")
        click.echo(
            f"  Outcomes: {dataset['outcome'].value_counts().to_dict()}"
        )

        if store and not dataset.empty:
            n = await store_to_database(dataset, config.infra.database_url)
            click.echo(f"  Stored {n} markets to database")
        else:
            click.echo("  Skipping database storage")

    _run_async(_collect())


@main.command()
def discover():
    """Run alpha discovery pipeline.

    Scans venues for new market categories, evaluates them with
    walk-forward backtesting, and recommends categories for paper
    trading deployment.
    """
    click.echo("Running strategy discovery pipeline...")

    from apex.config import load_config

    config = load_config()

    async def _discover():
        from apex.research.strategy_discovery import StrategyDiscovery

        discovery = StrategyDiscovery(db_url=config.infra.database_url)
        results = await discovery.discover_and_evaluate()

        if not results:
            click.echo("\nNo new categories found.")
            return

        click.echo(f"\nDiscovered {len(results)} categories:\n")
        for r in results:
            rec = r.get("recommendation", "?")
            cat = r.get("category", "?")
            venue = r.get("venue", "?")
            n = r.get("n_samples", 0)
            sharpe = r.get("oos_sharpe", "N/A")

            icon = (
                "[DEPLOY]"
                if rec == "deploy_paper"
                else "[MONITOR]"
                if rec == "monitor"
                else "[SKIP]"
            )
            click.echo(
                f"  {icon} {cat} ({venue}) - "
                f"{n} markets, Sharpe: {sharpe}"
            )

    _run_async(_discover())


@main.command()
@click.argument("model_name")
@click.option("--version", default="production", help="Version to inspect")
def inspect(model_name: str, version: str):
    """Inspect a registered model's details."""
    from apex.models.registry import ModelRegistry

    registry = ModelRegistry(store_path=Path("models_store"))

    try:
        versions = registry.list_versions(model_name)
    except Exception:
        click.echo(f"No versions found for '{model_name}'")
        return

    if not versions:
        click.echo(f"No versions registered for '{model_name}'")
        return

    click.echo(f"\nModel: {model_name}")
    click.echo(f"Total versions: {len(versions)}")
    click.echo("")

    for v in versions[-5:]:  # show last 5
        prod = " [PRODUCTION]" if v.get("is_production") else ""
        click.echo(f"  {v['version_id']}{prod}")
        click.echo(f"    Timestamp: {v.get('timestamp', '?')}")
        click.echo(f"    Checksum: {v.get('checksum', '?')}")
        metrics = v.get("metrics", {})
        if metrics:
            for k, val in list(metrics.items())[:5]:
                if isinstance(val, float):
                    click.echo(f"    {k}: {val:.6f}")
                else:
                    click.echo(f"    {k}: {val}")
        click.echo("")


if __name__ == "__main__":
    main()
