"""Orchestrates model retraining on schedule.

Schedule:
- Daily at 00:00 UTC: XGBoost (Model A) + LightGBM (Model B)
- Weekly Sunday 00:00 UTC: TFT (Model C) + LSTM Regime (Model D)
- Every 3 days at 02:00 UTC: PPO Position Manager (Model E)
- Monthly 1st at 04:00 UTC: FinBERT (Model F)
- Continuous: Bayesian Calibration (Model G) - no retrain needed
"""

from __future__ import annotations

import asyncio
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

import structlog

from apex.models.registry import ModelRegistry

logger = structlog.get_logger(__name__)


class TrainingJob:
    """Represents a single scheduled training job."""

    __slots__ = (
        "name",
        "train_fn",
        "schedule_type",
        "schedule_value",
        "last_run",
        "last_result",
        "running",
        "enabled",
    )

    def __init__(
        self,
        name: str,
        train_fn: Callable[..., Coroutine[Any, Any, dict]],
        schedule_type: str,
        schedule_value: int | str,
    ) -> None:
        self.name = name
        self.train_fn = train_fn
        self.schedule_type = schedule_type  # "daily", "weekly", "interval_days", "monthly"
        self.schedule_value = schedule_value  # day-of-week for weekly, interval for interval_days
        self.last_run: datetime | None = None
        self.last_result: dict | None = None
        self.running = False
        self.enabled = True


class TrainingScheduler:
    """Manages and executes the model retraining schedule.

    Each model has a defined training cadence.  The scheduler runs a
    polling loop, checks if each job is due, and runs them with
    concurrency control (max 1 training job at a time to avoid
    GPU/memory contention).
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        db_url: str,
        on_complete: Callable[[str, dict], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self.registry = model_registry
        self.db_url = db_url
        self._on_complete = on_complete
        self._jobs: dict[str, TrainingJob] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self.schedule = self._build_schedule()

    def _build_schedule(self) -> dict[str, TrainingJob]:
        """Define the training schedule for all models."""
        from apex.training.train_finbert import fine_tune_finbert
        from apex.training.train_lgbm import train_lightgbm
        from apex.training.train_ppo import train_ppo_position_manager
        from apex.training.train_regime import train_regime_detector
        from apex.training.train_tft import train_tft
        from apex.training.train_xgb import train_xgboost

        db = self.db_url
        reg = self.registry

        jobs = {
            "xgboost_prob": TrainingJob(
                name="xgboost_prob",
                train_fn=lambda: train_xgboost(db, reg, lookback_days=90),
                schedule_type="daily",
                schedule_value=0,  # hour (00:00 UTC)
            ),
            "lgbm_return": TrainingJob(
                name="lgbm_return",
                train_fn=lambda: train_lightgbm(db, reg, lookback_days=90),
                schedule_type="daily",
                schedule_value=0,
            ),
            "tft_quantile": TrainingJob(
                name="tft_quantile",
                train_fn=lambda: train_tft(db, reg, lookback_days=30),
                schedule_type="weekly",
                schedule_value=6,  # Sunday (0=Monday in isoweekday)
            ),
            "lstm_regime": TrainingJob(
                name="lstm_regime",
                train_fn=lambda: train_regime_detector(db, reg, lookback_days=60),
                schedule_type="weekly",
                schedule_value=6,
            ),
            "ppo_position_manager": TrainingJob(
                name="ppo_position_manager",
                train_fn=lambda: train_ppo_position_manager(
                    db, reg, total_timesteps=100_000
                ),
                schedule_type="interval_days",
                schedule_value=3,
            ),
            "finbert_sentiment": TrainingJob(
                name="finbert_sentiment",
                train_fn=lambda: fine_tune_finbert(db, reg),
                schedule_type="monthly",
                schedule_value=1,  # 1st of month
            ),
        }

        self._jobs = jobs
        return jobs

    def _is_due(self, job: TrainingJob, now: datetime) -> bool:
        """Check if a training job is due to run."""
        if not job.enabled or job.running:
            return False

        if job.last_run is None:
            return True  # never run before

        if job.schedule_type == "daily":
            # Run once per day at the specified hour
            target_hour = int(job.schedule_value)
            if now.hour == target_hour:
                last_date = job.last_run.date()
                if now.date() > last_date:
                    return True

        elif job.schedule_type == "weekly":
            # Run on specified day of week (0=Mon, 6=Sun)
            target_day = int(job.schedule_value)
            if now.weekday() == target_day:
                days_since = (now - job.last_run).days
                if days_since >= 6:  # at least 6 days since last run
                    return True

        elif job.schedule_type == "interval_days":
            interval = int(job.schedule_value)
            days_since = (now - job.last_run).total_seconds() / 86400
            if days_since >= interval:
                return True

        elif job.schedule_type == "monthly":
            target_day = int(job.schedule_value)
            if now.day == target_day:
                last_month = job.last_run.month
                if now.month != last_month or now.year != job.last_run.year:
                    return True

        return False

    async def _run_job(self, job: TrainingJob) -> dict | None:
        """Execute a single training job with error handling."""
        async with self._lock:
            job.running = True
            logger.info("scheduler.job_start", job=job.name)

            try:
                result = await job.train_fn()
                job.last_result = result
                job.last_run = datetime.now(timezone.utc)

                logger.info(
                    "scheduler.job_complete",
                    job=job.name,
                    status=result.get("status", "unknown"),
                    version=result.get("version_id", ""),
                )

                if self._on_complete:
                    await self._on_complete(job.name, result)

                return result

            except Exception as exc:
                logger.error(
                    "scheduler.job_failed",
                    job=job.name,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
                job.last_result = {
                    "status": "error",
                    "error": str(exc),
                }
                job.last_run = datetime.now(timezone.utc)
                return None

            finally:
                job.running = False

    async def run(self) -> None:
        """Main scheduler loop.

        Polls every 60 seconds, checks which jobs are due, and runs them
        sequentially (one at a time to manage resource usage).
        """
        self._running = True
        logger.info(
            "scheduler.start",
            jobs=[j.name for j in self._jobs.values()],
        )

        while self._running:
            now = datetime.now(timezone.utc)

            for job in self._jobs.values():
                if self._is_due(job, now):
                    await self._run_job(job)

                if not self._running:
                    break

            # Sleep for 60 seconds before next check
            await asyncio.sleep(60)

        logger.info("scheduler.stopped")

    async def stop(self) -> None:
        """Signal the scheduler to stop after the current job completes."""
        self._running = False

    async def run_now(self, model_name: str) -> dict | None:
        """Manually trigger a specific training job immediately."""
        job = self._jobs.get(model_name)
        if job is None:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(self._jobs.keys())}"
            )
        return await self._run_job(job)

    async def run_all(self) -> dict[str, dict | None]:
        """Run all training jobs immediately (sequential)."""
        results: dict[str, dict | None] = {}
        for name, job in self._jobs.items():
            results[name] = await self._run_job(job)
        return results

    def status(self) -> dict[str, dict[str, Any]]:
        """Return the status of all training jobs."""
        result: dict[str, dict[str, Any]] = {}
        for name, job in self._jobs.items():
            result[name] = {
                "schedule_type": job.schedule_type,
                "schedule_value": job.schedule_value,
                "enabled": job.enabled,
                "running": job.running,
                "last_run": job.last_run.isoformat() if job.last_run else None,
                "last_status": (
                    job.last_result.get("status") if job.last_result else None
                ),
            }
        return result

    def enable(self, model_name: str) -> None:
        """Enable a training job."""
        if model_name in self._jobs:
            self._jobs[model_name].enabled = True

    def disable(self, model_name: str) -> None:
        """Disable a training job."""
        if model_name in self._jobs:
            self._jobs[model_name].enabled = False
