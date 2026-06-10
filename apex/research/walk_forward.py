"""Walk-forward validation framework.

Implements rolling train/test splits for rigorous strategy evaluation.
Avoids look-ahead bias by strictly separating train and test windows
chronologically, with an optional gap period between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class WalkForwardSplit:
    """A single train/test split in the walk-forward sequence."""

    split_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_data: pd.DataFrame | None = None
    test_data: pd.DataFrame | None = None


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward split."""

    split_idx: int
    train_size: int
    test_size: int
    metrics: dict[str, float] = field(default_factory=dict)
    predictions: np.ndarray | None = None
    actuals: np.ndarray | None = None


class WalkForwardValidator:
    """Walk-forward backtesting for time-series model validation.

    Generates rolling train/test windows that advance through time,
    ensuring no data leakage.  Each split trains a model from scratch
    on the training window and evaluates on the subsequent test window.

    Parameters
    ----------
    train_days : number of days in each training window
    test_days : number of days in each test window
    n_splits : maximum number of train/test splits
    gap_days : days between train end and test start (embargo period)
    min_train_samples : minimum samples required in training window
    """

    def __init__(
        self,
        train_days: int = 60,
        test_days: int = 7,
        n_splits: int = 10,
        gap_days: int = 0,
        min_train_samples: int = 50,
    ) -> None:
        self.train_days = train_days
        self.test_days = test_days
        self.n_splits = n_splits
        self.gap_days = gap_days
        self.min_train_samples = min_train_samples

    def generate_splits(
        self,
        data: pd.DataFrame,
        date_column: str = "date",
    ) -> list[WalkForwardSplit]:
        """Generate rolling train/test splits.

        Parameters
        ----------
        data : DataFrame sorted by date_column
        date_column : name of the datetime column

        Returns
        -------
        list of WalkForwardSplit objects with data attached
        """
        if date_column not in data.columns:
            # If no date column, use index positions
            return self._generate_index_splits(data)

        data = data.sort_values(date_column).reset_index(drop=True)

        dates = pd.to_datetime(data[date_column])
        min_date = dates.min()
        max_date = dates.max()
        total_days = (max_date - min_date).days

        # Calculate step size to fit n_splits
        window_size = self.train_days + self.gap_days + self.test_days
        available_days = total_days - window_size
        if available_days <= 0:
            raise ValueError(
                f"Data span ({total_days} days) too short for "
                f"window size ({window_size} days)"
            )

        step = max(1, available_days // max(self.n_splits - 1, 1))
        splits: list[WalkForwardSplit] = []

        for i in range(self.n_splits):
            offset = pd.Timedelta(days=i * step)
            train_start_date = min_date + offset
            train_end_date = train_start_date + pd.Timedelta(days=self.train_days)
            test_start_date = train_end_date + pd.Timedelta(days=self.gap_days)
            test_end_date = test_start_date + pd.Timedelta(days=self.test_days)

            if test_end_date > max_date:
                break

            train_mask = (dates >= train_start_date) & (dates < train_end_date)
            test_mask = (dates >= test_start_date) & (dates < test_end_date)

            train_data = data[train_mask].copy()
            test_data = data[test_mask].copy()

            if len(train_data) < self.min_train_samples:
                continue

            if len(test_data) == 0:
                continue

            split = WalkForwardSplit(
                split_idx=len(splits),
                train_start=int(train_data.index[0]),
                train_end=int(train_data.index[-1]),
                test_start=int(test_data.index[0]),
                test_end=int(test_data.index[-1]),
                train_data=train_data,
                test_data=test_data,
            )
            splits.append(split)

        logger.info(
            "walk_forward.splits_generated",
            n_splits=len(splits),
            data_rows=len(data),
        )
        return splits

    def _generate_index_splits(
        self, data: pd.DataFrame
    ) -> list[WalkForwardSplit]:
        """Generate splits based on row indices when no date column exists."""
        n = len(data)
        train_size = int(n * 0.6 / max(self.n_splits, 1))
        test_size = int(n * 0.1 / max(self.n_splits, 1))
        gap_size = int(n * 0.02)

        if train_size < self.min_train_samples:
            train_size = self.min_train_samples

        step = max(1, (n - train_size - gap_size - test_size) // max(self.n_splits - 1, 1))

        splits: list[WalkForwardSplit] = []
        for i in range(self.n_splits):
            start = i * step
            train_end = start + train_size
            test_start = train_end + gap_size
            test_end = test_start + test_size

            if test_end > n:
                break

            split = WalkForwardSplit(
                split_idx=len(splits),
                train_start=start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_data=data.iloc[start:train_end].copy(),
                test_data=data.iloc[test_start:test_end].copy(),
            )
            splits.append(split)

        return splits

    def validate(
        self,
        model_factory: Callable[[], Any],
        data: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        date_column: str = "date",
        metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]] | None = None,
    ) -> dict[str, Any]:
        """Run walk-forward validation.

        Parameters
        ----------
        model_factory : callable that returns a fresh, untrained model instance
            The model must have train(X, y, feature_names) and predict(X) methods.
        data : full dataset
        feature_columns : list of feature column names
        target_column : name of the target column
        date_column : date column for splitting
        metric_fns : optional dict of {metric_name: fn(y_true, y_pred)}

        Returns
        -------
        dict with aggregate metrics, per-split results, and stability measures
        """
        if metric_fns is None:
            metric_fns = self._default_metrics(data, target_column)

        splits = self.generate_splits(data, date_column)

        if not splits:
            raise ValueError("No valid splits generated from data")

        split_results: list[WalkForwardResult] = []

        for split in splits:
            assert split.train_data is not None and split.test_data is not None

            X_train = split.train_data[feature_columns].values
            y_train = split.train_data[target_column].values
            X_test = split.test_data[feature_columns].values
            y_test = split.test_data[target_column].values

            # Train fresh model
            model = model_factory()

            try:
                if hasattr(model, "train"):
                    model.train(X_train, y_train, feature_columns)
                elif hasattr(model, "fit"):
                    model.fit(X_train, y_train)
                else:
                    raise TypeError(
                        "Model must have a train() or fit() method"
                    )

                predictions = np.asarray(model.predict(X_test)).ravel()
            except Exception as exc:
                logger.warning(
                    "walk_forward.split_failed",
                    split=split.split_idx,
                    error=str(exc),
                )
                continue

            # Compute metrics for this split
            metrics: dict[str, float] = {}
            for name, fn in metric_fns.items():
                try:
                    metrics[name] = float(fn(y_test, predictions))
                except Exception:
                    metrics[name] = float("nan")

            result = WalkForwardResult(
                split_idx=split.split_idx,
                train_size=len(X_train),
                test_size=len(X_test),
                metrics=metrics,
                predictions=predictions,
                actuals=y_test,
            )
            split_results.append(result)

            logger.info(
                "walk_forward.split_complete",
                split=split.split_idx,
                train_size=len(X_train),
                test_size=len(X_test),
                metrics={k: round(v, 4) for k, v in metrics.items()},
            )

        return self._aggregate_results(split_results)

    def _aggregate_results(
        self,
        results: list[WalkForwardResult],
    ) -> dict[str, Any]:
        """Aggregate per-split results into summary statistics."""
        if not results:
            return {
                "n_splits": 0,
                "error": "No successful splits",
            }

        metric_names = results[0].metrics.keys()
        aggregated: dict[str, Any] = {
            "n_splits": len(results),
            "total_test_samples": sum(r.test_size for r in results),
        }

        for metric in metric_names:
            values = [
                r.metrics[metric]
                for r in results
                if not np.isnan(r.metrics.get(metric, float("nan")))
            ]
            if values:
                aggregated[f"{metric}_mean"] = float(np.mean(values))
                aggregated[f"{metric}_std"] = float(np.std(values))
                aggregated[f"{metric}_min"] = float(np.min(values))
                aggregated[f"{metric}_max"] = float(np.max(values))
                aggregated[f"{metric}_median"] = float(np.median(values))

                # Stability: what fraction of splits are positive?
                if "sharpe" in metric.lower() or "return" in metric.lower():
                    aggregated[f"{metric}_positive_pct"] = float(
                        np.mean(np.array(values) > 0)
                    )

        # Per-split details
        aggregated["per_split"] = [
            {
                "split": r.split_idx,
                "train_size": r.train_size,
                "test_size": r.test_size,
                **{k: round(v, 6) for k, v in r.metrics.items()},
            }
            for r in results
        ]

        return aggregated

    def _default_metrics(
        self,
        data: pd.DataFrame,
        target_column: str,
    ) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        """Generate default metric functions based on target type."""
        from sklearn.metrics import (
            accuracy_score,
            brier_score_loss,
            mean_absolute_error,
            mean_squared_error,
        )

        # Detect if binary classification
        unique_vals = data[target_column].dropna().unique()
        is_binary = set(unique_vals).issubset({0, 1, 0.0, 1.0})

        metrics: dict[str, Callable] = {
            "mse": lambda y, p: mean_squared_error(y, p),
            "rmse": lambda y, p: float(np.sqrt(mean_squared_error(y, p))),
            "mae": lambda y, p: mean_absolute_error(y, p),
        }

        if is_binary:
            metrics["brier_score"] = lambda y, p: brier_score_loss(y, np.clip(p, 0, 1))
            metrics["accuracy"] = lambda y, p: accuracy_score(
                y, (np.asarray(p) >= 0.5).astype(int)
            )

        # Information coefficient (rank correlation)
        def _ic(y: np.ndarray, p: np.ndarray) -> float:
            from scipy.stats import spearmanr

            corr, _ = spearmanr(y, p)
            return float(corr) if np.isfinite(corr) else 0.0

        metrics["ic"] = _ic

        return metrics
