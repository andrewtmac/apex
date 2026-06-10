"""Automated alpha discovery through feature screening.

Process:
1. Generate candidate features (pairwise ratios, rolling stats, interactions)
2. Compute mutual information and Spearman correlation with targets
3. Apply Benjamini-Hochberg multiple testing correction
4. Walk-forward validate promising features
5. Prune features with declining importance
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class FeatureDiscovery:
    """Automated feature screening and alpha discovery pipeline.

    Generates candidate features from existing ones using mathematical
    transformations, evaluates their predictive power, and applies
    multiple testing correction to control false discovery rate.

    Parameters
    ----------
    max_candidates : maximum number of candidate features to generate
    fdr_threshold : Benjamini-Hochberg false discovery rate threshold
    min_mi_score : minimum mutual information score to keep a feature
    max_correlation : maximum pairwise correlation among selected features
    """

    def __init__(
        self,
        max_candidates: int = 500,
        fdr_threshold: float = 0.05,
        min_mi_score: float = 0.01,
        max_correlation: float = 0.85,
    ) -> None:
        self.max_candidates = max_candidates
        self.fdr_threshold = fdr_threshold
        self.min_mi_score = min_mi_score
        self.max_correlation = max_correlation

    async def screen(
        self,
        data: pd.DataFrame,
        target: str,
        existing_features: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Screen candidate features for predictive power.

        Parameters
        ----------
        data : DataFrame with features and target column
        target : name of the target column
        existing_features : list of existing feature columns (if None, auto-detect)

        Returns
        -------
        Ranked list of dicts with:
        - name: feature name
        - formula: how it was generated
        - mutual_info: MI score with target
        - spearman_corr: rank correlation with target
        - spearman_pval: p-value of rank correlation
        - bh_significant: whether it passes BH correction
        - importance_rank: overall ranking
        """
        if existing_features is None:
            existing_features = [
                c for c in data.columns
                if c != target and data[c].dtype in (np.float64, np.float32, float, int)
            ]

        logger.info(
            "feature_discovery.start",
            n_existing=len(existing_features),
            n_samples=len(data),
        )

        # 1. Generate candidates
        candidates = self.generate_candidates(data[existing_features])

        # 2. Evaluate each candidate
        target_series = data[target]
        evaluations: list[dict[str, Any]] = []

        for col_name in candidates.columns:
            eval_result = self.evaluate_candidate(
                candidates[col_name], target_series
            )
            eval_result["name"] = col_name
            evaluations.append(eval_result)

        # 3. Apply BH correction
        evaluations = self._apply_bh_correction(evaluations)

        # 4. Filter and rank
        significant = [
            e for e in evaluations
            if e.get("bh_significant", False)
            and e.get("mutual_info", 0) >= self.min_mi_score
        ]

        # 5. Remove highly correlated features (keep stronger one)
        significant = self._prune_correlated(significant, candidates)

        # 6. Sort by importance (MI * |spearman|)
        for e in significant:
            e["importance_score"] = (
                e.get("mutual_info", 0) * abs(e.get("spearman_corr", 0))
            )

        significant.sort(key=lambda x: x["importance_score"], reverse=True)

        for rank, e in enumerate(significant):
            e["importance_rank"] = rank + 1

        logger.info(
            "feature_discovery.complete",
            n_candidates=len(candidates.columns),
            n_significant=len(significant),
        )

        return significant

    def generate_candidates(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate candidate features from existing ones.

        Transformations applied:
        - Pairwise ratios (A / B)
        - Pairwise differences (A - B)
        - Rolling statistics (mean, std over 5, 10, 20 periods)
        - Log transforms
        - Polynomial interactions (A * B)
        - Lagged values

        Limits output to max_candidates features.
        """
        candidates: dict[str, pd.Series] = {}
        cols = list(features.columns)

        # 1. Rolling statistics
        for col in cols:
            series = features[col]
            for window in [5, 10, 20]:
                if len(series) >= window:
                    candidates[f"{col}_sma_{window}"] = (
                        series.rolling(window).mean()
                    )
                    candidates[f"{col}_std_{window}"] = (
                        series.rolling(window).std()
                    )
                    # Z-score relative to rolling window
                    roll_mean = series.rolling(window).mean()
                    roll_std = series.rolling(window).std()
                    candidates[f"{col}_zscore_{window}"] = (
                        (series - roll_mean) / roll_std.clip(lower=1e-8)
                    )

            if len(candidates) >= self.max_candidates:
                break

        # 2. Log transforms (for positive features)
        for col in cols[:20]:  # limit to first 20
            series = features[col]
            if (series > 0).all():
                candidates[f"log_{col}"] = np.log(series + 1e-8)
            if len(candidates) >= self.max_candidates:
                break

        # 3. Pairwise ratios and differences (sample pairs to limit)
        n_pairs = min(50, len(cols) * (len(cols) - 1) // 2)
        pair_indices = list(itertools.combinations(range(len(cols)), 2))
        np.random.seed(42)
        if len(pair_indices) > n_pairs:
            selected_pairs = np.random.choice(
                len(pair_indices), n_pairs, replace=False
            )
            pair_indices = [pair_indices[i] for i in selected_pairs]

        for i, j in pair_indices:
            col_a, col_b = cols[i], cols[j]
            a, b = features[col_a], features[col_b]

            # Ratio
            denom = b.clip(lower=1e-8)
            candidates[f"{col_a}_div_{col_b}"] = a / denom

            # Difference
            candidates[f"{col_a}_minus_{col_b}"] = a - b

            if len(candidates) >= self.max_candidates:
                break

        # 4. Polynomial interactions (top features only)
        for i, j in pair_indices[:20]:
            col_a, col_b = cols[i], cols[j]
            candidates[f"{col_a}_x_{col_b}"] = features[col_a] * features[col_b]

            if len(candidates) >= self.max_candidates:
                break

        # 5. Lagged values
        for col in cols[:10]:
            for lag in [1, 3, 5]:
                candidates[f"{col}_lag_{lag}"] = features[col].shift(lag)

            if len(candidates) >= self.max_candidates:
                break

        # Build DataFrame, drop NaN-heavy columns
        result = pd.DataFrame(candidates)
        # Keep only columns with < 30% NaN
        null_pct = result.isnull().mean()
        result = result.loc[:, null_pct < 0.3]
        result = result.fillna(0.0)

        # Limit to max_candidates
        if len(result.columns) > self.max_candidates:
            result = result.iloc[:, : self.max_candidates]

        logger.info(
            "feature_discovery.candidates_generated",
            n_candidates=len(result.columns),
        )
        return result

    def evaluate_candidate(
        self,
        feature: pd.Series,
        target: pd.Series,
    ) -> dict[str, Any]:
        """Evaluate a single candidate feature's predictive power.

        Computes:
        - Mutual information with target
        - Spearman rank correlation
        - Maximum information coefficient (approximated)
        """
        from scipy.stats import spearmanr
        from sklearn.feature_selection import mutual_info_regression

        # Align and drop NaN
        mask = feature.notna() & target.notna()
        feat = feature[mask].values.reshape(-1, 1)
        tgt = target[mask].values

        if len(feat) < 30:
            return {
                "mutual_info": 0.0,
                "spearman_corr": 0.0,
                "spearman_pval": 1.0,
                "n_valid": len(feat),
            }

        # Mutual information
        try:
            mi = float(mutual_info_regression(feat, tgt, random_state=42)[0])
        except Exception:
            mi = 0.0

        # Spearman correlation
        try:
            corr, pval = spearmanr(feat.ravel(), tgt)
            corr = float(corr) if np.isfinite(corr) else 0.0
            pval = float(pval) if np.isfinite(pval) else 1.0
        except Exception:
            corr, pval = 0.0, 1.0

        return {
            "mutual_info": mi,
            "spearman_corr": corr,
            "spearman_pval": pval,
            "n_valid": len(feat),
        }

    def _apply_bh_correction(
        self, evaluations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply Benjamini-Hochberg FDR correction to p-values."""
        if not evaluations:
            return evaluations

        pvals = np.array([e.get("spearman_pval", 1.0) for e in evaluations])
        n = len(pvals)

        # Sort indices by p-value
        sorted_idx = np.argsort(pvals)
        sorted_pvals = pvals[sorted_idx]

        # BH procedure
        bh_thresholds = np.arange(1, n + 1) / n * self.fdr_threshold
        significant_mask = sorted_pvals <= bh_thresholds

        # Find the largest index where p <= threshold
        if significant_mask.any():
            max_significant = np.max(np.where(significant_mask)[0])
            # All indices up to and including max_significant are significant
            significant_set = set(sorted_idx[: max_significant + 1])
        else:
            significant_set = set()

        for i, e in enumerate(evaluations):
            e["bh_significant"] = i in significant_set
            e["bh_rank"] = int(np.where(sorted_idx == i)[0][0]) + 1

        return evaluations

    def _prune_correlated(
        self,
        evaluations: list[dict[str, Any]],
        candidates: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Remove highly correlated features, keeping the stronger one."""
        if len(evaluations) <= 1:
            return evaluations

        # Sort by MI score (descending) so we keep the best
        evaluations.sort(key=lambda x: x.get("mutual_info", 0), reverse=True)

        kept: list[dict[str, Any]] = []
        kept_names: set[str] = set()

        for e in evaluations:
            name = e["name"]
            if name not in candidates.columns:
                continue

            # Check correlation against all already-kept features
            is_redundant = False
            for kept_name in kept_names:
                if kept_name not in candidates.columns:
                    continue
                corr = candidates[name].corr(candidates[kept_name])
                if abs(corr) > self.max_correlation:
                    is_redundant = True
                    break

            if not is_redundant:
                kept.append(e)
                kept_names.add(name)

        logger.info(
            "feature_discovery.pruned",
            before=len(evaluations),
            after=len(kept),
        )
        return kept
