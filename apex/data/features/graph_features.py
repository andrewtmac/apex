"""
Graph / Relational Features (15 features)

Rolling correlations, spectral clustering, network-centrality measures,
momentum spillover, and regime-conditional correlations across a universe
of related markets.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from apex.data.features.builder import FeatureExtractor

_EPS = 1e-12


class GraphFeatureExtractor(FeatureExtractor):
    """Computes 15 graph/relational features.

    Expected keys in *raw_data*::

        # Pairwise return series for this market vs related markets
        # Each row = one related market, columns = time steps
        related_returns     : list[list[float]]   # shape (K, T)
        own_returns         : list[float]          # shape (T,)

        # Cluster assignment (pre-computed by clustering service)
        cluster_id          : int
        cluster_size        : int
        cluster_members_returns : list[list[float]]  # returns of cluster peers

        # Network centrality (pre-computed from correlation graph)
        degree_centrality      : float
        betweenness_centrality : float
        pagerank_score         : float

        # High / low vol regime masks (boolean arrays matching T)
        high_vol_mask       : list[bool]
        low_vol_mask        : list[bool]
    """

    _NAMES: list[str] = [
        # Correlation (3)
        "rolling_correlation_top5",
        "correlation_stability",
        "avg_pairwise_correlation",
        # Clustering (3)
        "cluster_id",
        "cluster_size",
        "intra_cluster_correlation",
        # Network (3)
        "degree_centrality",
        "betweenness_centrality",
        "pagerank_score",
        # Spillover (3)
        "lagged_correlation_1h",
        "momentum_spillover",
        "mean_reversion_spillover",
        # Conditional (3)
        "correlation_high_vol",
        "correlation_low_vol",
        "regime_conditional_corr",
    ]

    def feature_names(self) -> list[str]:
        return list(self._NAMES)

    # -- correlation helpers -----------------------------------------------

    @staticmethod
    def _pearson(x: np.ndarray, y: np.ndarray) -> float:
        """Pearson correlation, NaN-safe."""
        if len(x) < 2 or len(y) < 2:
            return 0.0
        n = min(len(x), len(y))
        x, y = x[-n:], y[-n:]
        xm = x - np.mean(x)
        ym = y - np.mean(y)
        denom = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
        if denom < _EPS:
            return 0.0
        return float(np.sum(xm * ym) / denom)

    @staticmethod
    def _rolling_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        """Rolling Pearson correlation with a fixed window."""
        n = min(len(x), len(y))
        if n < window:
            return np.array([0.0])
        x, y = x[-n:], y[-n:]
        out = np.empty(n - window + 1, dtype=np.float64)
        for i in range(len(out)):
            xw = x[i : i + window]
            yw = y[i : i + window]
            xm = xw - np.mean(xw)
            ym = yw - np.mean(yw)
            d = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
            out[i] = np.sum(xm * ym) / d if d > _EPS else 0.0
        return out

    # -- extract -----------------------------------------------------------

    async def extract(
        self,
        market_id: str,
        venue: str,
        raw_data: dict[str, Any],
    ) -> dict[str, float]:
        feat: dict[str, float] = {}

        own = np.asarray(raw_data.get("own_returns", []), dtype=np.float64)
        related_raw = raw_data.get("related_returns", [])
        related = [np.asarray(r, dtype=np.float64) for r in related_raw]

        # ---- Correlation ----
        if related and len(own) >= 5:
            # Top-5 most correlated (by absolute correlation)
            corrs = [self._pearson(own, r) for r in related]
            abs_sorted = np.argsort(-np.abs(corrs))
            top5_idx = abs_sorted[: min(5, len(corrs))]
            top5_corrs = [corrs[i] for i in top5_idx]
            feat["rolling_correlation_top5"] = float(np.mean(top5_corrs))

            # Stability: std of rolling correlation with the single most-correlated peer
            best_peer = related[abs_sorted[0]]
            rc = self._rolling_corr(own, best_peer, window=min(30, len(own) // 2 + 1))
            feat["correlation_stability"] = 1.0 - float(np.std(rc)) if len(rc) > 1 else 1.0

            # Average pairwise
            feat["avg_pairwise_correlation"] = float(np.mean(corrs))
        else:
            feat["rolling_correlation_top5"] = 0.0
            feat["correlation_stability"] = 1.0
            feat["avg_pairwise_correlation"] = 0.0

        # ---- Clustering ----
        feat["cluster_id"] = float(raw_data.get("cluster_id", 0))
        feat["cluster_size"] = float(raw_data.get("cluster_size", 1))

        cluster_ret_raw = raw_data.get("cluster_members_returns", [])
        if cluster_ret_raw and len(own) >= 5:
            cluster_rets = [np.asarray(r, dtype=np.float64) for r in cluster_ret_raw]
            intra_corrs = [self._pearson(own, r) for r in cluster_rets]
            feat["intra_cluster_correlation"] = float(np.mean(intra_corrs)) if intra_corrs else 0.0
        else:
            feat["intra_cluster_correlation"] = 0.0

        # ---- Network centrality (pass-through from pre-computed graph) ----
        feat["degree_centrality"] = float(raw_data.get("degree_centrality", 0.0))
        feat["betweenness_centrality"] = float(raw_data.get("betweenness_centrality", 0.0))
        feat["pagerank_score"] = float(raw_data.get("pagerank_score", 0.0))

        # ---- Spillover ----
        if related and len(own) >= 10:
            # Lagged correlation (peer returns at t-1 vs own at t)
            lag_corrs = []
            for r in related:
                n = min(len(own), len(r))
                if n >= 10:
                    lag_corrs.append(self._pearson(own[1:n], r[: n - 1]))
            feat["lagged_correlation_1h"] = float(np.mean(lag_corrs)) if lag_corrs else 0.0

            # Momentum spillover: avg recent return of related markets
            recent_peer_rets = [float(np.sum(r[-5:])) for r in related if len(r) >= 5]
            feat["momentum_spillover"] = (
                float(np.mean(recent_peer_rets)) if recent_peer_rets else 0.0
            )

            # Mean-reversion spillover: negative of momentum (contrarian signal)
            feat["mean_reversion_spillover"] = -feat["momentum_spillover"]
        else:
            feat["lagged_correlation_1h"] = 0.0
            feat["momentum_spillover"] = 0.0
            feat["mean_reversion_spillover"] = 0.0

        # ---- Conditional correlations ----
        high_mask = np.asarray(raw_data.get("high_vol_mask", []), dtype=bool)
        low_mask = np.asarray(raw_data.get("low_vol_mask", []), dtype=bool)

        if related and len(own) >= 10:
            best_peer = related[0]
            n = min(len(own), len(best_peer))
            o, p = own[-n:], best_peer[-n:]

            # High-vol conditional correlation
            if len(high_mask) == n and np.sum(high_mask) >= 5:
                feat["correlation_high_vol"] = self._pearson(o[high_mask], p[high_mask])
            else:
                feat["correlation_high_vol"] = feat.get("rolling_correlation_top5", 0.0)

            # Low-vol conditional correlation
            if len(low_mask) == n and np.sum(low_mask) >= 5:
                feat["correlation_low_vol"] = self._pearson(o[low_mask], p[low_mask])
            else:
                feat["correlation_low_vol"] = feat.get("rolling_correlation_top5", 0.0)

            # Regime difference
            feat["regime_conditional_corr"] = (
                feat["correlation_high_vol"] - feat["correlation_low_vol"]
            )
        else:
            feat["correlation_high_vol"] = 0.0
            feat["correlation_low_vol"] = 0.0
            feat["regime_conditional_corr"] = 0.0

        return feat
