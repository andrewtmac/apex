"""
APEX Cross-Platform Correlation Monitor

Tracks cross-platform and cross-market correlations to prevent hidden
concentration risk.  When positions on different venues or markets are
correlated (e.g. two Polymarket contracts on the same underlying event),
the effective risk is much higher than the sum of individual risks.

Key capabilities:
- Rolling correlation matrix from recent returns
- Correlated position pair detection
- Net directional exposure per underlying event
- Correlation-based portfolio risk adjustment
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CorrelationMonitor:
    """Tracks cross-platform and cross-market correlations.

    Parameters
    ----------
    lookback_days : int
        Number of days of returns history to use for correlation computation.
    min_observations : int
        Minimum number of overlapping return observations required before
        computing correlation.  Below this, correlation defaults to 0.0.
    update_interval : int
        Minimum number of new observations between correlation matrix
        recomputations.
    """

    def __init__(
        self,
        lookback_days: int = 7,
        min_observations: int = 20,
        update_interval: int = 10,
    ) -> None:
        self.lookback_days = lookback_days
        self.min_observations = min_observations
        self.update_interval = update_interval

        # Returns keyed by position/market identifier
        self._returns: dict[str, list[float]] = defaultdict(list)
        self._market_ids: list[str] = []

        # Cached correlation matrix
        self.correlation_matrix: np.ndarray | None = None
        self._matrix_market_ids: list[str] = []
        self._obs_since_update: int = 0

    # ------------------------------------------------------------------
    # Returns tracking
    # ------------------------------------------------------------------

    def add_return(self, market_id: str, ret: float) -> None:
        """Record a single return observation for a market.

        Parameters
        ----------
        market_id : str
            Unique market identifier.
        ret : float
            Return for this period.
        """
        if market_id not in self._returns:
            self._market_ids.append(market_id)

        self._returns[market_id].append(ret)

        # Trim to lookback (assuming roughly 1 obs per hour, 24 * lookback_days)
        max_obs = self.lookback_days * 24
        if len(self._returns[market_id]) > max_obs:
            self._returns[market_id] = self._returns[market_id][-max_obs:]

        self._obs_since_update += 1

    def update(self, returns: dict[str, list[float]]) -> None:
        """Bulk update correlation data from recent returns.

        Parameters
        ----------
        returns : dict[str, list[float]]
            Market ID -> list of recent returns.  Overwrites existing
            data for each market.
        """
        for market_id, ret_list in returns.items():
            if market_id not in self._returns:
                self._market_ids.append(market_id)
            self._returns[market_id] = list(ret_list)

        self._recompute_matrix()

    # ------------------------------------------------------------------
    # Correlation matrix computation
    # ------------------------------------------------------------------

    def _recompute_matrix(self) -> None:
        """Recompute the correlation matrix from buffered returns."""
        active_ids = [
            mid for mid in self._market_ids
            if len(self._returns.get(mid, [])) >= self.min_observations
        ]

        if len(active_ids) < 2:
            self.correlation_matrix = None
            self._matrix_market_ids = active_ids
            return

        # Find the minimum common length
        min_len = min(len(self._returns[mid]) for mid in active_ids)
        min_len = max(min_len, self.min_observations)

        # Build returns matrix (n_obs x n_markets)
        returns_matrix = np.array([
            self._returns[mid][-min_len:]
            for mid in active_ids
        ]).T  # Shape: (min_len, n_markets)

        # Compute correlation matrix
        if returns_matrix.shape[0] < 2:
            self.correlation_matrix = np.eye(len(active_ids))
        else:
            # Use numpy corrcoef with NaN handling
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(returns_matrix, rowvar=False)

            # Replace NaN with 0 (happens when a series has zero variance)
            corr = np.nan_to_num(corr, nan=0.0)

            # Ensure diagonal is exactly 1.0
            np.fill_diagonal(corr, 1.0)

            self.correlation_matrix = corr

        self._matrix_market_ids = active_ids
        self._obs_since_update = 0

        logger.debug(
            "Correlation matrix recomputed: %d x %d",
            len(active_ids),
            len(active_ids),
        )

    def get_correlation_matrix(self) -> tuple[np.ndarray | None, list[str]]:
        """Return the current correlation matrix and its market ID ordering.

        Returns
        -------
        tuple[np.ndarray | None, list[str]]
            ``(matrix, market_ids)`` -- matrix is None if insufficient data.
        """
        # Recompute if stale
        if self._obs_since_update >= self.update_interval:
            self._recompute_matrix()

        return self.correlation_matrix, list(self._matrix_market_ids)

    # ------------------------------------------------------------------
    # Correlated position detection
    # ------------------------------------------------------------------

    def get_correlated_positions(
        self,
        threshold: float = 0.7,
    ) -> list[tuple[str, str, float]]:
        """Find position pairs with correlation above threshold.

        Parameters
        ----------
        threshold : float
            Absolute correlation threshold (0.0 to 1.0).

        Returns
        -------
        list[tuple[str, str, float]]
            List of ``(market_id_1, market_id_2, correlation)`` tuples
            where ``|correlation| >= threshold``.
        """
        matrix, market_ids = self.get_correlation_matrix()
        if matrix is None or len(market_ids) < 2:
            return []

        pairs: list[tuple[str, str, float]] = []
        n = len(market_ids)

        for i in range(n):
            for j in range(i + 1, n):
                corr = float(matrix[i, j])
                if abs(corr) >= threshold:
                    pairs.append((market_ids[i], market_ids[j], corr))

        # Sort by absolute correlation (descending)
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        return pairs

    def max_correlation_with(
        self,
        market_id: str,
        exclude_self: bool = True,
    ) -> float:
        """Maximum absolute correlation between a market and any other position.

        Returns 0.0 if the market is not in the correlation matrix.
        """
        matrix, market_ids = self.get_correlation_matrix()
        if matrix is None or market_id not in market_ids:
            return 0.0

        idx = market_ids.index(market_id)
        row = np.abs(matrix[idx])

        if exclude_self:
            row = np.delete(row, idx)

        if len(row) == 0:
            return 0.0

        return float(np.max(row))

    # ------------------------------------------------------------------
    # Net exposure
    # ------------------------------------------------------------------

    def net_exposure(
        self,
        positions: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Compute net directional exposure per underlying event.

        Groups positions by their ``"underlying_event"`` or ``"sector"``
        field and sums signed exposures.

        Parameters
        ----------
        positions : list[dict]
            Each dict should have:

            - ``"market_id"`` : str
            - ``"size_usd"`` : float
            - ``"direction"`` : int (+1 or -1)
            - ``"underlying_event"`` or ``"sector"`` : str (grouping key)

        Returns
        -------
        dict[str, float]
            Event/sector -> net signed USD exposure.
        """
        exposure: dict[str, float] = defaultdict(float)

        for pos in positions:
            event = pos.get("underlying_event") or pos.get("sector", "unknown")
            size = pos.get("size_usd", 0.0)
            direction = pos.get("direction", 1)
            exposure[event] += size * direction

        return dict(exposure)

    def gross_exposure(
        self,
        positions: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Compute gross (unsigned) exposure per event/sector.

        Unlike :meth:`net_exposure`, this sums absolute values, showing
        the total capital at risk regardless of direction.
        """
        exposure: dict[str, float] = defaultdict(float)

        for pos in positions:
            event = pos.get("underlying_event") or pos.get("sector", "unknown")
            size = pos.get("size_usd", 0.0)
            exposure[event] += abs(size)

        return dict(exposure)

    # ------------------------------------------------------------------
    # Risk-adjusted correlation
    # ------------------------------------------------------------------

    def effective_diversification_ratio(
        self,
        positions: list[dict[str, Any]],
    ) -> float:
        """Compute the portfolio diversification ratio.

        DR = sum(w_i * sigma_i) / sigma_portfolio

        A DR of 1.0 means no diversification benefit (perfectly correlated).
        Higher values indicate better diversification.

        Returns 1.0 if correlation matrix is not available.
        """
        matrix, market_ids = self.get_correlation_matrix()
        if matrix is None or len(positions) < 2:
            return 1.0

        # Build weight and volatility vectors for positions in the matrix
        weights = []
        vols = []
        indices = []

        for pos in positions:
            mid = pos.get("market_id", "")
            if mid in market_ids:
                idx = market_ids.index(mid)
                indices.append(idx)
                weights.append(pos.get("size_usd", 0.0))
                vols.append(pos.get("volatility", 0.10))

        if len(indices) < 2:
            return 1.0

        w = np.array(weights)
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        sigma = np.array(vols)

        # Numerator: weighted sum of individual volatilities
        numerator = float(np.sum(w * sigma))

        # Denominator: portfolio volatility
        sub_corr = matrix[np.ix_(indices, indices)]
        cov = np.outer(sigma, sigma) * sub_corr
        port_var = float(w @ cov @ w)
        port_vol = np.sqrt(max(port_var, 1e-12))

        if port_vol <= 0:
            return 1.0

        return float(numerator / port_vol)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Summary of correlation monitor state."""
        matrix, ids = self.get_correlation_matrix()
        correlated_pairs = self.get_correlated_positions(threshold=0.7)

        return {
            "n_markets_tracked": len(self._market_ids),
            "n_markets_in_matrix": len(ids),
            "matrix_shape": matrix.shape if matrix is not None else None,
            "highly_correlated_pairs": len(correlated_pairs),
            "top_correlations": correlated_pairs[:5],
        }
