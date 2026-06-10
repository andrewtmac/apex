"""
APEX Portfolio-Level Risk Management

Portfolio-level risk using Value-at-Risk (VaR), Conditional VaR (CVaR / Expected
Shortfall), and marginal contribution analysis.  Replaces simple drawdown
percentage thresholds with a proper risk decomposition framework.

Supports:
- Parametric VaR (Gaussian assumption)
- Historical VaR (empirical quantile)
- Monte Carlo CVaR (simulation-based expected shortfall)
- Marginal CVaR (how adding a position changes portfolio risk)
- Concentration and correlation-based risk limits
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from apex.config import RiskConfig
from apex.ensemble.signal import ApexSignal

logger = logging.getLogger(__name__)


class PortfolioRiskManager:
    """Portfolio-level risk using CVaR and Expected Shortfall.

    Parameters
    ----------
    config : RiskConfig
        Risk configuration with per-regime parameters.
    max_portfolio_cvar : float
        Maximum allowed portfolio CVaR as a fraction of total capital
        (default 15%).  If adding a new position would push CVaR above
        this, the position is rejected.
    max_single_position_pct : float
        Maximum single position as a fraction of total capital.
    max_sector_concentration : float
        Maximum capital deployed in a single sector / event category.
    max_venue_concentration : float
        Maximum capital deployed on a single venue.
    """

    def __init__(
        self,
        config: RiskConfig,
        max_portfolio_cvar: float = 0.15,
        max_single_position_pct: float = 0.12,
        max_sector_concentration: float = 0.30,
        max_venue_concentration: float = 0.50,
    ) -> None:
        self.config = config
        self.max_portfolio_cvar = max_portfolio_cvar
        self.max_single_position_pct = max_single_position_pct
        self.max_sector_concentration = max_sector_concentration
        self.max_venue_concentration = max_venue_concentration

        self.positions: list[dict[str, Any]] = []
        self._returns_history: list[np.ndarray] = []
        self._total_capital: float = 0.0

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def set_capital(self, total_capital: float) -> None:
        """Set the total portfolio capital."""
        self._total_capital = total_capital

    def add_position(self, position: dict[str, Any]) -> None:
        """Register a position in the portfolio.

        Parameters
        ----------
        position : dict
            Must contain:

            - ``"market_id"`` : str
            - ``"venue"`` : str
            - ``"size_usd"`` : float -- notional value
            - ``"direction"`` : int -- +1 or -1
            - ``"expected_return"`` : float -- expected return of the position
            - ``"volatility"`` : float -- estimated annualised volatility
            - ``"sector"`` : str -- event category (optional)
            - ``"entry_price"`` : float (optional)
        """
        self.positions.append(position)

    def remove_position(self, market_id: str) -> None:
        """Remove a position by market_id."""
        self.positions = [p for p in self.positions if p.get("market_id") != market_id]

    def clear_positions(self) -> None:
        """Remove all positions."""
        self.positions.clear()

    def record_returns(self, returns: np.ndarray) -> None:
        """Record a portfolio return observation for historical VaR.

        Parameters
        ----------
        returns : np.ndarray
            Per-position returns vector for a single time step.
        """
        self._returns_history.append(returns.copy())

    # ------------------------------------------------------------------
    # VaR computation
    # ------------------------------------------------------------------

    def compute_var(self, confidence: float = 0.95) -> float:
        """Parametric Value-at-Risk assuming Gaussian returns.

        Parameters
        ----------
        confidence : float
            Confidence level (e.g. 0.95 for 95% VaR).

        Returns
        -------
        float
            VaR as a positive number (maximum expected loss at the given
            confidence level), expressed as a fraction of portfolio value.
        """
        if not self.positions:
            return 0.0

        # Extract position data
        sizes = np.array([p.get("size_usd", 0.0) for p in self.positions])
        vols = np.array([p.get("volatility", 0.10) for p in self.positions])
        directions = np.array([p.get("direction", 1) for p in self.positions])

        total_notional = sizes.sum()
        if total_notional <= 0:
            return 0.0

        # Weights (signed by direction)
        weights = (sizes * directions) / total_notional

        # Portfolio volatility (assuming uncorrelated for parametric VaR)
        # For correlated positions, use compute_cvar with Monte Carlo
        port_vol = float(np.sqrt(np.sum((weights * vols) ** 2)))

        # Daily vol (annualised to daily: / sqrt(252))
        daily_vol = port_vol / np.sqrt(252)

        # VaR = z_alpha * sigma * portfolio_value
        z_alpha = sp_stats.norm.ppf(confidence)
        var_frac = z_alpha * daily_vol

        return float(max(0.0, var_frac))

    def compute_historical_var(self, confidence: float = 0.95) -> float:
        """Historical VaR from recorded return observations.

        Returns
        -------
        float
            Empirical VaR as a positive fraction.
        """
        if len(self._returns_history) < 10:
            return self.compute_var(confidence)

        # Stack returns and compute portfolio returns
        returns_matrix = np.array(self._returns_history)

        # Portfolio return = sum of position returns (weighted)
        sizes = np.array([p.get("size_usd", 0.0) for p in self.positions])
        total = sizes.sum() or 1.0
        weights = sizes / total

        # Trim matrix to match current number of positions
        n_positions = min(returns_matrix.shape[1], len(weights))
        portfolio_returns = returns_matrix[:, :n_positions] @ weights[:n_positions]

        # VaR = negative quantile of portfolio returns
        var = -float(np.percentile(portfolio_returns, (1.0 - confidence) * 100))
        return max(0.0, var)

    # ------------------------------------------------------------------
    # CVaR (Expected Shortfall)
    # ------------------------------------------------------------------

    def compute_cvar(
        self,
        confidence: float = 0.95,
        n_simulations: int = 10_000,
        correlation_matrix: np.ndarray | None = None,
    ) -> float:
        """Conditional VaR via Monte Carlo simulation.

        CVaR (Expected Shortfall) is the expected loss given that the loss
        exceeds the VaR threshold.  It is more sensitive to tail risk than
        VaR alone.

        Parameters
        ----------
        confidence : float
            Confidence level (e.g. 0.95).
        n_simulations : int
            Number of Monte Carlo scenarios.
        correlation_matrix : np.ndarray or None
            Position correlation matrix.  If None, positions are assumed
            independent.

        Returns
        -------
        float
            CVaR as a positive fraction of portfolio value.
        """
        if not self.positions:
            return 0.0

        n_pos = len(self.positions)
        sizes = np.array([p.get("size_usd", 0.0) for p in self.positions])
        vols = np.array([p.get("volatility", 0.10) for p in self.positions])
        expected_returns = np.array(
            [p.get("expected_return", 0.0) for p in self.positions]
        )
        directions = np.array([p.get("direction", 1) for p in self.positions])

        total_notional = sizes.sum()
        if total_notional <= 0:
            return 0.0

        weights = (sizes * directions) / total_notional

        # Daily vol
        daily_vols = vols / np.sqrt(252)
        daily_returns = expected_returns / 252

        # Build covariance matrix
        if correlation_matrix is not None and correlation_matrix.shape == (n_pos, n_pos):
            cov = np.outer(daily_vols, daily_vols) * correlation_matrix
        else:
            # Independent positions
            cov = np.diag(daily_vols ** 2)

        # Simulate
        rng = np.random.default_rng(42)
        try:
            simulated = rng.multivariate_normal(daily_returns, cov, size=n_simulations)
        except np.linalg.LinAlgError:
            # Fallback if covariance is not positive semi-definite
            cov_reg = cov + np.eye(n_pos) * 1e-6
            simulated = rng.multivariate_normal(daily_returns, cov_reg, size=n_simulations)

        # Portfolio returns
        port_returns = simulated @ weights

        # VaR threshold
        var_threshold = np.percentile(port_returns, (1.0 - confidence) * 100)

        # CVaR = mean of returns below VaR threshold
        tail_returns = port_returns[port_returns <= var_threshold]

        if len(tail_returns) == 0:
            return 0.0

        cvar = -float(np.mean(tail_returns))
        return max(0.0, cvar)

    # ------------------------------------------------------------------
    # Marginal CVaR
    # ------------------------------------------------------------------

    def marginal_cvar(
        self,
        new_position: dict[str, Any],
        confidence: float = 0.95,
        n_simulations: int = 10_000,
    ) -> float:
        """Compute how much adding a new position increases portfolio CVaR.

        Parameters
        ----------
        new_position : dict
            Position dict with same keys as :meth:`add_position`.
        confidence : float
            Confidence level.
        n_simulations : int
            Number of Monte Carlo scenarios.

        Returns
        -------
        float
            Marginal CVaR contribution (positive means more risk).
        """
        # CVaR before
        cvar_before = self.compute_cvar(confidence, n_simulations)

        # CVaR after (temporarily add position)
        self.positions.append(new_position)
        cvar_after = self.compute_cvar(confidence, n_simulations)
        self.positions.pop()

        marginal = cvar_after - cvar_before
        return float(marginal)

    # ------------------------------------------------------------------
    # Risk limit checks
    # ------------------------------------------------------------------

    def check_risk_limits(self, signal: ApexSignal) -> tuple[bool, str]:
        """Check all portfolio-level risk constraints.

        Parameters
        ----------
        signal : ApexSignal
            Signal with ``position_size_usd``, ``venue``, ``market_id``.

        Returns
        -------
        tuple[bool, str]
            ``(ok, reason)`` -- ok is True if all limits pass.
        """
        if self._total_capital <= 0:
            return False, "Total capital not set"

        size_usd = signal.position_size_usd

        # 1. Single position size limit
        if size_usd / self._total_capital > self.max_single_position_pct:
            return (
                False,
                f"Position size {size_usd:.2f} exceeds "
                f"{self.max_single_position_pct:.0%} of capital",
            )

        # 2. Portfolio CVaR check
        new_pos = {
            "market_id": signal.market_id,
            "venue": signal.venue,
            "size_usd": size_usd,
            "direction": signal.direction,
            "expected_return": signal.calibrated_edge,
            "volatility": signal.edge_ci_upper - signal.edge_ci_lower,
        }
        cvar = self.compute_cvar()
        marginal = self.marginal_cvar(new_pos)
        projected = cvar + marginal
        if projected > self.max_portfolio_cvar:
            return (
                False,
                f"Projected CVaR {projected:.4f} exceeds max {self.max_portfolio_cvar:.4f}",
            )

        # 3. Venue concentration
        venue_exposure = sum(
            p.get("size_usd", 0.0)
            for p in self.positions
            if p.get("venue") == signal.venue
        )
        venue_pct = (venue_exposure + size_usd) / self._total_capital
        if venue_pct > self.max_venue_concentration:
            return (
                False,
                f"Venue {signal.venue} concentration {venue_pct:.2%} exceeds "
                f"max {self.max_venue_concentration:.0%}",
            )

        # 4. Total deployed capital
        total_deployed = sum(p.get("size_usd", 0.0) for p in self.positions)
        deployed_pct = (total_deployed + size_usd) / self._total_capital
        if deployed_pct > 0.85:
            return False, f"Total deployed {deployed_pct:.2%} would exceed 85%"

        return True, "ALL_RISK_CHECKS_PASSED"

    # ------------------------------------------------------------------
    # Portfolio summary
    # ------------------------------------------------------------------

    def portfolio_summary(self) -> dict[str, Any]:
        """Return a summary of current portfolio risk metrics."""
        total_deployed = sum(p.get("size_usd", 0.0) for p in self.positions)

        # Venue breakdown
        venue_exposure: dict[str, float] = {}
        for p in self.positions:
            venue = p.get("venue", "unknown")
            venue_exposure[venue] = venue_exposure.get(venue, 0.0) + p.get("size_usd", 0.0)

        return {
            "n_positions": len(self.positions),
            "total_deployed_usd": total_deployed,
            "deployed_pct": total_deployed / self._total_capital if self._total_capital > 0 else 0.0,
            "var_95": self.compute_var(0.95),
            "cvar_95": self.compute_cvar(0.95, n_simulations=5000),
            "venue_exposure": venue_exposure,
            "total_capital": self._total_capital,
        }
