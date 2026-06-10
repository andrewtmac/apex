"""
APEX Trade Decision Gate

All conditions must pass for a trade to execute.  The gate is the final
checkpoint before an order is submitted, enforcing edge requirements,
ensemble confidence, regime constraints, spread limits, position manager
output validation, portfolio CVaR limits, and circuit breaker status.

Each condition returns a named result so rejected trades have an auditable
reason for post-mortem analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from apex.config import ApexConfig
from apex.ensemble.signal import ApexSignal

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of a gate evaluation.

    Attributes
    ----------
    approved : bool
        Whether the trade passed all gate conditions.
    reason : str
        Human-readable reason if rejected, or ``"ALL_CHECKS_PASSED"`` if approved.
    checks : dict[str, bool]
        Individual check results for audit trail.
    details : dict[str, str]
        Per-check detail messages explaining pass/fail.
    """

    approved: bool
    reason: str
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)


class TradeGate:
    """All conditions must pass for a trade to execute.

    The gate is designed to be conservative: if any single condition fails,
    the entire trade is rejected.  This is intentional -- it is cheaper to
    miss a marginal opportunity than to enter a bad trade.

    Parameters
    ----------
    config : ApexConfig
        Master configuration (used for risk regime lookups).
    min_edge : float
        Minimum calibrated edge (lower CI bound must exceed this).
    min_ensemble_score : float
        Minimum meta-learner ensemble score.
    max_spread_bps : float
        Maximum bid-ask spread in basis points.
    min_regime_confidence : float
        Minimum regime classification confidence.
    max_portfolio_cvar_pct : float
        Maximum portfolio CVaR as a percentage of total capital.
    max_correlation : float
        Maximum correlation with existing positions.
    blocked_regimes : list[str]
        Regimes where new entries are forbidden.
    """

    def __init__(
        self,
        config: ApexConfig | None = None,
        min_edge: float = 0.02,
        min_ensemble_score: float = 0.6,
        max_spread_bps: float = 500.0,
        min_regime_confidence: float = 0.3,
        max_portfolio_cvar_pct: float = 0.15,
        max_correlation: float = 0.85,
        blocked_regimes: list[str] | None = None,
    ) -> None:
        self.config = config
        self.min_edge = min_edge
        self.min_ensemble_score = min_ensemble_score
        self.max_spread_bps = max_spread_bps
        self.min_regime_confidence = min_regime_confidence
        self.max_portfolio_cvar_pct = max_portfolio_cvar_pct
        self.max_correlation = max_correlation
        self.blocked_regimes = blocked_regimes or ["CRISIS"]

        self._rejection_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        signal: ApexSignal,
        portfolio_state: dict[str, Any],
    ) -> tuple[bool, str]:
        """Evaluate all gate conditions.

        Parameters
        ----------
        signal : ApexSignal
            The ensemble signal to evaluate.
        portfolio_state : dict
            Current portfolio state with keys:

            - ``"circuit_breaker_level"`` : str -- current breaker level
            - ``"portfolio_cvar"`` : float -- current portfolio CVaR (fraction)
            - ``"total_capital"`` : float -- total available capital
            - ``"open_positions"`` : int -- number of open positions
            - ``"max_positions"`` : int -- maximum allowed positions (optional)
            - ``"venue_capital_available"`` : float -- capital available for this venue
            - ``"correlated_exposure"`` : float -- max correlation with existing positions (optional)

        Returns
        -------
        tuple[bool, str]
            ``(approved, reason)`` -- approved is True if all checks pass,
            reason describes the first failing check or ``"ALL_CHECKS_PASSED"``.
        """
        result = self.evaluate_detailed(signal, portfolio_state)
        return result.approved, result.reason

    def evaluate_detailed(
        self,
        signal: ApexSignal,
        portfolio_state: dict[str, Any],
    ) -> GateResult:
        """Evaluate all gate conditions with full audit trail.

        Same as :meth:`evaluate` but returns a :class:`GateResult` with
        individual check results.
        """
        checks: dict[str, bool] = {}
        details: dict[str, str] = {}

        # 1. Edge confidence interval lower bound > min_edge
        edge_ok = signal.edge_ci_lower > self.min_edge
        checks["edge_ci_lower"] = edge_ok
        details["edge_ci_lower"] = (
            f"edge_ci_lower={signal.edge_ci_lower:.4f} vs min={self.min_edge:.4f}"
        )

        # 2. Ensemble score > confidence threshold
        ensemble_ok = signal.ensemble_score > self.min_ensemble_score
        checks["ensemble_score"] = ensemble_ok
        details["ensemble_score"] = (
            f"ensemble_score={signal.ensemble_score:.4f} vs min={self.min_ensemble_score:.4f}"
        )

        # 3. Regime is not blocked
        regime_ok = signal.regime not in self.blocked_regimes
        checks["regime_allowed"] = regime_ok
        details["regime_allowed"] = (
            f"regime={signal.regime}, blocked={self.blocked_regimes}"
        )

        # 4. Regime confidence above minimum
        regime_conf_ok = signal.regime_confidence >= self.min_regime_confidence
        checks["regime_confidence"] = regime_conf_ok
        details["regime_confidence"] = (
            f"regime_confidence={signal.regime_confidence:.4f} vs min={self.min_regime_confidence:.4f}"
        )

        # 5. Spread < max_spread
        spread_ok = signal.spread_bps < self.max_spread_bps
        checks["spread_bps"] = spread_ok
        details["spread_bps"] = (
            f"spread_bps={signal.spread_bps:.1f} vs max={self.max_spread_bps:.1f}"
        )

        # 6. PPO outputs non-zero action
        ppo_ok = signal.recommended_size > 0.0 and signal.recommended_action != "HOLD"
        checks["ppo_action"] = ppo_ok
        details["ppo_action"] = (
            f"action={signal.recommended_action}, size={signal.recommended_size:.4f}"
        )

        # 7. Portfolio CVaR check
        current_cvar = portfolio_state.get("portfolio_cvar", 0.0)
        marginal = signal.marginal_cvar
        projected_cvar = current_cvar + marginal
        cvar_ok = projected_cvar < self.max_portfolio_cvar_pct
        checks["portfolio_cvar"] = cvar_ok
        details["portfolio_cvar"] = (
            f"projected_cvar={projected_cvar:.4f} (current={current_cvar:.4f} + "
            f"marginal={marginal:.4f}) vs max={self.max_portfolio_cvar_pct:.4f}"
        )

        # 8. Circuit breaker is GREEN or YELLOW
        breaker_level = portfolio_state.get("circuit_breaker_level", "GREEN")
        breaker_ok = breaker_level in ("GREEN", "YELLOW")
        checks["circuit_breaker"] = breaker_ok
        details["circuit_breaker"] = f"level={breaker_level}"

        # 9. Position count limit
        max_positions = portfolio_state.get("max_positions", 50)
        open_positions = portfolio_state.get("open_positions", 0)
        position_limit_ok = open_positions < max_positions
        checks["position_limit"] = position_limit_ok
        details["position_limit"] = (
            f"open={open_positions} vs max={max_positions}"
        )

        # 10. Venue capital available
        venue_capital = portfolio_state.get("venue_capital_available", float("inf"))
        capital_ok = signal.position_size_usd <= venue_capital
        checks["venue_capital"] = capital_ok
        details["venue_capital"] = (
            f"size_usd={signal.position_size_usd:.2f} vs available={venue_capital:.2f}"
        )

        # 11. Correlation check (if provided)
        corr_exposure = portfolio_state.get("correlated_exposure", 0.0)
        corr_ok = corr_exposure < self.max_correlation
        checks["correlation"] = corr_ok
        details["correlation"] = (
            f"max_corr={corr_exposure:.4f} vs threshold={self.max_correlation:.4f}"
        )

        # Aggregate
        all_passed = all(checks.values())

        if all_passed:
            reason = "ALL_CHECKS_PASSED"
        else:
            # Find first failing check
            failed = [name for name, passed in checks.items() if not passed]
            reason = f"REJECTED: {failed[0]} -- {details[failed[0]]}"
            # Track rejection reasons
            for f in failed:
                self._rejection_counts[f] = self._rejection_counts.get(f, 0) + 1

        if not all_passed:
            logger.info(
                "TradeGate REJECTED %s/%s: %s",
                signal.market_id,
                signal.strategy,
                reason,
            )
        else:
            logger.debug(
                "TradeGate APPROVED %s/%s",
                signal.market_id,
                signal.strategy,
            )

        return GateResult(
            approved=all_passed,
            reason=reason,
            checks=checks,
            details=details,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def rejection_stats(self) -> dict[str, int]:
        """Cumulative rejection counts by check name."""
        return self._rejection_counts.copy()

    def reset_stats(self) -> None:
        """Reset rejection counters."""
        self._rejection_counts.clear()
