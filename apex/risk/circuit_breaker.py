"""
APEX Tiered Circuit Breaker System

Implements a five-level circuit breaker that progressively restricts trading
activity as drawdown deepens or consecutive losses accumulate.

Levels
------
GREEN  : Normal operation -- all systems go.
YELLOW : 10% DD or 5 consecutive losses -- reduce position sizing by 50%.
ORANGE : 20% DD or 8 consecutive losses -- stop opening new positions.
RED    : 30% DD or single loss > 10% of equity -- close non-hedged positions.
BLACK  : 40% DD -- emergency close everything.

The breaker only *escalates* automatically.  De-escalation requires an
explicit call to :meth:`acknowledge` (manual review) or sustained equity
recovery above the recovery thresholds.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class BreakerLevel(Enum):
    """Circuit breaker severity levels (ordered from least to most severe)."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"
    BLACK = "BLACK"


# Ordered severity for comparison
_SEVERITY: dict[BreakerLevel, int] = {
    BreakerLevel.GREEN: 0,
    BreakerLevel.YELLOW: 1,
    BreakerLevel.ORANGE: 2,
    BreakerLevel.RED: 3,
    BreakerLevel.BLACK: 4,
}

# Drawdown thresholds
_DD_THRESHOLDS: dict[BreakerLevel, float] = {
    BreakerLevel.YELLOW: 0.10,
    BreakerLevel.ORANGE: 0.20,
    BreakerLevel.RED: 0.30,
    BreakerLevel.BLACK: 0.40,
}

# Consecutive loss thresholds
_CONSEC_LOSS_THRESHOLDS: dict[BreakerLevel, int] = {
    BreakerLevel.YELLOW: 5,
    BreakerLevel.ORANGE: 8,
}

# Single-loss threshold (fraction of equity)
_SINGLE_LOSS_RED_THRESHOLD: float = 0.10

# Position sizing multipliers per level
_SIZING_MULTIPLIERS: dict[BreakerLevel, float] = {
    BreakerLevel.GREEN: 1.0,
    BreakerLevel.YELLOW: 0.5,
    BreakerLevel.ORANGE: 0.0,  # No new positions
    BreakerLevel.RED: 0.0,
    BreakerLevel.BLACK: 0.0,
}

# Recovery thresholds: drawdown must recover to this level (from peak) before
# the breaker de-escalates from the corresponding level.
_RECOVERY_THRESHOLDS: dict[BreakerLevel, float] = {
    BreakerLevel.YELLOW: 0.05,
    BreakerLevel.ORANGE: 0.10,
    BreakerLevel.RED: 0.15,
    BreakerLevel.BLACK: 0.20,
}


class CircuitBreaker:
    """Tiered circuit breaker system.

    Parameters
    ----------
    initial_equity : float
        Starting equity.  If zero, the first :meth:`update` call sets it.
    auto_recovery : bool
        If ``True``, the breaker automatically de-escalates when equity
        recovers above recovery thresholds.  If ``False``, de-escalation
        requires :meth:`acknowledge`.
    """

    def __init__(
        self,
        initial_equity: float = 0.0,
        auto_recovery: bool = True,
    ) -> None:
        self.level: BreakerLevel = BreakerLevel.GREEN
        self.consecutive_losses: int = 0
        self.peak_equity: float = initial_equity
        self.current_equity: float = initial_equity
        self.auto_recovery = auto_recovery

        # Audit trail
        self._history: list[dict[str, Any]] = []
        self._last_transition: datetime | None = None
        self._total_trades: int = 0
        self._total_losses: int = 0

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update(
        self,
        equity: float,
        trade_result: float | None = None,
    ) -> BreakerLevel:
        """Update breaker state after an equity change or trade result.

        Parameters
        ----------
        equity : float
            Current total portfolio equity.
        trade_result : float or None
            PnL of the most recent trade (None if this is just an equity
            snapshot with no new trade).

        Returns
        -------
        BreakerLevel
            The (possibly updated) circuit breaker level.
        """
        old_level = self.level
        self.current_equity = equity

        # Initialise peak on first call
        if self.peak_equity <= 0.0:
            self.peak_equity = equity

        # Update peak equity (high-water mark)
        if equity > self.peak_equity:
            self.peak_equity = equity

        # Process trade result
        if trade_result is not None:
            self._total_trades += 1
            if trade_result < 0:
                self.consecutive_losses += 1
                self._total_losses += 1
            else:
                self.consecutive_losses = 0

        # Compute drawdown from peak
        drawdown = self._drawdown()

        # Compute single-trade loss fraction
        single_loss_frac = 0.0
        if trade_result is not None and trade_result < 0 and self.peak_equity > 0:
            single_loss_frac = abs(trade_result) / self.peak_equity

        # Determine level based on conditions (escalation only)
        new_level = BreakerLevel.GREEN

        # Drawdown-based escalation
        for level, threshold in sorted(
            _DD_THRESHOLDS.items(),
            key=lambda x: _SEVERITY[x[0]],
            reverse=True,
        ):
            if drawdown >= threshold:
                new_level = level
                break

        # Consecutive loss escalation (take the more severe)
        for level, threshold in sorted(
            _CONSEC_LOSS_THRESHOLDS.items(),
            key=lambda x: _SEVERITY[x[0]],
            reverse=True,
        ):
            if self.consecutive_losses >= threshold:
                if _SEVERITY[level] > _SEVERITY[new_level]:
                    new_level = level
                break

        # Single catastrophic loss -> RED
        if single_loss_frac >= _SINGLE_LOSS_RED_THRESHOLD:
            if _SEVERITY[BreakerLevel.RED] > _SEVERITY[new_level]:
                new_level = BreakerLevel.RED

        # Circuit breaker can escalate freely but only de-escalate under
        # controlled conditions
        if _SEVERITY[new_level] > _SEVERITY[self.level]:
            # Escalation: always allowed
            self.level = new_level
        elif self.auto_recovery and _SEVERITY[new_level] < _SEVERITY[self.level]:
            # De-escalation: only if drawdown has recovered sufficiently
            recovery_threshold = _RECOVERY_THRESHOLDS.get(self.level, 0.0)
            if drawdown <= recovery_threshold:
                self.level = new_level

        # Log transition
        if self.level != old_level:
            self._last_transition = datetime.now(timezone.utc)
            transition = {
                "timestamp": self._last_transition.isoformat(),
                "from": old_level.value,
                "to": self.level.value,
                "drawdown": drawdown,
                "consecutive_losses": self.consecutive_losses,
                "equity": equity,
                "peak": self.peak_equity,
            }
            self._history.append(transition)
            logger.warning(
                "Circuit breaker %s -> %s (DD=%.2f%%, consec_losses=%d, equity=%.2f)",
                old_level.value,
                self.level.value,
                drawdown * 100,
                self.consecutive_losses,
                equity,
            )

        return self.level

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------

    def can_open_new_position(self) -> bool:
        """True if the current level allows opening new positions."""
        return self.level in (BreakerLevel.GREEN, BreakerLevel.YELLOW)

    def sizing_multiplier(self) -> float:
        """Position sizing multiplier for the current level.

        GREEN: 1.0, YELLOW: 0.5, others: 0.0.
        """
        return _SIZING_MULTIPLIERS[self.level]

    def should_close_non_hedged(self) -> bool:
        """True if the breaker mandates closing non-hedged positions."""
        return _SEVERITY[self.level] >= _SEVERITY[BreakerLevel.RED]

    def should_close_all(self) -> bool:
        """True if the breaker mandates emergency liquidation."""
        return self.level == BreakerLevel.BLACK

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------

    def _drawdown(self) -> float:
        """Current drawdown from peak as a fraction (0.0 = no drawdown)."""
        if self.peak_equity <= 0.0:
            return 0.0
        return max(0.0, (self.peak_equity - self.current_equity) / self.peak_equity)

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown as a percentage."""
        return self._drawdown() * 100.0

    # ------------------------------------------------------------------
    # Manual controls
    # ------------------------------------------------------------------

    def acknowledge(self, new_level: BreakerLevel | None = None) -> None:
        """Manually acknowledge and optionally de-escalate the breaker.

        This is intended for human operator intervention.  If ``new_level``
        is not provided, the breaker drops one level.

        Parameters
        ----------
        new_level : BreakerLevel or None
            Target level.  Must be less severe than current.
        """
        if new_level is None:
            # Drop one level
            current_sev = _SEVERITY[self.level]
            if current_sev == 0:
                return  # Already GREEN
            for lvl, sev in _SEVERITY.items():
                if sev == current_sev - 1:
                    new_level = lvl
                    break

        assert new_level is not None

        if _SEVERITY[new_level] >= _SEVERITY[self.level]:
            logger.warning(
                "Acknowledge ignored: target %s is not less severe than current %s",
                new_level.value,
                self.level.value,
            )
            return

        old = self.level
        self.level = new_level
        self._last_transition = datetime.now(timezone.utc)
        self._history.append({
            "timestamp": self._last_transition.isoformat(),
            "from": old.value,
            "to": new_level.value,
            "type": "manual_acknowledge",
        })
        logger.info("Circuit breaker manually set: %s -> %s", old.value, new_level.value)

    def force_black(self) -> None:
        """Emergency: force BLACK level regardless of metrics."""
        old = self.level
        self.level = BreakerLevel.BLACK
        self._last_transition = datetime.now(timezone.utc)
        self._history.append({
            "timestamp": self._last_transition.isoformat(),
            "from": old.value,
            "to": "BLACK",
            "type": "force_black",
        })
        logger.critical("Circuit breaker FORCE BLACK activated from %s", old.value)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[dict[str, Any]]:
        """Ordered list of breaker level transitions."""
        return list(self._history)

    def state_dict(self) -> dict[str, Any]:
        """Serialize breaker state for persistence."""
        return {
            "level": self.level.value,
            "consecutive_losses": self.consecutive_losses,
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "auto_recovery": self.auto_recovery,
            "total_trades": self._total_trades,
            "total_losses": self._total_losses,
            "history": self._history,
        }

    @classmethod
    def from_state_dict(cls, d: dict[str, Any]) -> CircuitBreaker:
        """Restore breaker from a serialized state dict."""
        instance = cls(
            initial_equity=d.get("current_equity", 0.0),
            auto_recovery=d.get("auto_recovery", True),
        )
        instance.level = BreakerLevel(d["level"])
        instance.consecutive_losses = d.get("consecutive_losses", 0)
        instance.peak_equity = d.get("peak_equity", 0.0)
        instance._total_trades = d.get("total_trades", 0)
        instance._total_losses = d.get("total_losses", 0)
        instance._history = d.get("history", [])
        return instance

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(level={self.level.value}, "
            f"DD={self.drawdown_pct:.1f}%, "
            f"consec_losses={self.consecutive_losses}, "
            f"equity={self.current_equity:.2f})"
        )
