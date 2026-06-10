"""
APEX Risk Management

Portfolio-level risk management with tiered circuit breakers, Bayesian
position sizing, HMM regime detection, and dynamic capital allocation.

All sub-module imports are lazy to avoid hard dependency on numpy,
scipy, hmmlearn, etc. at package import time.  Import individual
modules directly when needed::

    from apex.risk.circuit_breaker import CircuitBreaker
    from apex.risk.regime_detector import RegimeDetector
"""

__all__ = [
    "BreakerLevel",
    "CapitalAllocator",
    "CircuitBreaker",
    "CorrelationMonitor",
    "PortfolioRiskManager",
    "PositionSizer",
    "RegimeDetector",
]


def __getattr__(name: str):
    """Lazy imports for all risk sub-modules."""
    if name in ("BreakerLevel", "CircuitBreaker"):
        from apex.risk import circuit_breaker as _cb
        return getattr(_cb, name)
    if name == "CapitalAllocator":
        from apex.risk.capital_allocator import CapitalAllocator
        return CapitalAllocator
    if name == "CorrelationMonitor":
        from apex.risk.correlation_monitor import CorrelationMonitor
        return CorrelationMonitor
    if name == "PortfolioRiskManager":
        from apex.risk.portfolio_risk import PortfolioRiskManager
        return PortfolioRiskManager
    if name == "PositionSizer":
        from apex.risk.position_sizer import PositionSizer
        return PositionSizer
    if name == "RegimeDetector":
        from apex.risk.regime_detector import RegimeDetector
        return RegimeDetector
    raise AttributeError(f"module 'apex.risk' has no attribute {name!r}")
