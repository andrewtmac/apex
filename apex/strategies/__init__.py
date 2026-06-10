"""
APEX Trading Strategies

Multi-strategy ensemble for prediction market and options trading.

Strategies:
    S1  BayesianForecaster       Bayesian event probability estimation
    S2  Microstructure           VPIN-based informed flow detection
    S3  InfoArb                  Cross-market information arbitrage
    S4  CalibrationExploit       Systematic miscalibration exploitation
    S5  Convergence              Resolution convergence trading
    S6  SmartMarketMaking        Avellaneda-Stoikov market making
    S7  VolSurface               IV surface arbitrage (TastyTrade)
    S8  Earnings                 ML earnings + options (TastyTrade)
    S12 RegimeSelector           Regime-adaptive strategy allocation
    S13 ExecutionOptimizer       Multi-venue execution routing

Modules are imported lazily to avoid hard dependency on nautilus_trader
and ML libraries at package import time.
"""

__all__ = [
    "ApexStrategy",
    "ApexStrategyConfig",
    "BayesianForecasterStrategy",
    "CalibrationExploitStrategy",
    "ConvergenceStrategy",
    "EarningsStrategy",
    "ExecutionOptimizer",
    "InfoArbStrategy",
    "MicrostructureStrategy",
    "RegimeSelectorStrategy",
    "SmartMarketMakingStrategy",
    "VolSurfaceStrategy",
]


def __getattr__(name: str):
    """Lazy imports for strategy classes (depend on nautilus_trader)."""
    _strategy_map = {
        "ApexStrategy": ("apex.strategies.apex_strategy", "ApexStrategy"),
        "ApexStrategyConfig": ("apex.strategies.apex_strategy", "ApexStrategyConfig"),
        "BayesianForecasterStrategy": ("apex.strategies.bayesian_forecaster", "BayesianForecasterStrategy"),
        "CalibrationExploitStrategy": ("apex.strategies.calibration_exploit", "CalibrationExploitStrategy"),
        "ConvergenceStrategy": ("apex.strategies.convergence", "ConvergenceStrategy"),
        "EarningsStrategy": ("apex.strategies.earnings", "EarningsStrategy"),
        "InfoArbStrategy": ("apex.strategies.info_arb", "InfoArbStrategy"),
        "MicrostructureStrategy": ("apex.strategies.microstructure", "MicrostructureStrategy"),
        "RegimeSelectorStrategy": ("apex.strategies.regime_selector", "RegimeSelectorStrategy"),
        "SmartMarketMakingStrategy": ("apex.strategies.smart_mm", "SmartMarketMakingStrategy"),
        "VolSurfaceStrategy": ("apex.strategies.vol_surface", "VolSurfaceStrategy"),
    }

    if name in _strategy_map:
        module_path, cls_name = _strategy_map[name]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)

    raise AttributeError(f"module 'apex.strategies' has no attribute {name!r}")
