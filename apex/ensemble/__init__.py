"""
APEX Ensemble Pipeline

Stacked generalization with Thompson Sampling exploration and a strict
trade decision gate.

All sub-module imports are lazy to avoid hard dependency on numpy,
lightgbm, scipy, etc. at package import time.  Import individual
modules directly when needed::

    from apex.ensemble.signal import ApexSignal
    from apex.ensemble.meta_learner import MetaLearner
"""

__all__ = [
    "ApexSignal",
    "GateResult",
    "MetaLearner",
    "ThompsonSampler",
    "TradeGate",
]


def __getattr__(name: str):
    """Lazy imports for all ensemble sub-modules."""
    if name == "ApexSignal":
        from apex.ensemble.signal import ApexSignal
        return ApexSignal
    if name == "ThompsonSampler":
        from apex.ensemble.thompson_sampling import ThompsonSampler
        return ThompsonSampler
    if name == "MetaLearner":
        from apex.ensemble.meta_learner import MetaLearner
        return MetaLearner
    if name == "TradeGate":
        from apex.ensemble.trade_gate import TradeGate
        return TradeGate
    if name == "GateResult":
        from apex.ensemble.trade_gate import GateResult
        return GateResult
    raise AttributeError(f"module 'apex.ensemble' has no attribute {name!r}")
