"""
APEX Thompson Sampling for Model Weights

Exploration/exploitation for dynamic model weight selection.  Maintains a
Beta distribution for each model's reliability and samples weights at
decision time, naturally balancing exploitation of known-good models with
exploration of uncertain ones.

The Thompson Sampling approach is particularly useful in APEX because:
1. It adapts to non-stationary model performance (models degrade over time).
2. It automatically down-weights models that stop performing.
3. The exploration term prevents over-concentration on a single model.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ThompsonSampler:
    """Exploration/exploitation for model weight selection.

    Maintains a ``Beta(alpha, beta)`` distribution for each model name,
    representing the posterior belief about that model's probability of
    producing a profitable signal.

    Parameters
    ----------
    model_names : list[str]
        Names of the Level 0 models to track (e.g.
        ``["xgboost", "lgbm", "tft", "sentiment"]``).
    prior_alpha : float
        Initial alpha (successes) for the Beta prior.  Default ``1.0``
        gives a uniform prior.
    prior_beta : float
        Initial beta (failures) for the Beta prior.
    decay : float
        Exponential decay factor applied to alpha and beta at each
        :meth:`decay_params` call.  ``1.0`` means no decay.
        Values like ``0.995`` make the sampler forget old observations
        gradually, keeping it responsive to regime changes.
    min_weight : float
        Minimum weight any model can receive (prevents total exclusion).
    """

    def __init__(
        self,
        model_names: list[str],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        decay: float = 1.0,
        min_weight: float = 0.05,
    ) -> None:
        if not model_names:
            raise ValueError("model_names must not be empty")

        self.model_names = list(model_names)
        self.alphas: dict[str, float] = {name: prior_alpha for name in model_names}
        self.betas: dict[str, float] = {name: prior_beta for name in model_names}
        self.decay = decay
        self.min_weight = min_weight

        # Tracking stats
        self._total_updates: int = 0
        self._success_counts: dict[str, int] = {name: 0 for name in model_names}
        self._failure_counts: dict[str, int] = {name: 0 for name in model_names}

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_weights(self, rng: np.random.Generator | None = None) -> dict[str, float]:
        """Sample normalised weights from Beta distributions.

        Each model's weight is drawn from ``Beta(alpha, beta)`` and then
        the vector is normalised so weights sum to 1.0, with a minimum
        floor of ``self.min_weight`` per model.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        dict[str, float]
            Model name -> weight mapping (sums to 1.0).
        """
        rng = rng or np.random.default_rng()

        raw_weights: dict[str, float] = {}
        for name in self.model_names:
            sample = rng.beta(self.alphas[name], self.betas[name])
            raw_weights[name] = max(sample, self.min_weight)

        # Normalise
        total = sum(raw_weights.values())
        if total <= 0:
            # Fallback: equal weights
            equal = 1.0 / len(self.model_names)
            return {name: equal for name in self.model_names}

        return {name: w / total for name, w in raw_weights.items()}

    def expected_weights(self) -> dict[str, float]:
        """Return the mean of each Beta distribution, normalised.

        Unlike :meth:`sample_weights`, this is deterministic and useful
        for logging or monitoring the current belief state.
        """
        means: dict[str, float] = {}
        for name in self.model_names:
            a, b = self.alphas[name], self.betas[name]
            means[name] = max(a / (a + b), self.min_weight)

        total = sum(means.values())
        return {name: m / total for name, m in means.items()}

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def update(self, model_name: str, correct: bool) -> None:
        """Update Beta parameters after observing a trade outcome.

        Parameters
        ----------
        model_name : str
            Which model's signal was used.
        correct : bool
            Whether the trade was profitable (True) or not (False).

        Raises
        ------
        KeyError
            If ``model_name`` is not in the tracked models.
        """
        if model_name not in self.alphas:
            raise KeyError(
                f"Unknown model '{model_name}'. "
                f"Known models: {self.model_names}"
            )

        if correct:
            self.alphas[model_name] += 1.0
            self._success_counts[model_name] += 1
        else:
            self.betas[model_name] += 1.0
            self._failure_counts[model_name] += 1

        self._total_updates += 1

        logger.debug(
            "Thompson update: %s %s -> Beta(%.1f, %.1f)",
            model_name,
            "success" if correct else "failure",
            self.alphas[model_name],
            self.betas[model_name],
        )

    def update_continuous(self, model_name: str, pnl: float, threshold: float = 0.0) -> None:
        """Update from a continuous PnL outcome.

        Converts a continuous PnL to a binary success/failure signal
        using the provided threshold.

        Parameters
        ----------
        model_name : str
            Which model's signal was used.
        pnl : float
            Realised PnL of the trade.
        threshold : float
            PnL above this value counts as a success.
        """
        self.update(model_name, correct=(pnl > threshold))

    def batch_update(self, results: list[tuple[str, bool]]) -> None:
        """Update multiple model outcomes at once.

        Parameters
        ----------
        results : list of (model_name, correct) tuples.
        """
        for model_name, correct in results:
            self.update(model_name, correct)

    # ------------------------------------------------------------------
    # Decay (for non-stationarity)
    # ------------------------------------------------------------------

    def decay_params(self) -> None:
        """Apply exponential decay to all Beta parameters.

        This makes the sampler gradually forget old observations, keeping
        it responsive to changing model performance.  Call this periodically
        (e.g. daily or after each rebalance).

        The effective sample size decreases by factor ``self.decay`` each
        call, but alpha and beta are floored at 1.0 to maintain a valid
        Beta distribution.
        """
        if self.decay >= 1.0:
            return

        for name in self.model_names:
            self.alphas[name] = max(1.0, self.alphas[name] * self.decay)
            self.betas[name] = max(1.0, self.betas[name] * self.decay)

        logger.debug(
            "Thompson decay applied (factor=%.4f), effective counts reduced",
            self.decay,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def model_reliability(self, model_name: str) -> float:
        """Posterior mean reliability estimate for a model.

        Returns ``alpha / (alpha + beta)`` -- the expected probability
        that this model produces a profitable signal.
        """
        a = self.alphas[model_name]
        b = self.betas[model_name]
        return a / (a + b)

    def model_uncertainty(self, model_name: str) -> float:
        """Posterior standard deviation of reliability estimate.

        Higher values mean we are less certain about the model's quality.
        """
        a = self.alphas[model_name]
        b = self.betas[model_name]
        return float(np.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1))))

    def summary(self) -> dict[str, dict[str, float]]:
        """Return a summary dict for all models.

        Returns
        -------
        dict[str, dict]
            ``{model_name: {alpha, beta, mean, std, successes, failures}}``.
        """
        out: dict[str, dict[str, float]] = {}
        for name in self.model_names:
            a, b = self.alphas[name], self.betas[name]
            out[name] = {
                "alpha": a,
                "beta": b,
                "mean": a / (a + b),
                "std": self.model_uncertainty(name),
                "successes": float(self._success_counts[name]),
                "failures": float(self._failure_counts[name]),
            }
        return out

    @property
    def total_updates(self) -> int:
        """Total number of update calls."""
        return self._total_updates

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Serialize sampler state for persistence."""
        return {
            "model_names": self.model_names,
            "alphas": self.alphas.copy(),
            "betas": self.betas.copy(),
            "decay": self.decay,
            "min_weight": self.min_weight,
            "total_updates": self._total_updates,
            "success_counts": self._success_counts.copy(),
            "failure_counts": self._failure_counts.copy(),
        }

    @classmethod
    def from_state_dict(cls, d: dict[str, Any]) -> ThompsonSampler:
        """Restore sampler from a serialized state dict."""
        instance = cls(
            model_names=d["model_names"],
            decay=d.get("decay", 1.0),
            min_weight=d.get("min_weight", 0.05),
        )
        instance.alphas = d["alphas"]
        instance.betas = d["betas"]
        instance._total_updates = d.get("total_updates", 0)
        instance._success_counts = d.get("success_counts", {n: 0 for n in d["model_names"]})
        instance._failure_counts = d.get("failure_counts", {n: 0 for n in d["model_names"]})
        return instance
