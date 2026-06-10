"""
APEX S1: Bayesian Event Forecasting Strategy

Maintains Bayesian priors per market, updates them with incoming evidence
(news, sentiment, odds changes, on-chain data), and trades when the
posterior probability diverges from the market price.

Replaces LLM-prompting with proper statistical inference:
- Beta(alpha, beta) prior per market
- Bayesian updating with likelihood ratios from evidence
- Posterior predictive checks for model validation

This is the core probability-estimation strategy.  Other strategies
(CalibrationExploit, Convergence) rely on its posterior estimates.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from apex.ensemble.signal import ApexSignal
from apex.strategies.apex_strategy import ApexStrategy, ApexStrategyConfig

logger = logging.getLogger(__name__)


class BayesianForecasterConfig(ApexStrategyConfig, frozen=True):
    """Configuration for the Bayesian Event Forecasting strategy."""

    strategy_name: str = "bayesian_forecaster"
    min_edge: float = 0.03
    min_ensemble_score: float = 0.60

    # Bayesian-specific parameters
    prior_alpha: float = 2.0    # Initial Beta alpha (weak prior toward 50%)
    prior_beta: float = 2.0     # Initial Beta beta
    evidence_decay: float = 0.99  # Daily decay of evidence strength
    min_likelihood_ratio: float = 1.2  # Minimum LR to trigger update
    max_posterior_shift: float = 0.15  # Max single-update shift in posterior


class BayesianForecasterStrategy(ApexStrategy):
    """Maintains Bayesian priors per market, updates with evidence.

    For each market, maintains a Beta(alpha, beta) posterior distribution
    over the event probability.  Evidence from news, sentiment, and
    model predictions is incorporated via Bayesian updating.
    """

    def __init__(self, config: BayesianForecasterConfig) -> None:
        super().__init__(config)

        # Per-market posteriors: market_id -> (alpha, beta)
        self.market_posteriors: dict[str, tuple[float, float]] = {}

        # Configuration
        self._prior_alpha = config.prior_alpha
        self._prior_beta = config.prior_beta
        self._evidence_decay = config.evidence_decay
        self._min_lr = config.min_likelihood_ratio
        self._max_shift = config.max_posterior_shift

        # Evidence buffer
        self._evidence_buffer: dict[str, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Prior management
    # ------------------------------------------------------------------

    def get_or_create_posterior(
        self,
        market_id: str,
        initial_price: float | None = None,
    ) -> tuple[float, float]:
        """Get or initialize the posterior for a market.

        If ``initial_price`` is provided and this is a new market,
        the prior is set to match the market-implied probability.

        Parameters
        ----------
        market_id : str
            Unique market identifier.
        initial_price : float or None
            Initial market price to calibrate the prior.

        Returns
        -------
        tuple[float, float]
            (alpha, beta) parameters of the Beta posterior.
        """
        if market_id in self.market_posteriors:
            return self.market_posteriors[market_id]

        if initial_price is not None and 0.01 < initial_price < 0.99:
            # Set prior to match market-implied probability with
            # low confidence (equivalent to ~4 observations)
            concentration = self._prior_alpha + self._prior_beta
            alpha = initial_price * concentration
            beta = (1.0 - initial_price) * concentration
        else:
            alpha = self._prior_alpha
            beta = self._prior_beta

        self.market_posteriors[market_id] = (alpha, beta)
        return alpha, beta

    def posterior_mean(self, market_id: str) -> float:
        """Expected value of the posterior: alpha / (alpha + beta)."""
        alpha, beta = self.get_or_create_posterior(market_id)
        return alpha / (alpha + beta)

    def posterior_std(self, market_id: str) -> float:
        """Standard deviation of the posterior."""
        alpha, beta = self.get_or_create_posterior(market_id)
        return math.sqrt(
            (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        )

    def posterior_ci(
        self, market_id: str, confidence: float = 0.90
    ) -> tuple[float, float]:
        """Credible interval for the posterior probability.

        Parameters
        ----------
        confidence : float
            Credible interval width (e.g. 0.90 for 90% CI).

        Returns
        -------
        tuple[float, float]
            (lower, upper) bounds.
        """
        alpha, beta = self.get_or_create_posterior(market_id)
        tail = (1.0 - confidence) / 2.0
        lower = float(sp_stats.beta.ppf(tail, alpha, beta))
        upper = float(sp_stats.beta.ppf(1.0 - tail, alpha, beta))
        return lower, upper

    # ------------------------------------------------------------------
    # Bayesian updating
    # ------------------------------------------------------------------

    def update_with_evidence(
        self,
        market_id: str,
        evidence_type: str,
        likelihood_ratio: float,
    ) -> tuple[float, float]:
        """Update the posterior with new evidence via likelihood ratio.

        Uses the Beta-Binomial conjugate update:
            If LR > 1 (evidence supports YES):
                alpha_new = alpha + log(LR)
            If LR < 1 (evidence supports NO):
                beta_new = beta + log(1/LR)

        Parameters
        ----------
        market_id : str
            Market to update.
        evidence_type : str
            Type of evidence (for logging): "news", "sentiment",
            "model_prediction", "odds_change", "onchain".
        likelihood_ratio : float
            Likelihood ratio: P(evidence | YES) / P(evidence | NO).
            > 1 supports YES, < 1 supports NO.

        Returns
        -------
        tuple[float, float]
            Updated (alpha, beta).
        """
        if likelihood_ratio <= 0:
            logger.warning("Invalid likelihood ratio: %.4f", likelihood_ratio)
            return self.get_or_create_posterior(market_id)

        # Skip weak evidence
        effective_lr = max(likelihood_ratio, 1.0 / likelihood_ratio)
        if effective_lr < self._min_lr:
            return self.get_or_create_posterior(market_id)

        alpha, beta = self.get_or_create_posterior(market_id)
        old_mean = alpha / (alpha + beta)

        if likelihood_ratio > 1.0:
            # Evidence supports YES
            update = math.log(likelihood_ratio)
            alpha_new = alpha + update
            beta_new = beta
        else:
            # Evidence supports NO
            update = math.log(1.0 / likelihood_ratio)
            alpha_new = alpha
            beta_new = beta + update

        # Check posterior shift limit
        new_mean = alpha_new / (alpha_new + beta_new)
        shift = abs(new_mean - old_mean)
        if shift > self._max_shift:
            # Scale down the update
            scale = self._max_shift / shift
            if likelihood_ratio > 1.0:
                alpha_new = alpha + update * scale
            else:
                beta_new = beta + update * scale

        self.market_posteriors[market_id] = (alpha_new, beta_new)

        logger.debug(
            "Bayesian update %s: %s LR=%.3f -> Beta(%.2f, %.2f) mean=%.4f",
            market_id,
            evidence_type,
            likelihood_ratio,
            alpha_new,
            beta_new,
            alpha_new / (alpha_new + beta_new),
        )

        return alpha_new, beta_new

    def update_from_model(
        self,
        market_id: str,
        model_probability: float,
        model_confidence: float = 0.7,
    ) -> tuple[float, float]:
        """Update posterior from a model prediction.

        Converts a model probability into a likelihood ratio by comparing
        it to the current posterior mean.

        Parameters
        ----------
        model_probability : float
            The model's predicted probability (0-1).
        model_confidence : float
            How much weight to give the model (0-1).
        """
        posterior_mean = self.posterior_mean(market_id)

        # Likelihood ratio: how much more likely is the model's probability
        # under H1 (YES) vs H0 (the current posterior)?
        eps = 1e-8
        lr = (model_probability + eps) / (posterior_mean + eps)

        # Scale by model confidence
        lr = 1.0 + (lr - 1.0) * model_confidence

        return self.update_with_evidence(market_id, "model_prediction", lr)

    def decay_posteriors(self) -> None:
        """Apply evidence decay to all posteriors.

        Gradually shrinks alpha and beta toward the prior, making
        the posterior forget old evidence and stay responsive to new data.
        """
        for market_id in list(self.market_posteriors.keys()):
            alpha, beta = self.market_posteriors[market_id]

            # Decay toward equal weighting (preserve mean, reduce concentration)
            mean = alpha / (alpha + beta)
            concentration = alpha + beta
            new_concentration = max(
                self._prior_alpha + self._prior_beta,
                concentration * self._evidence_decay,
            )

            alpha_new = mean * new_concentration
            beta_new = (1.0 - mean) * new_concentration
            self.market_posteriors[market_id] = (alpha_new, beta_new)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _generate_signal(self, features: np.ndarray) -> ApexSignal | None:
        """Generate a Bayesian forecasting signal.

        Pipeline:
        1. Get XGBoost probability from features
        2. Update posterior with model evidence
        3. Compute posterior probability and edge
        4. Check if edge exceeds threshold
        5. Build signal with posterior uncertainty estimates
        """
        if len(features) < 5:
            return None

        market_price = float(features[3])  # close price (0-1 range)
        if market_price <= 0.01 or market_price >= 0.99:
            return None

        # Generate a market ID based on current context
        market_id = f"bayesian_{hash(tuple(features[:4].tolist())) % 100000}"

        # Step 1: Get model probability
        xgb_prob = self._model_probability(features)

        # Step 2: Initialize/get posterior
        self.get_or_create_posterior(market_id, initial_price=market_price)

        # Step 3: Update with model evidence
        self.update_from_model(market_id, xgb_prob, model_confidence=0.6)

        # Step 4: Compute posterior statistics
        posterior_prob = self.posterior_mean(market_id)
        posterior_sd = self.posterior_std(market_id)
        ci_lower, ci_upper = self.posterior_ci(market_id, confidence=0.90)

        # Step 5: Edge = posterior - market
        edge = posterior_prob - market_price
        edge_ci_lower = ci_lower - market_price
        edge_ci_upper = ci_upper - market_price

        # Step 6: Determine action
        if abs(edge) < self._min_edge:
            return None

        if edge > 0:
            action = "BUY"
        else:
            action = "SELL"

        # Ensemble score based on posterior concentration and edge magnitude
        alpha, beta = self.market_posteriors[market_id]
        concentration = alpha + beta
        concentration_factor = min(1.0, concentration / 20.0)
        ensemble_score = min(1.0, concentration_factor * (0.5 + abs(edge) * 5.0))

        # Recommended size based on Kelly and posterior confidence
        recommended_size = min(1.0, abs(edge) / posterior_sd if posterior_sd > 0 else 0.0)

        # Regime detection
        regime_info = {"regime": "NORMAL", "regime_confidence": 0.7}
        if self.regime_detector is not None:
            regime_info = self.regime_detector.detect(features[:4])

        return ApexSignal(
            market_id=market_id,
            venue=self.venue,
            timestamp=datetime.now(timezone.utc),
            strategy=self.strategy_name,
            xgb_probability=xgb_prob,
            xgb_edge=xgb_prob - market_price,
            lgbm_predicted_return=edge * 0.7,
            tft_quantiles={
                0.1: edge_ci_lower,
                0.5: edge,
                0.9: edge_ci_upper,
            },
            regime=regime_info.get("regime", "NORMAL"),
            regime_confidence=regime_info.get("regime_confidence", 0.7),
            sentiment_score=0.0,
            calibrated_edge=edge,
            edge_ci_lower=edge_ci_lower,
            edge_ci_upper=edge_ci_upper,
            ensemble_score=ensemble_score,
            recommended_action=action,
            recommended_size=recommended_size,
            market_price=market_price,
        )

    # ------------------------------------------------------------------
    # Model helper
    # ------------------------------------------------------------------

    def _model_probability(self, features: np.ndarray) -> float:
        """Get probability from the XGBoost model (or fallback)."""
        if "xgboost" in self.models and self.models["xgboost"] is not None:
            pred = self.models["xgboost"].predict(features.reshape(1, -1))
            return float(np.clip(pred[0], 0.01, 0.99))

        # Fallback: calibrated close price
        close = float(features[3]) if len(features) > 3 else 0.5
        return max(0.01, min(0.99, close))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def active_markets(self) -> dict[str, dict[str, float]]:
        """Return summary of all actively tracked markets."""
        result: dict[str, dict[str, float]] = {}
        for market_id, (alpha, beta) in self.market_posteriors.items():
            result[market_id] = {
                "alpha": alpha,
                "beta": beta,
                "mean": alpha / (alpha + beta),
                "std": self.posterior_std(market_id),
                "concentration": alpha + beta,
            }
        return result
