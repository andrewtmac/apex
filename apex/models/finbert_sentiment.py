"""
APEX Model F: FinBERT Sentiment Classifier

FinBERT-based headline sentiment classifier.  Uses the ProsusAI/finbert
model (110M parameters) for CPU inference at <50ms per headline.

This is a trained transformer classifier -- NOT an LLM prompt.  It outputs
softmax probabilities across three classes: positive, negative, neutral.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FinBERTSentimentModel:
    """
    FinBERT headline sentiment classifier.

    Lazy-loads the model on first use to avoid slow import-time downloads.

    Usage
    -----
    >>> model = FinBERTSentimentModel()
    >>> model.load()
    >>> sentiment = model.predict("Fed signals rate pause")
    >>> # {"positive": 0.72, "negative": 0.08, "neutral": 0.20}
    >>> score = model.aggregate_sentiment(sentiments, timestamps)
    """

    LABELS = ["positive", "negative", "neutral"]

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        max_length: int = 128,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._device_str = device

        self.tokenizer: Any = None
        self.model: Any = None
        self._loaded: bool = False
        self._device: Any = None  # torch.device

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load model and tokenizer from HuggingFace Hub (or local cache).
        This downloads ~440 MB on first call and caches for future runs.
        """
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self._device_str is None:
            self._device = torch.device("cpu")  # FinBERT is fast enough on CPU
        else:
            self._device = torch.device(self._device_str)

        logger.info("Loading FinBERT from %s ...", self.model_name)
        t0 = time.monotonic()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
        ).to(self._device)
        self.model.eval()

        elapsed = time.monotonic() - t0
        logger.info("FinBERT loaded in %.1fs on %s", elapsed, self._device)
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ------------------------------------------------------------------
    # Single prediction
    # ------------------------------------------------------------------

    def predict(self, headline: str) -> dict[str, float]:
        """
        Predict sentiment for a single headline.

        Parameters
        ----------
        headline : news headline text.

        Returns
        -------
        Dict mapping label -> probability, e.g.:
            {"positive": 0.72, "negative": 0.08, "neutral": 0.20}
        """
        self._ensure_loaded()
        import torch

        inputs = self.tokenizer(
            headline,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        return self._probs_to_dict(probs)

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        headlines: list[str],
        batch_size: int = 32,
    ) -> list[dict[str, float]]:
        """
        Batch inference for efficiency.

        Parameters
        ----------
        headlines : list of headline strings.
        batch_size : number of headlines per forward pass.

        Returns
        -------
        List of sentiment dicts, one per headline.
        """
        self._ensure_loaded()
        import torch

        results: list[dict[str, float]] = []

        for i in range(0, len(headlines), batch_size):
            batch = headlines[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            for row in probs:
                results.append(self._probs_to_dict(row))

        return results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_sentiment(
        self,
        sentiments: list[dict[str, float]],
        timestamps: list[float],
        half_life_hours: float = 4.0,
        reference_time: float | None = None,
    ) -> float:
        """
        Compute exponentially weighted moving average sentiment score.

        Converts each sentiment dict into a scalar:
            score = positive - negative
        Then applies exponential decay based on age relative to *reference_time*.

        Parameters
        ----------
        sentiments : list of sentiment dicts from ``predict`` / ``predict_batch``.
        timestamps : list of Unix timestamps (seconds) for each headline.
        half_life_hours : decay half-life in hours.
        reference_time : reference time (Unix seconds). Defaults to ``time.time()``.

        Returns
        -------
        float in [-1, 1] -- aggregated sentiment score.
        """
        if not sentiments:
            return 0.0

        if reference_time is None:
            reference_time = time.time()

        decay_rate = math.log(2) / (half_life_hours * 3600)

        weighted_sum = 0.0
        weight_total = 0.0

        for sent, ts in zip(sentiments, timestamps):
            score = sent.get("positive", 0.0) - sent.get("negative", 0.0)
            age_seconds = max(reference_time - ts, 0.0)
            weight = math.exp(-decay_rate * age_seconds)
            weighted_sum += score * weight
            weight_total += weight

        if weight_total < 1e-12:
            return 0.0

        return float(np.clip(weighted_sum / weight_total, -1.0, 1.0))

    def sentiment_score(self, headline: str) -> float:
        """
        Convenience: scalar sentiment in [-1, 1] for a single headline.

        +1 = strongly positive, -1 = strongly negative.
        """
        probs = self.predict(headline)
        return probs["positive"] - probs["negative"]

    # ------------------------------------------------------------------
    # News velocity
    # ------------------------------------------------------------------

    @staticmethod
    def compute_news_velocity(
        timestamps: list[float],
        window_hours: float = 1.0,
        reference_time: float | None = None,
    ) -> float:
        """
        Compute news arrival rate (headlines/hour) in a recent window.

        Useful as a feature for the ensemble: high velocity may indicate
        a market-moving event.

        Returns
        -------
        float -- headlines per hour in the lookback window.
        """
        if not timestamps:
            return 0.0

        if reference_time is None:
            reference_time = time.time()

        cutoff = reference_time - window_hours * 3600
        recent = [t for t in timestamps if t >= cutoff]
        return len(recent) / window_hours

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _probs_to_dict(self, probs: np.ndarray) -> dict[str, float]:
        """Map probability array to labelled dict."""
        # FinBERT label ordering: positive=0, negative=1, neutral=2
        return {
            "positive": float(probs[0]),
            "negative": float(probs[1]),
            "neutral": float(probs[2]),
        }

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"<FinBERTSentimentModel [{status}, model={self.model_name}]>"
