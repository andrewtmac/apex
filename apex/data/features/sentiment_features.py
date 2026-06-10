"""
Sentiment Features (25 features)

Rolling headline sentiment windows, sentiment velocity/dispersion,
social-media signals, NLP statistics, and cross-reference features
(sentiment vs price/volume).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as sp_stats

from apex.data.features.builder import FeatureExtractor

_EPS = 1e-12


class SentimentFeatureExtractor(FeatureExtractor):
    """Computes 25 sentiment features.

    Expected keys in *raw_data*::

        # Headline sentiment scores (each entry is a float in [-1, 1])
        headline_sentiments_1h  : list[float]
        headline_sentiments_4h  : list[float]
        headline_sentiments_24h : list[float]

        # Headline metadata
        headline_timestamps_1h  : list[float]  # epoch seconds
        headline_lengths        : list[int]     # character counts of headlines

        # Entity-level
        entity_sentiment_score  : float
        entity_sentiment_rank   : float   # 0-1 percentile among entities
        entity_mention_count    : float

        # Social / Reddit
        reddit_sentiments       : list[float]   # post-level sentiment scores
        reddit_post_counts_1h   : float
        reddit_velocity         : float         # posts/hour recent

        # Source diversity
        source_ids              : list[str]     # unique source identifiers

        # Cross-reference (pre-computed correlations from data layer)
        sentiment_price_corr    : float
        sentiment_volume_corr   : float

        # FinBERT placeholder
        finbert_positive_prob   : float | None
    """

    _NAMES: list[str] = [
        # Headline sentiment (3)
        "sentiment_1h",
        "sentiment_4h",
        "sentiment_24h",
        # Sentiment velocity (2)
        "sentiment_velocity_1h",
        "sentiment_acceleration",
        # Headline counts (3)
        "news_count_1h",
        "news_count_4h",
        "news_count_24h",
        # Sentiment dispersion (2)
        "sentiment_std_24h",
        "sentiment_skew_24h",
        # Entity sentiment (2)
        "entity_sentiment_score",
        "entity_sentiment_rank",
        # Social (3)
        "reddit_sentiment",
        "reddit_volume",
        "reddit_velocity",
        # Overall (3)
        "combined_sentiment_score",
        "sentiment_momentum",
        "sentiment_reversal_signal",
        # NLP (3)
        "avg_headline_length",
        "entity_mention_frequency",
        "source_diversity_index",
        # Cross-reference (3)
        "sentiment_price_correlation",
        "sentiment_volume_correlation",
        "sentiment_divergence",
        # Placeholder (1)
        "finbert_positive_prob",
    ]

    def feature_names(self) -> list[str]:
        return list(self._NAMES)

    async def extract(
        self,
        market_id: str,
        venue: str,
        raw_data: dict[str, Any],
    ) -> dict[str, float]:
        feat: dict[str, float] = {}

        # Grab raw arrays
        sent_1h = np.asarray(raw_data.get("headline_sentiments_1h", []), dtype=np.float64)
        sent_4h = np.asarray(raw_data.get("headline_sentiments_4h", []), dtype=np.float64)
        sent_24h = np.asarray(raw_data.get("headline_sentiments_24h", []), dtype=np.float64)

        # ---- Headline sentiment (rolling means) ----
        feat["sentiment_1h"] = float(np.mean(sent_1h)) if len(sent_1h) else 0.0
        feat["sentiment_4h"] = float(np.mean(sent_4h)) if len(sent_4h) else 0.0
        feat["sentiment_24h"] = float(np.mean(sent_24h)) if len(sent_24h) else 0.0

        # ---- Sentiment velocity & acceleration ----
        # Velocity: change in rolling mean from earlier to later half of 1h window
        if len(sent_1h) >= 4:
            mid = len(sent_1h) // 2
            early_mean = float(np.mean(sent_1h[:mid]))
            late_mean = float(np.mean(sent_1h[mid:]))
            feat["sentiment_velocity_1h"] = late_mean - early_mean
        else:
            feat["sentiment_velocity_1h"] = 0.0

        # Acceleration: compare velocity of 4h first-half vs second-half
        if len(sent_4h) >= 8:
            q1 = len(sent_4h) // 4
            q2 = len(sent_4h) // 2
            q3 = 3 * len(sent_4h) // 4
            v1 = float(np.mean(sent_4h[q1:q2]) - np.mean(sent_4h[:q1]))
            v2 = float(np.mean(sent_4h[q3:]) - np.mean(sent_4h[q2:q3]))
            feat["sentiment_acceleration"] = v2 - v1
        else:
            feat["sentiment_acceleration"] = 0.0

        # ---- Headline counts ----
        feat["news_count_1h"] = float(len(sent_1h))
        feat["news_count_4h"] = float(len(sent_4h))
        feat["news_count_24h"] = float(len(sent_24h))

        # ---- Dispersion ----
        if len(sent_24h) >= 2:
            feat["sentiment_std_24h"] = float(np.std(sent_24h, ddof=1))
            feat["sentiment_skew_24h"] = float(sp_stats.skew(sent_24h, bias=False))
        else:
            feat["sentiment_std_24h"] = 0.0
            feat["sentiment_skew_24h"] = 0.0

        # ---- Entity sentiment ----
        feat["entity_sentiment_score"] = float(raw_data.get("entity_sentiment_score", 0.0))
        feat["entity_sentiment_rank"] = float(raw_data.get("entity_sentiment_rank", 0.5))

        # ---- Social / Reddit ----
        reddit_sent = np.asarray(raw_data.get("reddit_sentiments", []), dtype=np.float64)
        feat["reddit_sentiment"] = float(np.mean(reddit_sent)) if len(reddit_sent) else 0.0
        feat["reddit_volume"] = float(raw_data.get("reddit_post_counts_1h", 0.0))
        feat["reddit_velocity"] = float(raw_data.get("reddit_velocity", 0.0))

        # ---- Overall ----
        # Combined: weighted blend of headline + reddit + entity
        weights = np.array([0.5, 0.3, 0.2])
        components = np.array([
            feat["sentiment_1h"],
            feat["reddit_sentiment"],
            feat["entity_sentiment_score"],
        ])
        feat["combined_sentiment_score"] = float(np.dot(weights, components))

        # Sentiment momentum: 1h vs 24h difference
        feat["sentiment_momentum"] = feat["sentiment_1h"] - feat["sentiment_24h"]

        # Reversal signal: strong recent sentiment opposite to longer-term
        if abs(feat["sentiment_24h"]) > _EPS:
            ratio = feat["sentiment_1h"] / feat["sentiment_24h"]
            feat["sentiment_reversal_signal"] = float(
                -1.0 if ratio < -0.5 else (1.0 if ratio > 2.0 else 0.0)
            )
        else:
            feat["sentiment_reversal_signal"] = 0.0

        # ---- NLP stats ----
        headline_lengths = np.asarray(
            raw_data.get("headline_lengths", []), dtype=np.float64
        )
        feat["avg_headline_length"] = (
            float(np.mean(headline_lengths)) if len(headline_lengths) else 0.0
        )

        entity_mention_count = float(raw_data.get("entity_mention_count", 0.0))
        total_articles = float(len(sent_24h)) if len(sent_24h) else 1.0
        feat["entity_mention_frequency"] = entity_mention_count / total_articles

        # Source diversity: Shannon entropy of source distribution
        source_ids: list[str] = raw_data.get("source_ids", [])
        if source_ids:
            _, counts = np.unique(source_ids, return_counts=True)
            probs = counts / counts.sum()
            feat["source_diversity_index"] = float(-np.sum(probs * np.log(probs + _EPS)))
        else:
            feat["source_diversity_index"] = 0.0

        # ---- Cross-reference ----
        feat["sentiment_price_correlation"] = float(
            raw_data.get("sentiment_price_corr", 0.0)
        )
        feat["sentiment_volume_correlation"] = float(
            raw_data.get("sentiment_volume_corr", 0.0)
        )

        # Divergence: sentiment direction differs from price direction
        price_dir = float(raw_data.get("price_direction_1h", 0.0))
        sent_dir = feat["sentiment_1h"]
        feat["sentiment_divergence"] = float(
            abs(np.sign(sent_dir) - np.sign(price_dir)) / 2.0
        )

        # ---- FinBERT placeholder ----
        finbert_val = raw_data.get("finbert_positive_prob")
        feat["finbert_positive_prob"] = float(finbert_val) if finbert_val is not None else 0.5

        return feat
