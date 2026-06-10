"""
APEX Social Sentiment Ingester

Placeholder module for Reddit / Twitter social media scraping
and NLP sentiment pipeline integration.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog

from apex.config import ApexConfig
from apex.data.store import FeatureStore
from apex.data.streams import SOCIAL_STREAM, StreamPublisher

logger = structlog.get_logger(__name__)

_POLL_INTERVAL = 60  # seconds


class SocialIngester:
    """
    Social media sentiment ingester.

    Currently a structured placeholder.  When live data sources are added
    (Reddit pushshift, Twitter/X API, etc.) the ``_fetch_*`` methods will
    be implemented.

    Lifecycle::

        ingester = SocialIngester(config)
        await ingester.start()
        await ingester.stop()
    """

    def __init__(self, config: ApexConfig):
        self._redis_url = config.infra.redis_url
        self._db_url = config.infra.database_url
        self._publisher: Optional[StreamPublisher] = None
        self._store: Optional[FeatureStore] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._running = False
        self._tasks: list[asyncio.Task] = []

        # Subreddits / hashtags to track
        self._subreddits = [
            "wallstreetbets",
            "polymarket",
            "sports betting",
            "weather",
            "politics",
        ]
        self._hashtags = [
            "#polymarket",
            "#kalshi",
            "#predictionmarkets",
        ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._publisher = StreamPublisher(self._redis_url)
        await self._publisher.connect()

        self._store = FeatureStore(self._redis_url, self._db_url)
        await self._store.connect()

        self._http = httpx.AsyncClient(timeout=httpx.Timeout(20.0))
        self._running = True

        self._tasks = [
            asyncio.create_task(self._poll_loop(), name="social-poll"),
        ]
        logger.info(
            "social_ingester.started",
            subreddits=self._subreddits,
            hashtags=self._hashtags,
        )

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._http:
            await self._http.aclose()
        if self._publisher:
            await self._publisher.close()
        if self._store:
            await self._store.close()
        logger.info("social_ingester.stopped")

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        while self._running:
            await self._fetch_all()
            await asyncio.sleep(_POLL_INTERVAL)

    async def _fetch_all(self) -> None:
        """Aggregate sentiment signals from all configured sources."""
        assert self._publisher is not None and self._store is not None

        signals: list[dict[str, Any]] = []

        # Reddit (placeholder)
        reddit_signals = await self._fetch_reddit()
        signals.extend(reddit_signals)

        # Twitter / X (placeholder)
        twitter_signals = await self._fetch_twitter()
        signals.extend(twitter_signals)

        if signals:
            aggregate = self._aggregate_sentiment(signals)
            await self._publisher.publish(SOCIAL_STREAM, aggregate)
            await self._store.put(
                entity_id="social:aggregate",
                feature_set="social_sentiment",
                features=aggregate,
            )
            logger.debug("social.published", n_signals=len(signals))

    # ------------------------------------------------------------------
    # Source fetchers (placeholders)
    # ------------------------------------------------------------------

    async def _fetch_reddit(self) -> list[dict[str, Any]]:
        """
        Fetch posts from tracked subreddits.

        TODO: Implement using Reddit JSON endpoints or Pushshift.
        The structure is ready — just needs the HTTP calls and parsing.

        Expected flow:
        1. GET https://www.reddit.com/r/{subreddit}/new.json?limit=25
        2. Parse title, selftext, score, num_comments
        3. Run NLP sentiment on title
        4. Return list of signal dicts
        """
        # Placeholder: return empty list until Reddit source is wired
        return []

    async def _fetch_twitter(self) -> list[dict[str, Any]]:
        """
        Fetch recent tweets matching tracked hashtags.

        TODO: Implement using Twitter API v2 or X API.
        Requires bearer token in env (TWITTER_BEARER_TOKEN).

        Expected flow:
        1. GET /2/tweets/search/recent?query={hashtag}
        2. Parse text, author, public_metrics
        3. Run NLP sentiment
        4. Return list of signal dicts
        """
        # Placeholder: return empty list until Twitter source is wired
        return []

    # ------------------------------------------------------------------
    # NLP Pipeline (placeholder)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_sentiment(text: str) -> dict[str, float]:
        """
        Run sentiment analysis on text.

        TODO: Wire up FinBERT or distilbert-base-uncased-finetuned-sst-2:

            from transformers import pipeline
            nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            result = nlp(text)[0]
            return {"label": result["label"], "score": result["score"]}

        For now returns a neutral placeholder.
        """
        return {"label": "neutral", "score": 0.5, "positive": 0.33, "negative": 0.33, "neutral": 0.34}

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_sentiment(signals: list[dict[str, Any]]) -> dict[str, Any]:
        """Combine individual signals into an aggregate sentiment reading."""
        if not signals:
            return {
                "ts": datetime.now(timezone.utc).isoformat(),
                "n_signals": 0,
                "avg_score": 0.5,
                "bullish_pct": 0,
                "bearish_pct": 0,
                "sources": [],
            }

        scores = [s.get("sentiment_score", 0.5) for s in signals]
        avg_score = sum(scores) / len(scores)
        bullish = sum(1 for s in scores if s > 0.6)
        bearish = sum(1 for s in scores if s < 0.4)

        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "n_signals": len(signals),
            "avg_score": round(avg_score, 4),
            "bullish_pct": round(bullish / len(signals) * 100, 1),
            "bearish_pct": round(bearish / len(signals) * 100, 1),
            "sources": list({s.get("source", "unknown") for s in signals}),
            "top_topics": _extract_topics(signals),
        }


def _extract_topics(signals: list[dict[str, Any]], top_n: int = 5) -> list[str]:
    """Extract the most mentioned topics from signals."""
    topic_counts: dict[str, int] = {}
    for s in signals:
        for topic in s.get("topics", []):
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
    sorted_topics = sorted(topic_counts, key=topic_counts.get, reverse=True)  # type: ignore[arg-type]
    return sorted_topics[:top_n]
