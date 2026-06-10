"""
APEX News Ingester

Polls NewsAPI.ai (Event Registry) for recent articles, extracts metadata,
and publishes to the apex:news Redis stream.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog

from apex.config import ApexConfig
from apex.data.store import FeatureStore
from apex.data.streams import NEWS_STREAM, StreamPublisher

logger = structlog.get_logger(__name__)

_POLL_INTERVAL = 30  # seconds
_NEWSAPI_BASE = "https://eventregistry.org/api/v1"


class NewsIngester:
    """
    Ingests news articles from NewsAPI.ai (Event Registry).

    Lifecycle::

        ingester = NewsIngester(config)
        await ingester.start()
        await ingester.stop()
    """

    def __init__(self, config: ApexConfig):
        self._api_key = config.data_sources.newsapi_ai_key
        self._redis_url = config.infra.redis_url
        self._db_url = config.infra.database_url
        self._publisher: Optional[StreamPublisher] = None
        self._store: Optional[FeatureStore] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._seen_uris: set[str] = set()
        self._max_seen = 10_000
        self._running = False
        self._tasks: list[asyncio.Task] = []

        # Categories relevant to prediction markets
        self._categories = [
            "politics",
            "economics",
            "finance",
            "weather",
            "sports",
            "technology",
            "science",
            "health",
        ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._publisher = StreamPublisher(self._redis_url)
        await self._publisher.connect()

        self._store = FeatureStore(self._redis_url, self._db_url)
        await self._store.connect()

        self._http = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        self._running = True

        self._tasks = [
            asyncio.create_task(self._poll_loop(), name="news-poll"),
        ]
        logger.info("news_ingester.started")

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
        logger.info("news_ingester.stopped")

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        while self._running:
            await self._fetch_articles()
            await asyncio.sleep(_POLL_INTERVAL)

    async def _fetch_articles(self) -> None:
        """Fetch recent articles from Event Registry."""
        assert self._http is not None and self._publisher is not None and self._store is not None

        try:
            payload = {
                "action": "getArticles",
                "keyword": "prediction market OR election OR weather forecast OR sports betting",
                "articlesPage": 1,
                "articlesCount": 50,
                "articlesSortBy": "date",
                "articlesSortByAsc": False,
                "articlesArticleBodyLen": 300,
                "resultType": "articles",
                "dataType": ["news"],
                "lang": "eng",
                "apiKey": self._api_key,
            }

            resp = await self._http.post(
                f"{_NEWSAPI_BASE}/article/getArticles",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

            articles = data.get("articles", {}).get("results", [])
            new_count = 0

            for article in articles:
                uri = article.get("uri", "")
                if not uri or uri in self._seen_uris:
                    continue

                self._seen_uris.add(uri)
                new_count += 1

                parsed = self._parse_article(article)
                await self._publisher.publish(NEWS_STREAM, parsed)
                await self._store.put(
                    entity_id=f"news:{uri}",
                    feature_set="news_article",
                    features=parsed,
                )

            # Trim seen set to prevent memory bloat
            if len(self._seen_uris) > self._max_seen:
                to_remove = len(self._seen_uris) - self._max_seen
                for _ in range(to_remove):
                    self._seen_uris.pop()

            if new_count > 0:
                logger.info("news.articles_fetched", new=new_count, total_seen=len(self._seen_uris))

        except Exception:
            logger.exception("news.fetch_failed")

    # ------------------------------------------------------------------
    # Article parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_article(article: dict[str, Any]) -> dict[str, Any]:
        """Extract structured fields from a raw Event Registry article."""
        # Extract entities (people, orgs, locations)
        concepts = article.get("concepts", [])
        entities = [
            {
                "label": c.get("label", {}).get("eng", ""),
                "type": c.get("type", ""),
                "score": c.get("score", 0),
            }
            for c in concepts[:10]
        ]

        # Extract categories
        categories = [
            {
                "label": cat.get("label", ""),
                "score": cat.get("score", 0),
            }
            for cat in article.get("categories", [])[:5]
        ]

        source = article.get("source", {})

        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "uri": article.get("uri", ""),
            "title": article.get("title", ""),
            "body": article.get("body", "")[:500],  # truncate body
            "source_name": source.get("title", ""),
            "source_uri": source.get("uri", ""),
            "published_at": article.get("dateTimePub", ""),
            "language": article.get("lang", "eng"),
            "entities": entities,
            "categories": categories,
            "relevance": article.get("relevance", 0),
            "sentiment": None,  # placeholder for FinBERT - will be filled by NLP pipeline
            "image_url": article.get("image", ""),
        }

    # ------------------------------------------------------------------
    # Sentiment placeholder
    # ------------------------------------------------------------------

    @staticmethod
    def compute_sentiment(title: str) -> Optional[dict[str, float]]:
        """
        Placeholder for FinBERT sentiment analysis.

        In production this will run::

            from transformers import pipeline
            nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            result = nlp(title)

        For now returns None; the NLP pipeline will enrich articles downstream.
        """
        return None
