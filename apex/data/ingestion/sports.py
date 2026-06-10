"""
APEX Sports Odds Ingester

Polls The Odds API for live odds across major sports, tracks line
movements, and publishes to the apex:sports Redis stream.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog

from apex.config import ApexConfig
from apex.data.store import FeatureStore
from apex.data.streams import SPORTS_STREAM, StreamPublisher

logger = structlog.get_logger(__name__)

_POLL_INTERVAL = 60  # seconds
_BASE_URL = "https://api.the-odds-api.com/v4"

# Sports to track
DEFAULT_SPORTS = [
    "americanfootball_nfl",
    "basketball_nba",
    "baseball_mlb",
    "icehockey_nhl",
    "soccer_epl",
    "soccer_usa_mls",
    "mma_mixed_martial_arts",
]


class SportsIngester:
    """
    Ingests live odds from The Odds API.

    Lifecycle::

        ingester = SportsIngester(config)
        await ingester.start()
        await ingester.stop()
    """

    def __init__(self, config: ApexConfig):
        self._api_key = config.data_sources.odds_api_key
        self._redis_url = config.infra.redis_url
        self._db_url = config.infra.database_url
        self._publisher: Optional[StreamPublisher] = None
        self._store: Optional[FeatureStore] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._sports = DEFAULT_SPORTS
        self._prev_odds: dict[str, dict[str, Any]] = {}  # event_id -> last odds
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._remaining_requests: Optional[int] = None

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
            asyncio.create_task(self._poll_loop(), name="sports-poll"),
        ]
        logger.info("sports_ingester.started", n_sports=len(self._sports))

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
        logger.info("sports_ingester.stopped")

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        while self._running:
            await self._fetch_all_sports()
            await asyncio.sleep(_POLL_INTERVAL)

    async def _fetch_all_sports(self) -> None:
        for sport in self._sports:
            # Check API quota
            if self._remaining_requests is not None and self._remaining_requests < 5:
                logger.warning(
                    "sports.quota_low",
                    remaining=self._remaining_requests,
                )
                return

            await self._fetch_sport_odds(sport)
            await asyncio.sleep(1)  # rate limiting

    async def _fetch_sport_odds(self, sport: str) -> None:
        """Fetch odds for a single sport from The Odds API."""
        assert self._http is not None and self._publisher is not None and self._store is not None

        try:
            resp = await self._http.get(
                f"{_BASE_URL}/sports/{sport}/odds",
                params={
                    "apiKey": self._api_key,
                    "regions": "us",
                    "markets": "h2h,spreads,totals",
                    "oddsFormat": "american",
                    "bookmakers": "draftkings,fanduel,betmgm,pointsbet",
                },
            )
            resp.raise_for_status()

            # Track API usage from response headers
            remaining = resp.headers.get("x-requests-remaining")
            if remaining is not None:
                self._remaining_requests = int(remaining)

            events = resp.json()
            new_events = 0

            for event in events:
                event_id = event.get("id", "")
                parsed = self._parse_event(event)

                # Detect line movement
                prev = self._prev_odds.get(event_id)
                if prev:
                    parsed["line_movement"] = self._compute_line_movement(prev, parsed)

                self._prev_odds[event_id] = parsed

                await self._publisher.publish(SPORTS_STREAM, parsed)
                await self._store.put(
                    entity_id=f"sports:{event_id}",
                    feature_set="odds_snapshot",
                    features=parsed,
                )
                new_events += 1

            if new_events > 0:
                logger.info(
                    "sports.odds_fetched",
                    sport=sport,
                    events=new_events,
                    api_remaining=self._remaining_requests,
                )

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.warning("sports.rate_limited", sport=sport)
            else:
                logger.warning("sports.fetch_failed", sport=sport, status=exc.response.status_code)
        except Exception:
            logger.exception("sports.fetch_error", sport=sport)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_event(event: dict[str, Any]) -> dict[str, Any]:
        """Parse a raw Odds API event into a structured dict."""
        bookmakers = event.get("bookmakers", [])

        # Aggregate odds across bookmakers
        h2h_odds: dict[str, list[int]] = {}  # team -> [line1, line2, ...]
        spreads: dict[str, list[float]] = {}
        totals: list[float] = []

        for bm in bookmakers:
            for market in bm.get("markets", []):
                mkt_key = market.get("key", "")
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price", 0)

                    if mkt_key == "h2h":
                        h2h_odds.setdefault(name, []).append(price)
                    elif mkt_key == "spreads":
                        point = outcome.get("point", 0)
                        spreads.setdefault(name, []).append(point)
                    elif mkt_key == "totals" and name in ("Over", "Under"):
                        point = outcome.get("point", 0)
                        totals.append(point)

        # Compute consensus (average) odds
        consensus_h2h = {
            team: round(sum(lines) / len(lines))
            for team, lines in h2h_odds.items()
            if lines
        }
        consensus_spreads = {
            team: round(sum(pts) / len(pts), 1)
            for team, pts in spreads.items()
            if pts
        }
        consensus_total = round(sum(totals) / len(totals), 1) if totals else None

        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_id": event.get("id", ""),
            "sport": event.get("sport_key", ""),
            "sport_title": event.get("sport_title", ""),
            "commence_time": event.get("commence_time", ""),
            "home_team": event.get("home_team", ""),
            "away_team": event.get("away_team", ""),
            "n_bookmakers": len(bookmakers),
            "consensus_h2h": consensus_h2h,
            "consensus_spreads": consensus_spreads,
            "consensus_total": consensus_total,
            "bookmaker_names": [bm.get("title", "") for bm in bookmakers],
        }

    @staticmethod
    def _compute_line_movement(
        prev: dict[str, Any],
        current: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute how odds have moved since last snapshot."""
        movements: dict[str, Any] = {}

        prev_h2h = prev.get("consensus_h2h", {})
        curr_h2h = current.get("consensus_h2h", {})
        for team in set(prev_h2h) | set(curr_h2h):
            old = prev_h2h.get(team, 0)
            new = curr_h2h.get(team, 0)
            if old != new:
                movements[f"h2h_{team}"] = {"from": old, "to": new, "delta": new - old}

        prev_total = prev.get("consensus_total")
        curr_total = current.get("consensus_total")
        if prev_total and curr_total and prev_total != curr_total:
            movements["total"] = {
                "from": prev_total,
                "to": curr_total,
                "delta": round(curr_total - prev_total, 1),
            }

        return movements
