#!/usr/bin/env python3
"""Sports Arbitrage Agent for APEX V2.

Compares sportsbook odds to Kalshi pricing to find mispricings.

Edge source: Sportsbooks are extremely well-calibrated (they make
money from vig, not mispricing). If Kalshi says 55% YES but
sportsbooks say 70% YES, that's a 15% edge.

HIGH CONVICTION ONLY: edge > 8%, confidence > 75%.
"""

import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger()

# The Odds API (needs real key — get free at the-odds-api.com)
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_KEY = "demo"  # Replace with real key

# Free alternative: scrape from ESPN or similar
ESPN_ODDS_BASE = "https://site.api.espn.com/apis/site/v2/sports"

# Kalshi sports series → sport mapping
SPORT_MAP = {
    "KXNBA": "basketball_nba",
    "KXMLB": "baseball_mlb",
    "KXNHL": "icehockey_nhl",
    "KXNFL": "americanfootball_nfl",
}


class SportsAgent:
    """Trades sports futures using sportsbook odds arbitrage."""

    def __init__(self):
        self._odds_cache: dict[str, dict] = {}
        self._cache_ts: dict[str, float] = {}
        self._cache_ttl = 1800  # 30 minutes

    @staticmethod
    def american_to_prob(odds: float) -> float:
        """Convert American odds to implied probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return -odds / (-odds + 100)

    @staticmethod
    def remove_vig(probs: dict[str, float]) -> dict[str, float]:
        """Remove vig from sportsbook odds using multiplicative method.

        If book has Yankees -150 (60%) and opponent +130 (43.5%),
        total = 103.5%. We normalize to 100%.
        """
        total = sum(probs.values())
        if total <= 0:
            return probs
        return {k: v / total for k, v in probs.items()}

    async def fetch_championship_odds(self, client: httpx.AsyncClient,
                                       sport: str) -> dict[str, float]:
        """Fetch championship odds from sportsbooks.

        Returns dict of team_name → fair_probability (vig removed).
        Tries ESPN API first (free), falls back to The Odds API.
        """
        now = time.time()
        if (sport in self._odds_cache
                and now - self._cache_ts.get(sport, 0) < self._cache_ttl):
            return self._odds_cache[sport]

        # Map sport key to ESPN path
        espn_map = {
            "basketball_nba": "basketball/nba",
            "baseball_mlb": "baseball/mlb",
            "icehockey_nhl": "hockey/nhl",
            "americanfootball_nfl": "football/nfl",
        }

        # Try ESPN free API first
        espn_path = espn_map.get(sport)
        if espn_path:
            try:
                resp = await client.get(
                    f"{ESPN_ODDS_BASE}/{espn_path}/standings",
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # ESPN standings don't have championship odds directly
                    # but we can estimate from win percentage
                    teams = {}
                    for group in data.get("children", []):
                        for standing in group.get("standings", {}).get("entries", []):
                            team = standing.get("team", {})
                            name = team.get("displayName", "")
                            stats = {s["name"]: s["value"]
                                    for s in standing.get("stats", [])}
                            win_pct = stats.get("winPercent", 0)
                            if name and win_pct:
                                teams[name] = win_pct

                    if teams:
                        # Normalize to probabilities
                        total = sum(teams.values())
                        fair_probs = {k: v / total for k, v in teams.items()}
                        self._odds_cache[sport] = fair_probs
                        self._cache_ts[sport] = now
                        logger.info("sports.espn_odds", sport=sport,
                                    teams=len(fair_probs))
                        return fair_probs
            except Exception as e:
                logger.debug("sports.espn_error", sport=sport, error=str(e))

        # Fall back to The Odds API
        try:
            resp = await client.get(
                f"{ODDS_API_BASE}/sports/{sport}/odds",
                params={
                    "apiKey": ODDS_API_KEY,
                    "regions": "us",
                    "markets": "h2h",
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                team_probs: dict[str, list[float]] = {}
                for event in data:
                    for bookmaker in event.get("bookmakers", []):
                        for market in bookmaker.get("markets", []):
                            for outcome in market.get("outcomes", []):
                                team = outcome.get("name", "")
                                odds = outcome.get("price", 0)
                                if odds and team:
                                    prob = self.american_to_prob(odds)
                                    if team not in team_probs:
                                        team_probs[team] = []
                                    team_probs[team].append(prob)

                if team_probs:
                    avg_probs = {
                        team: sum(probs) / len(probs)
                        for team, probs in team_probs.items()
                    }
                    fair_probs = self.remove_vig(avg_probs)
                    self._odds_cache[sport] = fair_probs
                    self._cache_ts[sport] = now
                    logger.info("sports.odds_fetched", sport=sport,
                                teams=len(fair_probs))
                    return fair_probs
            else:
                logger.warning("sports.odds_api_error",
                               sport=sport, status=resp.status_code)
        except Exception as e:
            logger.warning("sports.odds_error", sport=sport, error=str(e))

        return {}

    def match_team(self, kalshi_question: str,
                   sportsbook_teams: dict[str, float]) -> Optional[tuple[str, float]]:
        """Match a Kalshi question to a sportsbook team.

        Kalshi: "Will Washington win the 2026 Pro Baseball Championship?"
        Sportsbook: {"Washington Nationals": 0.05, ...}
        """
        q = kalshi_question.lower()

        for team, prob in sportsbook_teams.items():
            team_lower = team.lower()
            # Try exact team name match
            # Extract city/mascot from team name
            parts = team_lower.split()
            for part in parts:
                if len(part) > 3 and part in q:
                    return team, prob

        return None

    def evaluate_signal(self, market: dict,
                        sportsbook_prob: float) -> Optional[dict]:
        """Compare sportsbook probability to Kalshi price."""
        question = market.get("question", "")
        market_price = market.get("current_price", 0.5)

        # Edge = sportsbook fair prob - Kalshi price
        edge = sportsbook_prob - market_price

        if abs(edge) < 0.08:
            return None

        # Confidence: higher when sportsbooks agree
        # and edge is large
        confidence = min(0.90, 0.60 + abs(edge) * 0.5)

        if confidence < 0.75:
            return None

        direction = "BUY" if edge > 0 else "SELL"

        # Conservative sizing for sports
        kelly = 0.15
        size_pct = min(kelly * abs(edge) / 0.10, 0.06)
        size_pct *= confidence

        return {
            "market_id": market["market_id"],
            "venue": "kalshi",
            "question": question[:80],
            "direction": direction,
            "edge": round(edge, 4),
            "true_prob": round(sportsbook_prob, 4),
            "market_price": round(market_price, 4),
            "size_pct": round(size_pct, 4),
            "strategy": "sports",
            "confidence": round(confidence, 3),
            "reason": (
                f"Sportsbook {sportsbook_prob:.1%} vs Kalshi "
                f"{market_price:.1%}, edge {edge:+.1%}"
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "end_date": market.get("end_date", ""),
        }

    async def generate_signals(self, markets: list[dict]) -> list[dict]:
        """Generate sports arbitrage signals with convergence filter."""
        all_signals = []

        async with httpx.AsyncClient(timeout=30) as client:
            # Group markets by sport
            sport_markets: dict[str, list[dict]] = defaultdict(list)
            for market in markets:
                ticker = market.get("market_id", "")
                for prefix, sport in SPORT_MAP.items():
                    if ticker.startswith(prefix):
                        sport_markets[sport].append(market)
                        break

            # Fetch odds and evaluate each sport
            for sport, sport_mks in sport_markets.items():
                odds = await self.fetch_championship_odds(client, sport)
                if not odds:
                    continue

                for market in sport_mks:
                    match = self.match_team(
                        market.get("question", ""), odds
                    )
                    if match:
                        team, prob = match
                        signal = self.evaluate_signal(market, prob)
                        if signal:
                            all_signals.append(signal)

        # Convergence: one trade per event
        event_groups: dict[str, list[dict]] = defaultdict(list)
        for s in all_signals:
            close = s.get("end_date", "")[:10]
            q = s.get("question", "").lower()
            event_key = f"sports_{close}_{q[:20]}"
            event_groups[event_key].append(s)

        signals = []
        for event_key, group in event_groups.items():
            if len(group) == 1:
                signals.append(group[0])
                continue

            group.sort(key=lambda s: abs(s["edge"]), reverse=True)
            best = group[0]
            second = group[1]

            edge_gap = abs(abs(best["edge"]) - abs(second["edge"]))
            if edge_gap < 0.05:
                continue
            if best["direction"] != second["direction"]:
                continue
            signals.append(best)

        signals.sort(key=lambda s: abs(s["edge"]), reverse=True)

        logger.info(
            "sports.signals_generated",
            n=len(signals),
            markets_scanned=len(markets),
        )

        return signals
