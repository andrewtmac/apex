"""
Sports Features (20 features)

Odds movement, betting-volume proxies, historical team stats, situational
factors (rest, travel), weather conditions, and model-based power ratings.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from apex.data.features.builder import FeatureExtractor

_EPS = 1e-12


def _implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability (vig-included)."""
    if decimal_odds <= 1.0:
        return 0.0
    return 1.0 / decimal_odds


class SportsFeatureExtractor(FeatureExtractor):
    """Computes 20 sports-market features.

    Expected keys in *raw_data*::

        # Odds (decimal format)
        home_odds          : float
        away_odds          : float
        draw_odds          : float | None     # None for sports without draws

        # Odds movement
        odds_history_1h    : list[float]      # home odds, 1-min snapshots
        line_open          : float            # opening line (spread/total)
        line_current       : float            # current line

        # Volume / betting
        total_bet_volume   : float            # estimated total handle
        public_bet_pct     : float            # 0-1, % on popular side
        sharp_money_pct    : float            # 0-1, % classified as sharp

        # Historical
        home_wins_season   : int
        home_games_season  : int
        away_wins_season   : int
        away_games_season  : int
        h2h_home_wins      : int
        h2h_total_games    : int

        # Situational
        rest_days_home     : float
        rest_days_away     : float
        travel_distance_km : float            # away team travel

        # Weather
        game_temp_c        : float | None
        game_wind_kph      : float | None
        game_precip_mm     : float | None

        # Model ratings
        elo_home           : float
        elo_away           : float
        power_rating_home  : float
        power_rating_away  : float
    """

    _NAMES: list[str] = [
        # Odds (3)
        "home_implied_prob",
        "away_implied_prob",
        "draw_implied_prob",
        # Movement (3)
        "odds_velocity_1h",
        "odds_acceleration",
        "line_movement_magnitude",
        # Volume (3)
        "betting_volume_proxy",
        "public_pct_proxy",
        "sharp_money_indicator",
        # Historical (3)
        "home_win_rate_season",
        "away_win_rate_season",
        "h2h_record",
        # Situational (3)
        "rest_days_home",
        "rest_days_away",
        "travel_distance",
        # Weather (3)
        "game_temp",
        "game_wind",
        "game_precipitation",
        # Model (2)
        "elo_differential",
        "power_rating_diff",
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

        # ---- Odds ----
        home_odds = float(raw_data.get("home_odds", 2.0))
        away_odds = float(raw_data.get("away_odds", 2.0))
        draw_odds_raw = raw_data.get("draw_odds")

        feat["home_implied_prob"] = _implied_prob(home_odds)
        feat["away_implied_prob"] = _implied_prob(away_odds)
        feat["draw_implied_prob"] = (
            _implied_prob(float(draw_odds_raw)) if draw_odds_raw is not None else 0.0
        )

        # ---- Movement ----
        odds_hist = np.asarray(raw_data.get("odds_history_1h", [home_odds]), dtype=np.float64)

        if len(odds_hist) >= 2:
            velocity = np.diff(odds_hist)
            feat["odds_velocity_1h"] = float(velocity[-1])
            if len(velocity) >= 2:
                feat["odds_acceleration"] = float(np.diff(velocity)[-1])
            else:
                feat["odds_acceleration"] = 0.0
        else:
            feat["odds_velocity_1h"] = 0.0
            feat["odds_acceleration"] = 0.0

        line_open = float(raw_data.get("line_open", 0.0))
        line_current = float(raw_data.get("line_current", line_open))
        feat["line_movement_magnitude"] = abs(line_current - line_open)

        # ---- Volume ----
        feat["betting_volume_proxy"] = float(raw_data.get("total_bet_volume", 0.0))
        feat["public_pct_proxy"] = float(raw_data.get("public_bet_pct", 0.5))

        # Sharp money indicator: difference between sharp % and public %
        sharp_pct = float(raw_data.get("sharp_money_pct", 0.5))
        public_pct = feat["public_pct_proxy"]
        feat["sharp_money_indicator"] = sharp_pct - public_pct

        # ---- Historical ----
        home_wins = int(raw_data.get("home_wins_season", 0))
        home_games = int(raw_data.get("home_games_season", 1))
        away_wins = int(raw_data.get("away_wins_season", 0))
        away_games = int(raw_data.get("away_games_season", 1))

        feat["home_win_rate_season"] = home_wins / max(home_games, 1)
        feat["away_win_rate_season"] = away_wins / max(away_games, 1)

        h2h_home = int(raw_data.get("h2h_home_wins", 0))
        h2h_total = int(raw_data.get("h2h_total_games", 1))
        feat["h2h_record"] = h2h_home / max(h2h_total, 1)

        # ---- Situational ----
        feat["rest_days_home"] = float(raw_data.get("rest_days_home", 3.0))
        feat["rest_days_away"] = float(raw_data.get("rest_days_away", 3.0))
        # Normalise travel distance to [0, ~1] range using log transform
        travel_km = float(raw_data.get("travel_distance_km", 0.0))
        feat["travel_distance"] = float(np.log1p(travel_km) / 10.0)  # log(1+km)/10

        # ---- Weather ----
        game_temp = raw_data.get("game_temp_c")
        feat["game_temp"] = float(game_temp) if game_temp is not None else 20.0

        game_wind = raw_data.get("game_wind_kph")
        feat["game_wind"] = float(game_wind) if game_wind is not None else 0.0

        game_precip = raw_data.get("game_precip_mm")
        feat["game_precipitation"] = float(game_precip) if game_precip is not None else 0.0

        # ---- Model ratings ----
        elo_home = float(raw_data.get("elo_home", 1500.0))
        elo_away = float(raw_data.get("elo_away", 1500.0))
        feat["elo_differential"] = elo_home - elo_away

        pr_home = float(raw_data.get("power_rating_home", 0.0))
        pr_away = float(raw_data.get("power_rating_away", 0.0))
        feat["power_rating_diff"] = pr_home - pr_away

        return feat
