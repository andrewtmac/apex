"""
Macro / Cross-Market Features (20 features)

Market-regime signals (equity indices, VIX, crypto), calendar seasonality
(cyclical encoding), macro-event proximity, and cross-venue divergence.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from apex.data.features.builder import FeatureExtractor

_EPS = 1e-12


def _cyclical_encode(value: float, period: float) -> tuple[float, float]:
    """Encode a cyclic quantity as (sin, cos) pair."""
    angle = 2.0 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


class MacroFeatureExtractor(FeatureExtractor):
    """Computes 20 macro / cross-market features.

    Expected keys in *raw_data*::

        # Equity / VIX
        spy_price_1h_ago   : float
        spy_price_24h_ago  : float
        spy_price_now      : float
        vix_level          : float
        vix_level_24h_ago  : float

        # Crypto
        btc_price_now      : float
        btc_price_1h_ago   : float
        btc_price_24h_ago  : float
        eth_price_now      : float
        eth_price_1h_ago   : float
        crypto_total_mcap  : float
        btc_mcap           : float

        # Timestamp (epoch seconds, defaults to now)
        timestamp          : float

        # Macro events (epoch seconds or days)
        next_fomc_ts       : float | None
        is_earnings_season : bool
        next_opex_ts       : float | None

        # Cross-venue
        poly_price         : float | None
        kalshi_price       : float | None
        poly_volume        : float | None
        kalshi_volume      : float | None
        poly_spread        : float | None
        kalshi_spread      : float | None
    """

    _NAMES: list[str] = [
        # Market indices (4)
        "spy_return_1h",
        "spy_return_24h",
        "vix_level",
        "vix_change_24h",
        # Crypto (4)
        "btc_return_1h",
        "btc_return_24h",
        "eth_return_1h",
        "crypto_dominance",
        # Calendar (6)
        "hour_of_day_sin",
        "hour_of_day_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "is_market_hours",
        "is_weekend",
        # Macro events (3)
        "days_to_next_fomc",
        "is_earnings_season",
        "days_to_next_opex",
        # Cross-venue (3)
        "poly_kalshi_price_divergence",
        "cross_venue_volume_ratio",
        "venue_spread_ratio",
    ]

    def feature_names(self) -> list[str]:
        return list(self._NAMES)

    async def extract(
        self,
        market_id: str,
        venue: str,
        raw_data: dict[str, Any],
    ) -> dict[str, float]:
        now = float(raw_data.get("timestamp", time.time()))
        feat: dict[str, float] = {}

        # ---- Equity / VIX ----
        spy_now = float(raw_data.get("spy_price_now", 0.0))
        spy_1h = float(raw_data.get("spy_price_1h_ago", spy_now))
        spy_24h = float(raw_data.get("spy_price_24h_ago", spy_now))

        feat["spy_return_1h"] = (
            (spy_now - spy_1h) / spy_1h if spy_1h > _EPS else 0.0
        )
        feat["spy_return_24h"] = (
            (spy_now - spy_24h) / spy_24h if spy_24h > _EPS else 0.0
        )

        vix = float(raw_data.get("vix_level", 20.0))
        vix_24h = float(raw_data.get("vix_level_24h_ago", vix))
        feat["vix_level"] = vix
        feat["vix_change_24h"] = vix - vix_24h

        # ---- Crypto ----
        btc_now = float(raw_data.get("btc_price_now", 0.0))
        btc_1h = float(raw_data.get("btc_price_1h_ago", btc_now))
        btc_24h = float(raw_data.get("btc_price_24h_ago", btc_now))
        eth_now = float(raw_data.get("eth_price_now", 0.0))
        eth_1h = float(raw_data.get("eth_price_1h_ago", eth_now))

        feat["btc_return_1h"] = (
            (btc_now - btc_1h) / btc_1h if btc_1h > _EPS else 0.0
        )
        feat["btc_return_24h"] = (
            (btc_now - btc_24h) / btc_24h if btc_24h > _EPS else 0.0
        )
        feat["eth_return_1h"] = (
            (eth_now - eth_1h) / eth_1h if eth_1h > _EPS else 0.0
        )

        crypto_total = float(raw_data.get("crypto_total_mcap", 1.0))
        btc_mcap = float(raw_data.get("btc_mcap", 0.0))
        feat["crypto_dominance"] = (
            btc_mcap / crypto_total if crypto_total > _EPS else 0.0
        )

        # ---- Calendar ----
        import datetime as _dt

        dt = _dt.datetime.fromtimestamp(now, tz=_dt.timezone.utc)
        hour = dt.hour + dt.minute / 60.0
        dow = dt.weekday()  # Mon=0 .. Sun=6

        h_sin, h_cos = _cyclical_encode(hour, 24.0)
        d_sin, d_cos = _cyclical_encode(dow, 7.0)
        feat["hour_of_day_sin"] = h_sin
        feat["hour_of_day_cos"] = h_cos
        feat["day_of_week_sin"] = d_sin
        feat["day_of_week_cos"] = d_cos

        # US equity market hours: 9:30-16:00 ET (approx 13:30-20:00 UTC)
        feat["is_market_hours"] = 1.0 if (13.5 <= hour < 20.0 and dow < 5) else 0.0
        feat["is_weekend"] = 1.0 if dow >= 5 else 0.0

        # ---- Macro events ----
        next_fomc = raw_data.get("next_fomc_ts")
        if next_fomc is not None:
            feat["days_to_next_fomc"] = max(float(next_fomc) - now, 0.0) / 86400.0
        else:
            feat["days_to_next_fomc"] = 45.0  # neutral default (~6 weeks)

        feat["is_earnings_season"] = 1.0 if raw_data.get("is_earnings_season", False) else 0.0

        next_opex = raw_data.get("next_opex_ts")
        if next_opex is not None:
            feat["days_to_next_opex"] = max(float(next_opex) - now, 0.0) / 86400.0
        else:
            feat["days_to_next_opex"] = 21.0  # neutral default (~3 weeks)

        # ---- Cross-venue divergence ----
        poly_price = raw_data.get("poly_price")
        kalshi_price = raw_data.get("kalshi_price")
        if poly_price is not None and kalshi_price is not None:
            feat["poly_kalshi_price_divergence"] = abs(float(poly_price) - float(kalshi_price))
        else:
            feat["poly_kalshi_price_divergence"] = 0.0

        poly_vol = raw_data.get("poly_volume")
        kalshi_vol = raw_data.get("kalshi_volume")
        if poly_vol is not None and kalshi_vol is not None and float(kalshi_vol) > _EPS:
            feat["cross_venue_volume_ratio"] = float(poly_vol) / float(kalshi_vol)
        else:
            feat["cross_venue_volume_ratio"] = 1.0

        poly_spread = raw_data.get("poly_spread")
        kalshi_spread = raw_data.get("kalshi_spread")
        if poly_spread is not None and kalshi_spread is not None and float(kalshi_spread) > _EPS:
            feat["venue_spread_ratio"] = float(poly_spread) / float(kalshi_spread)
        else:
            feat["venue_spread_ratio"] = 1.0

        return feat
