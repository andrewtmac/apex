"""
Prediction Market Features (30 features)

Features tailored to prediction / event markets (Polymarket, Kalshi, etc.):
contract-level signals, spread analytics, time-decay curves, cross-market
divergences, and market-quality metrics.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from scipy import stats as sp_stats

from apex.data.features.builder import FeatureExtractor

_EPS = 1e-12


class PredictionFeatureExtractor(FeatureExtractor):
    """Computes 30 prediction-market features.

    Expected keys in *raw_data*::

        # Contract info
        last_price        : float          # latest trade price (0-1 scale)
        best_bid          : float
        best_ask          : float
        expiry_ts         : float          # expiry epoch seconds
        created_ts        : float          # creation epoch seconds

        # Volume
        volume_24h        : float          # USD volume last 24 h
        volume_7d         : float          # USD volume last 7 d
        volume_history    : list[float]    # hourly volume snapshots

        # Spread history
        spread_history_1h : list[float]    # spread snapshots (1-min) last hour
        spread_history_7d : list[float]    # spread snapshots (hourly) last 7 d

        # Price history
        price_history_1h  : list[float]    # 1-min prices last hour
        price_history_7d  : list[float]    # hourly prices last 7 d

        # Resolution stats
        historical_resolution_rate : float  # 0-1
        category_resolution_rate   : float  # 0-1

        # Orderbook depth (multiple levels)
        bid_sizes         : list[float]
        ask_sizes         : list[float]

        # Correlated markets
        correlated_prices : list[float]    # latest prices of related markets
        num_correlated    : int

        # Participants
        num_unique_traders_est : float
    """

    _NAMES: list[str] = [
        # Contract (4)
        "implied_probability",
        "days_to_expiry",
        "hours_to_expiry",
        "log_time_to_expiry",
        # Volume (4)
        "pred_volume_24h",
        "pred_volume_7d",
        "volume_acceleration",
        "volume_zscore_7d",
        # Spread (3)
        "spread_bps",
        "spread_zscore_1h",
        "spread_percentile_7d",
        # Price position (4)
        "distance_from_50c",
        "distance_from_round_number",
        "price_velocity_1h",
        "price_acceleration",
        # Historical (2)
        "historical_resolution_rate",
        "category_resolution_rate",
        # Orderbook (3)
        "depth_ratio_bid_ask",
        "depth_imbalance_5_levels",
        "total_liquidity_usd",
        # Cross-market (3)
        "num_correlated_markets",
        "avg_correlated_price",
        "correlated_price_divergence",
        # Time decay (3)
        "exponential_decay_factor",
        "sqrt_time_factor",
        "days_since_creation",
        # Market quality (4)
        "num_unique_traders_est",
        "price_stability_24h",
        "max_drawdown_7d",
        "recovery_rate",
    ]

    def feature_names(self) -> list[str]:
        return list(self._NAMES)

    async def extract(
        self,
        market_id: str,
        venue: str,
        raw_data: dict[str, Any],
    ) -> dict[str, float]:
        now = time.time()
        feat: dict[str, float] = {}

        last_price = float(raw_data.get("last_price", 0.5))
        best_bid = float(raw_data.get("best_bid", 0.0))
        best_ask = float(raw_data.get("best_ask", 0.0))
        expiry_ts = float(raw_data.get("expiry_ts", now + 86400))
        created_ts = float(raw_data.get("created_ts", now - 86400))

        # ---- Contract ----
        feat["implied_probability"] = last_price

        seconds_to_expiry = max(expiry_ts - now, 0.0)
        feat["days_to_expiry"] = seconds_to_expiry / 86400.0
        feat["hours_to_expiry"] = seconds_to_expiry / 3600.0
        feat["log_time_to_expiry"] = float(math.log(seconds_to_expiry + 1.0))

        # ---- Volume ----
        vol_24h = float(raw_data.get("volume_24h", 0.0))
        vol_7d = float(raw_data.get("volume_7d", 0.0))
        feat["pred_volume_24h"] = vol_24h
        feat["pred_volume_7d"] = vol_7d

        vol_hist = np.asarray(raw_data.get("volume_history", []), dtype=np.float64)
        if len(vol_hist) >= 3:
            vel = np.diff(vol_hist)
            accel = np.diff(vel)
            feat["volume_acceleration"] = float(accel[-1])
        else:
            feat["volume_acceleration"] = 0.0

        if len(vol_hist) >= 2:
            vol_mean = np.mean(vol_hist)
            vol_std = np.std(vol_hist, ddof=1)
            feat["volume_zscore_7d"] = (
                float((vol_hist[-1] - vol_mean) / vol_std) if vol_std > _EPS else 0.0
            )
        else:
            feat["volume_zscore_7d"] = 0.0

        # ---- Spread ----
        mid = (best_bid + best_ask) / 2.0 if (best_bid + best_ask) > 0 else last_price
        spread = best_ask - best_bid
        feat["spread_bps"] = (spread / mid * 10_000) if mid > _EPS else 0.0

        spread_hist_1h = np.asarray(raw_data.get("spread_history_1h", []), dtype=np.float64)
        if len(spread_hist_1h) >= 2:
            sp_mean = np.mean(spread_hist_1h)
            sp_std = np.std(spread_hist_1h, ddof=1)
            feat["spread_zscore_1h"] = (
                float((spread - sp_mean) / sp_std) if sp_std > _EPS else 0.0
            )
        else:
            feat["spread_zscore_1h"] = 0.0

        spread_hist_7d = np.asarray(raw_data.get("spread_history_7d", []), dtype=np.float64)
        if len(spread_hist_7d) >= 2:
            feat["spread_percentile_7d"] = float(
                sp_stats.percentileofscore(spread_hist_7d, spread, kind="rank") / 100.0
            )
        else:
            feat["spread_percentile_7d"] = 0.5

        # ---- Price position ----
        feat["distance_from_50c"] = abs(last_price - 0.5)

        # Distance from nearest $0.10 round number
        nearest_round = round(last_price * 10) / 10.0
        feat["distance_from_round_number"] = abs(last_price - nearest_round)

        price_hist_1h = np.asarray(raw_data.get("price_history_1h", []), dtype=np.float64)
        if len(price_hist_1h) >= 2:
            velocity = np.diff(price_hist_1h)
            feat["price_velocity_1h"] = float(velocity[-1])
            if len(velocity) >= 2:
                feat["price_acceleration"] = float(np.diff(velocity)[-1])
            else:
                feat["price_acceleration"] = 0.0
        else:
            feat["price_velocity_1h"] = 0.0
            feat["price_acceleration"] = 0.0

        # ---- Historical resolution ----
        feat["historical_resolution_rate"] = float(
            raw_data.get("historical_resolution_rate", 0.5)
        )
        feat["category_resolution_rate"] = float(
            raw_data.get("category_resolution_rate", 0.5)
        )

        # ---- Orderbook depth ----
        bid_sizes = np.asarray(raw_data.get("bid_sizes", [0.0]), dtype=np.float64)
        ask_sizes = np.asarray(raw_data.get("ask_sizes", [0.0]), dtype=np.float64)

        total_bid = float(np.sum(bid_sizes))
        total_ask = float(np.sum(ask_sizes))
        feat["depth_ratio_bid_ask"] = (
            total_bid / total_ask if total_ask > _EPS else 1.0
        )

        # Top-5 level imbalance
        top5_bid = float(np.sum(bid_sizes[:5]))
        top5_ask = float(np.sum(ask_sizes[:5]))
        denom = top5_bid + top5_ask
        feat["depth_imbalance_5_levels"] = (
            (top5_bid - top5_ask) / denom if denom > _EPS else 0.0
        )

        feat["total_liquidity_usd"] = total_bid + total_ask

        # ---- Cross-market ----
        corr_prices = np.asarray(raw_data.get("correlated_prices", []), dtype=np.float64)
        num_corr = int(raw_data.get("num_correlated", len(corr_prices)))
        feat["num_correlated_markets"] = float(num_corr)

        if len(corr_prices) > 0:
            avg_corr = float(np.mean(corr_prices))
            feat["avg_correlated_price"] = avg_corr
            feat["correlated_price_divergence"] = abs(last_price - avg_corr)
        else:
            feat["avg_correlated_price"] = last_price
            feat["correlated_price_divergence"] = 0.0

        # ---- Time decay ----
        days_remaining = max(seconds_to_expiry / 86400.0, _EPS)
        feat["exponential_decay_factor"] = float(math.exp(-0.1 * days_remaining))
        feat["sqrt_time_factor"] = float(math.sqrt(days_remaining))
        feat["days_since_creation"] = (now - created_ts) / 86400.0

        # ---- Market quality ----
        feat["num_unique_traders_est"] = float(raw_data.get("num_unique_traders_est", 0.0))

        price_hist_7d = np.asarray(raw_data.get("price_history_7d", []), dtype=np.float64)
        if len(price_hist_7d) >= 2:
            feat["price_stability_24h"] = 1.0 - float(np.std(price_hist_7d[-24:], ddof=1))

            # Max drawdown over 7d
            cum_max = np.maximum.accumulate(price_hist_7d)
            drawdowns = (price_hist_7d - cum_max) / np.maximum(cum_max, _EPS)
            max_dd = float(np.min(drawdowns))
            feat["max_drawdown_7d"] = max_dd

            # Recovery rate: fraction of drawdown recovered at end
            dd_idx = int(np.argmin(drawdowns))
            if dd_idx < len(price_hist_7d) - 1 and max_dd < -_EPS:
                recovery = (price_hist_7d[-1] - price_hist_7d[dd_idx]) / abs(
                    max_dd * cum_max[dd_idx]
                )
                feat["recovery_rate"] = float(np.clip(recovery, 0.0, 1.0))
            else:
                feat["recovery_rate"] = 1.0
        else:
            feat["price_stability_24h"] = 1.0
            feat["max_drawdown_7d"] = 0.0
            feat["recovery_rate"] = 1.0

        return feat
