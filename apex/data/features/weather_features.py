"""
Weather Features (15 features)

Ensemble forecast statistics, model agreement metrics, climatology
comparisons, temperature trends, extreme-event probabilities, and
calibration/reliability scores.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from apex.data.features.builder import FeatureExtractor

_EPS = 1e-12


class WeatherFeatureExtractor(FeatureExtractor):
    """Computes 15 weather-related features.

    Expected keys in *raw_data*::

        # Ensemble forecast members (e.g., GFS/ECMWF ensembles)
        ensemble_temps        : list[float]   # temperature forecasts from N models
        threshold_temp        : float         # the contract strike / threshold

        # Climatology
        climatology_mean      : float         # long-run average for this day/location
        climatology_record_hi : float
        climatology_record_lo : float

        # Recent forecast skill
        recent_forecast_errors : list[float]  # signed errors of last N forecasts

        # Temperature trend (daily observations)
        daily_temps_7d        : list[float]   # observed daily temps, last 7 days

        # Source metadata
        num_sources           : int
        source_reliability    : list[float]   # per-source reliability score (0-1)

        # Forecast bias
        forecast_bias_estimate : float        # estimated additive bias
    """

    _NAMES: list[str] = [
        # Forecast (3)
        "ensemble_mean_temp",
        "ensemble_spread",
        "ensemble_median",
        # Model agreement (2)
        "num_models_above_threshold",
        "model_agreement_pct",
        # Historical (3)
        "climatology_baseline",
        "forecast_vs_climatology",
        "recent_forecast_mae",
        # Trends (2)
        "temp_trend_7d",
        "warming_cooling_rate",
        # Extreme (3)
        "distance_to_record_high",
        "distance_to_record_low",
        "extreme_probability",
        # Calibration (2)
        "forecast_bias_correction",
        "source_reliability_score",
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

        ensemble = np.asarray(raw_data.get("ensemble_temps", []), dtype=np.float64)
        threshold = float(raw_data.get("threshold_temp", 0.0))

        # ---- Forecast ----
        if len(ensemble) > 0:
            feat["ensemble_mean_temp"] = float(np.mean(ensemble))
            feat["ensemble_spread"] = float(np.std(ensemble, ddof=1)) if len(ensemble) > 1 else 0.0
            feat["ensemble_median"] = float(np.median(ensemble))
        else:
            feat["ensemble_mean_temp"] = 0.0
            feat["ensemble_spread"] = 0.0
            feat["ensemble_median"] = 0.0

        # ---- Model agreement ----
        if len(ensemble) > 0:
            above = np.sum(ensemble > threshold)
            feat["num_models_above_threshold"] = float(above)
            feat["model_agreement_pct"] = float(max(above, len(ensemble) - above)) / len(ensemble)
        else:
            feat["num_models_above_threshold"] = 0.0
            feat["model_agreement_pct"] = 0.5

        # ---- Historical / climatology ----
        clim_mean = float(raw_data.get("climatology_mean", 0.0))
        feat["climatology_baseline"] = clim_mean
        feat["forecast_vs_climatology"] = feat["ensemble_mean_temp"] - clim_mean

        recent_errors = np.asarray(
            raw_data.get("recent_forecast_errors", []), dtype=np.float64
        )
        if len(recent_errors) > 0:
            feat["recent_forecast_mae"] = float(np.mean(np.abs(recent_errors)))
        else:
            feat["recent_forecast_mae"] = 0.0

        # ---- Trends ----
        daily_temps = np.asarray(raw_data.get("daily_temps_7d", []), dtype=np.float64)
        if len(daily_temps) >= 2:
            from scipy import stats as sp_stats

            x = np.arange(len(daily_temps), dtype=np.float64)
            slope, _, _, _, _ = sp_stats.linregress(x, daily_temps)
            feat["temp_trend_7d"] = float(slope)
            feat["warming_cooling_rate"] = float(slope)  # degrees per day
        else:
            feat["temp_trend_7d"] = 0.0
            feat["warming_cooling_rate"] = 0.0

        # ---- Extreme ----
        record_hi = float(raw_data.get("climatology_record_hi", 50.0))
        record_lo = float(raw_data.get("climatology_record_lo", -30.0))
        ens_mean = feat["ensemble_mean_temp"]

        feat["distance_to_record_high"] = record_hi - ens_mean
        feat["distance_to_record_low"] = ens_mean - record_lo

        # Extreme probability: fraction of ensemble exceeding either record
        if len(ensemble) > 0:
            extreme_count = np.sum((ensemble > record_hi) | (ensemble < record_lo))
            feat["extreme_probability"] = float(extreme_count) / len(ensemble)
        else:
            feat["extreme_probability"] = 0.0

        # ---- Calibration ----
        feat["forecast_bias_correction"] = float(
            raw_data.get("forecast_bias_estimate", 0.0)
        )

        source_rel = np.asarray(
            raw_data.get("source_reliability", []), dtype=np.float64
        )
        if len(source_rel) > 0:
            feat["source_reliability_score"] = float(np.mean(source_rel))
        else:
            feat["source_reliability_score"] = 0.5

        return feat
