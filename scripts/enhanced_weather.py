#!/usr/bin/env python3
"""Enhanced Weather Forecaster for APEX V2.

Multi-source ensemble with calibrated probability estimation.

Sources (weighted by historical accuracy):
- NWS (Weather.gov) — official government forecast
- GFS (via Open-Meteo) — NOAA global model
- ECMWF (via Open-Meteo) — European model, often best
- HRRR (via Open-Meteo) — high-res, best for <24h
- Historical climatology — baseline prior

Key improvements over naive ensemble:
1. Historical accuracy weighting (better models get more weight)
2. Calibrated sigma from actual forecast error distributions
3. Regime detection (stable vs volatile weather patterns)
4. Bayesian probability with climatological prior
5. Only trades when edge exceeds confidence-adjusted threshold
"""

import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger()

DB_PATH = Path(__file__).parent.parent / "data" / "weather" / "weather_history.db"

# Forecast source default weights (overridden by historical accuracy)
DEFAULT_SOURCE_WEIGHTS = {
    "hrrr": 0.35,    # Best for <24h
    "ecmwf": 0.25,   # Best medium-range
    "nws": 0.20,     # Good all-around (uses MOS)
    "gfs": 0.15,     # Decent but less accurate
    "climatology": 0.05,  # Baseline prior
}

# Default forecast error sigma by lead time (°F)
# These are calibrated from NWS verification data
DEFAULT_SIGMA_BY_LEAD = {
    0: 1.5,    # Same day
    6: 1.8,    # 6 hours
    12: 2.2,   # 12 hours
    24: 3.0,   # 1 day
    48: 4.5,   # 2 days
    72: 5.5,   # 3 days
    168: 7.0,  # 7 days
}


@dataclass
class CalibratedForecast:
    """A calibrated temperature forecast with uncertainty."""
    city_key: str
    target_date: str
    # Point forecasts from each source
    source_forecasts: dict[str, float] = field(default_factory=dict)
    # Weighted ensemble mean
    ensemble_mean_f: float = 0.0
    # Calibrated sigma (forecast uncertainty in °F)
    calibrated_sigma: float = 3.0
    # Ensemble std (model disagreement)
    ensemble_std_f: float = 0.0
    # Number of contributing models
    n_models: int = 0
    # Confidence (0-1, higher = more certain)
    confidence: float = 0.5
    # Regime
    regime: str = "NORMAL"
    # Lead time in hours
    lead_hours: float = 24.0
    # Source weights used
    source_weights: dict[str, float] = field(default_factory=dict)
    # Climatological normal for this date
    climatology_mean: float = 75.0
    # Model names
    models: list = field(default_factory=list)


class EnhancedWeatherForecaster:
    """Multi-source ensemble forecaster with calibrated uncertainty."""

    def __init__(self):
        self._db_path = DB_PATH
        self._accuracy_cache: dict = {}
        self._climatology_cache: dict = {}
        self._load_cache()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _load_cache(self):
        """Load accuracy metrics and climatology into memory."""
        if not self._db_path.exists():
            return

        conn = self._get_conn()
        c = conn.cursor()

        # Load accuracy metrics
        try:
            c.execute("SELECT * FROM forecast_accuracy")
            cols = [d[0] for d in c.description]
            for row in c.fetchall():
                key = (row[0], row[1], row[2])  # city, source, bin
                self._accuracy_cache[key] = dict(zip(cols, row))
        except sqlite3.OperationalError:
            pass

        # Load climate normals from actual data
        try:
            c.execute("""
                SELECT city_key,
                       CAST(strftime('%m', target_date) AS INTEGER) as month,
                       CAST(strftime('%d', target_date) AS INTEGER) as day,
                       AVG(forecast_high_f),
                       COUNT(*)
                FROM forecasts WHERE source = 'actual'
                GROUP BY city_key, month, day
            """)
            for row in c.fetchall():
                self._climatology_cache[(row[0], row[1], row[2])] = {
                    "mean": row[3],
                    "n": row[4],
                }
        except sqlite3.OperationalError:
            pass

        conn.close()

    def get_source_weight(self, city_key: str, source: str, lead_hours: float) -> float:
        """Get accuracy-adjusted weight for a forecast source."""
        # Find the right lead-time bin
        bins = [(0, "0-6h"), (6, "6-12h"), (12, "12-24h"),
                (24, "24-48h"), (48, "48-72h"), (72, "72-168h")]
        bin_name = "72-168h"
        for threshold, name in bins:
            if lead_hours < threshold:
                break
            bin_name = name

        key = (city_key, source, bin_name)
        if key in self._accuracy_cache:
            acc = self._accuracy_cache[key]
            n = acc.get("n_samples", 0)
            if n >= 5:
                # Weight by inverse RMSE (better models get more weight)
                rmse = acc.get("rmse_f", 5.0)
                if rmse > 0:
                    return 1.0 / rmse

        # Fall back to defaults
        return DEFAULT_SOURCE_WEIGHTS.get(source, 0.1)

    def get_calibrated_sigma(self, city_key: str, lead_hours: float,
                              ensemble_std: float, n_models: int) -> float:
        """Get calibrated forecast uncertainty from historical accuracy data."""
        # Find the right lead-time bin
        bins = [(0, "0-6h"), (6, "6-12h"), (12, "12-24h"),
                (24, "24-48h"), (48, "48-72h"), (72, "72-168h")]
        bin_name = "72-168h"
        for threshold, name in bins:
            if lead_hours < threshold:
                break
            bin_name = name

        # Check for city-specific accuracy data
        best_sigma = None
        for source in ["nws", "gfs", "ecmwf", "hrrr"]:
            key = (city_key, source, bin_name)
            if key in self._accuracy_cache:
                acc = self._accuracy_cache[key]
                n = acc.get("n_samples", 0)
                if n >= 10:
                    rmse = acc.get("rmse_f", 0)
                    if rmse > 0:
                        if best_sigma is None or rmse < best_sigma:
                            best_sigma = rmse

        if best_sigma is not None:
            return best_sigma

        # Fall back to default sigma interpolated by lead time
        lead_keys = sorted(DEFAULT_SIGMA_BY_LEAD.keys())
        for i, lk in enumerate(lead_keys):
            if lead_hours <= lk:
                if i == 0:
                    return DEFAULT_SIGMA_BY_LEAD[lk]
                # Interpolate
                prev = lead_keys[i - 1]
                frac = (lead_hours - prev) / (lk - prev)
                return (DEFAULT_SIGMA_BY_LEAD[prev] * (1 - frac) +
                        DEFAULT_SIGMA_BY_LEAD[lk] * frac)

        return DEFAULT_SIGMA_BY_LEAD[lead_keys[-1]]

    def get_climatology(self, city_key: str, target_date: str) -> float | None:
        """Get climatological normal high for a city/date."""
        try:
            dt = datetime.strptime(target_date, "%Y-%m-%d")
            key = (city_key, dt.month, dt.day)
            if key in self._climatology_cache:
                return self._climatology_cache[key]["mean"]
        except (ValueError, KeyError):
            pass
        return None

    def detect_regime(self, forecasts: dict[str, float],
                      city_key: str, target_date: str) -> tuple[str, float]:
        """Detect weather regime: STABLE, NORMAL, VOLATILE.

        Returns (regime, regime_multiplier) where multiplier
        adjusts sigma (1.0 = no change, >1.0 = wider, <1.0 = tighter).
        """
        values = list(forecasts.values())
        if len(values) < 2:
            return "NORMAL", 1.0

        spread = max(values) - min(values)
        std = float(np.std(values))

        # Check climatology for context
        clim = self.get_climatology(city_key, target_date)
        if clim is not None:
            avg_forecast = float(np.mean(values))
            clim_deviation = abs(avg_forecast - clim)
        else:
            clim_deviation = 0

        # STABLE: models agree closely, close to climatology
        if spread < 2.0 and std < 1.0 and clim_deviation < 5:
            return "STABLE", 0.7  # Tighter sigma

        # VOLATILE: models disagree or far from climatology
        if spread > 6.0 or std > 3.0 or clim_deviation > 15:
            return "VOLATILE", 1.5  # Wider sigma

        # Slightly uncertain
        if spread > 4.0 or std > 2.0:
            return "UNCERTAIN", 1.2

        return "NORMAL", 1.0

    def build_forecast(self, city_key: str, target_date: str,
                       source_forecasts: dict[str, float],
                       lead_hours: float = 24.0) -> CalibratedForecast:
        """Build a calibrated ensemble forecast from multiple sources.

        Parameters
        ----------
        city_key : str
            City identifier (nyc, chi, la, etc.)
        target_date : str
            Target date in YYYY-MM-DD format
        source_forecasts : dict
            Forecast high temps by source, e.g. {"nws": 85, "gfs": 84, "ecmwf": 86}
        lead_hours : float
            Hours until the forecast target (affects sigma and weighting)
        """
        if not source_forecasts:
            # Fall back to climatology
            clim = self.get_climatology(city_key, target_date)
            if clim is not None:
                return CalibratedForecast(
                    city_key=city_key,
                    target_date=target_date,
                    source_forecasts={"climatology": clim},
                    ensemble_mean_f=clim,
                    calibrated_sigma=8.0,
                    ensemble_std_f=0,
                    n_models=1,
                    confidence=0.1,
                    regime="UNKNOWN",
                    lead_hours=lead_hours,
                    source_weights={"climatology": 1.0},
                    climatology_mean=clim,
                    models=["climatology"],
                )
            return CalibratedForecast(
                city_key=city_key, target_date=target_date,
                ensemble_mean_f=75.0, calibrated_sigma=10.0,
            )

        # Detect regime
        regime, regime_mult = self.detect_regime(
            source_forecasts, city_key, target_date
        )

        # Get accuracy-adjusted weights for each source
        weights = {}
        for source in source_forecasts:
            w = self.get_source_weight(city_key, source, lead_hours)
            weights[source] = w

        # Normalize weights
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}
        else:
            n = len(source_forecasts)
            weights = {k: 1.0 / n for k in source_forecasts}

        # Weighted ensemble mean
        ensemble_mean = sum(
            source_forecasts[s] * weights[s] for s in source_forecasts
        )

        # Ensemble std (model disagreement)
        if len(source_forecasts) > 1:
            variance = sum(
                weights[s] * (source_forecasts[s] - ensemble_mean) ** 2
                for s in source_forecasts
            )
            ensemble_std = math.sqrt(variance)
        else:
            ensemble_std = 0.0

        # Calibrated sigma from historical accuracy
        base_sigma = self.get_calibrated_sigma(
            city_key, lead_hours, ensemble_std, len(source_forecasts)
        )

        # Adjust sigma by regime
        calibrated_sigma = base_sigma * regime_mult

        # Adjust sigma by model agreement
        # More models agreeing = tighter sigma
        n_models = len(source_forecasts)
        if n_models >= 4 and ensemble_std < 2:
            calibrated_sigma *= 0.85  # Boost confidence
        elif n_models < 2:
            calibrated_sigma *= 1.3  # Less confidence with few models

        # Confidence from sigma (inverse relationship)
        # sigma=1.5°F → conf=0.95, sigma=3°F → conf=0.8, sigma=6°F → conf=0.5
        confidence = max(0.1, min(0.99, 1.0 - (calibrated_sigma - 1.0) / 10.0))

        # Blend confidence with model agreement
        agreement_bonus = 0
        if ensemble_std < 1.5:
            agreement_bonus = 0.1
        elif ensemble_std > 4:
            agreement_bonus = -0.15

        confidence = max(0.1, min(0.99, confidence + agreement_bonus))

        # Climatology
        clim = self.get_climatology(city_key, target_date) or ensemble_mean

        # Build model list
        model_names = []
        model_details = []
        for source, temp in source_forecasts.items():
            model_names.append(source)
            model_details.append(type("M", (), {
                "model": source,
                "high_f": temp,
                "weight": weights.get(source, 0),
            })())

        return CalibratedForecast(
            city_key=city_key,
            target_date=target_date,
            source_forecasts=source_forecasts,
            ensemble_mean_f=round(ensemble_mean, 1),
            calibrated_sigma=round(calibrated_sigma, 2),
            ensemble_std_f=round(ensemble_std, 2),
            n_models=n_models,
            confidence=round(confidence, 3),
            regime=regime,
            lead_hours=lead_hours,
            source_weights=weights,
            climatology_mean=clim,
            models=model_details,
        )

    def estimate_probability(self, forecast: CalibratedForecast,
                             threshold: float, direction: str,
                             range_low: float = 0,
                             range_high: float = 0) -> float:
        """Estimate calibrated probability for a temperature market.

        Uses calibrated sigma instead of raw ensemble std.
        Applies Bayesian shrinkage toward climatology prior.
        """
        mu = forecast.ensemble_mean_f
        sigma = forecast.calibrated_sigma

        # Bayesian shrinkage toward climatology for low-confidence forecasts
        # Weight: confidence determines how much we trust the forecast vs climatology
        clim = forecast.climatology_mean
        if forecast.confidence < 0.7 and clim > 0:
            # Shrink toward climatology
            shrink = 1.0 - forecast.confidence  # 0.3 at conf=0.7
            mu = mu * (1 - shrink) + clim * shrink

        if direction == "range" and range_high > range_low:
            # P(range_low <= X <= range_high) using normal CDF
            z_high = (range_high - mu) / sigma
            z_low = (range_low - mu) / sigma
            phi_high = 0.5 * (1 + math.erf(z_high / math.sqrt(2)))
            phi_low = 0.5 * (1 + math.erf(z_low / math.sqrt(2)))
            prob = phi_high - phi_low

        elif direction == "above":
            z = (mu - threshold) / sigma
            prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))

        elif direction == "below":
            z = (threshold - mu) / sigma
            prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))

        else:
            prob = 0.5

        return max(0.01, min(0.99, prob))

    def should_trade(self, forecast: CalibratedForecast,
                     market_price: float, threshold: float,
                     direction: str, range_low: float = 0,
                     range_high: float = 0) -> dict | None:
        """Decide whether to trade a weather market.

        Returns signal dict if we should trade, None otherwise.

        Trade criteria:
        1. Edge > minimum (confidence-adjusted)
        2. Confidence > minimum
        3. Not in VOLATILE regime (unless edge is huge)
        4. At least 2 forecast sources available
        5. Forecast not in noise zone near threshold (Chicago trap)
        """
        # Minimum edge scales with confidence
        # High confidence: min edge = 0.05
        # Low confidence: min edge = 0.15
        min_edge = 0.05 + 0.10 * (1.0 - forecast.confidence)

        # VOLATILE regime: raise bar significantly
        if forecast.regime == "VOLATILE":
            min_edge *= 2.0

        # Need at least 2 models
        if forecast.n_models < 2:
            return None

        # ---- PROXIMITY FILTER (Chicago trap avoidance) ----
        # If the forecast is too close to the threshold, skip.
        # When the ensemble mean is within ~1 sigma of the threshold,
        # the trade is essentially a coin flip — the model can't
        # confidently say which side the outcome will land on.
        # This is exactly what killed us on Chicago (5 trades, -$47).
        sigma = forecast.calibrated_sigma
        dist_from_threshold = abs(forecast.ensemble_mean_f - threshold)
        threshold_distance_sigma = dist_from_threshold / max(sigma, 0.5)

        # For range markets, use distance from range center
        if direction == "range" and range_high > range_low:
            range_mid = (range_low + range_high) / 2
            dist_from_threshold = abs(forecast.ensemble_mean_f - range_mid)
            threshold_distance_sigma = dist_from_threshold / max(sigma, 0.5)

        # Skip if forecast is in the noise zone (< 1.0 sigma from threshold)
        # and the regime is anything less than STABLE
        if threshold_distance_sigma < 1.0:
            if forecast.regime != "STABLE":
                logger.debug("weather.proximity_skip",
                            ensemble=forecast.ensemble_mean_f,
                            threshold=threshold,
                            dist_sigma=round(threshold_distance_sigma, 2),
                            regime=forecast.regime)
                return None
            # STABLE regime but very close — raise min edge
            min_edge = max(min_edge, 0.12)

        # Estimate true probability
        true_prob = self.estimate_probability(
            forecast, threshold, direction, range_low, range_high
        )

        edge = true_prob - market_price

        if abs(edge) < min_edge:
            return None

        # ---- SELL PRICE CEILING ----
        # Hard cap at $0.85. Above this, one loss erases 4+ wins.
        # Real data: $0.60-$0.80 avg +$37/trade (sweet spot).
        # Above $0.80 avg -$11/trade (net loser).
        SELL_CEILING = 0.85
        if edge < 0 and market_price > SELL_CEILING:
            logger.debug("weather.sell_ceiling_skip",
                        market_price=market_price,
                        ceiling=SELL_CEILING,
                        edge=round(edge, 4))
            return None

        # ---- MIN BUY PRICE FILTER ----
        # Don't buy contracts priced below $0.05 — likely noise
        if edge > 0 and market_price < 0.05:
            return None

        # ---- EDGE-SCALED POSITION SIZING ----
        # Bigger bets on bigger edges. The first hour showed $22 avg win
        # vs $27 avg loss. To fix this, we size UP on high-edge trades
        # (which have higher WR) and size DOWN on marginal trades.
        kelly = 0.25
        if edge > 0:
            # BUY: edge / (1/price - 1) simplified
            odds = market_price / (1 - market_price) if market_price < 1 else 1
            size_pct = kelly * edge / max(odds, 0.01)
        else:
            # SELL: edge / (1/(1-price) - 1) simplified
            odds = (1 - market_price) / market_price if market_price > 0 else 1
            size_pct = kelly * abs(edge) / max(odds, 0.01)

        # Cap at 12% and scale by confidence
        size_pct = min(size_pct, 0.12)
        size_pct *= forecast.confidence

        # Scale by model agreement
        if forecast.ensemble_std_f < 2:
            size_pct *= 1.15
        elif forecast.ensemble_std_f > 5:
            size_pct *= 0.6

        # ---- EDGE MAGNITUDE SCALING ----
        # High-edge trades (>15%) get a size boost — these are the
        # Austin/Denver type trades that made us $50-94 each.
        # Low-edge trades (5-10%) get trimmed — these are the marginal
        # trades that gave us -$30 losses.
        abs_edge = abs(edge)
        if abs_edge >= 0.20:
            size_pct *= 1.4   # Huge edge — max confidence
        elif abs_edge >= 0.15:
            size_pct *= 1.2   # Strong edge — boost
        elif abs_edge <= 0.07:
            size_pct *= 0.7   # Marginal edge — trim

        # ---- PROXIMITY SCALING ----
        # Trades where forecast is far from threshold are more reliable
        if threshold_distance_sigma >= 2.5:
            size_pct *= 1.15  # Very clear signal
        elif threshold_distance_sigma < 1.5:
            size_pct *= 0.85  # Somewhat close — be conservative

        # ---- SELL PRICE SIZING CURVE ----
        # Concentrate capital in the $0.60-$0.80 sweet spot.
        # Real data: $0.60-$0.80 avg +$37/trade (100% WR).
        # Above $0.80 avg -$11/trade (net loser).
        if edge < 0:
            if 0.60 <= market_price <= 0.80:
                size_pct *= 1.15  # Sweet spot — boost
            elif market_price > 0.80:
                size_pct *= 0.75  # Above sweet spot — trim
            elif market_price < 0.50:
                size_pct *= 0.80  # Low price — small edge per dollar

        # Minimum position size
        if size_pct < 0.02:
            return None

        direction_trade = "BUY" if edge > 0 else "SELL"

        return {
            "direction": direction_trade,
            "edge": round(edge, 4),
            "true_prob": round(true_prob, 4),
            "market_price": round(market_price, 4),
            "size_pct": round(size_pct, 4),
            "confidence": forecast.confidence,
            "regime": forecast.regime,
            "ensemble_mean_f": forecast.ensemble_mean_f,
            "ensemble_std_f": forecast.ensemble_std_f,
            "calibrated_sigma": forecast.calibrated_sigma,
            "n_models": forecast.n_models,
            "source_weights": forecast.source_weights,
            "climatology_mean": forecast.climatology_mean,
            "threshold_distance_sigma": round(threshold_distance_sigma, 2),
            "abs_edge": round(abs_edge, 4),
            "reason": (
                f"Ensemble {forecast.ensemble_mean_f}°F "
                f"(σ={forecast.calibrated_sigma:.1f}) "
                f"vs {threshold}°F ({direction}), "
                f"conf {forecast.confidence:.0%}, edge {edge:+.1%}, "
                f"regime={forecast.regime}"
            ),
        }


def get_forecaster() -> EnhancedWeatherForecaster:
    """Get or create the singleton forecaster."""
    if not hasattr(get_forecaster, "_instance"):
        get_forecaster._instance = EnhancedWeatherForecaster()
    return get_forecaster._instance
