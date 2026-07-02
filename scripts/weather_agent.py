#!/usr/bin/env python3
"""Weather Strategy Agent for APEX V2 — Enhanced Ensemble Edition.

Upgrades:
1. ENSEMBLE FORECASTS: NWS + GFS + ECMWF + HRRR combined
2. CALIBRATED PROBABILITIES: Historical accuracy-based sigma
3. REGIME DETECTION: Stable vs volatile weather patterns
4. BAYESIAN SHRINKAGE: Toward climatology when uncertain
5. RANGE MARKET MODELING: Truncated normal for X-Y°F ranges
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import math as _math
import structlog

from enhanced_weather import EnhancedWeatherForecaster, get_forecaster

logger = structlog.get_logger()

ACCURACY_FILE = Path(__file__).parent.parent / "data" / "weather_accuracy.json"

# ---------------------------------------------------------------------------
# City configuration
# ---------------------------------------------------------------------------

CITY_CONFIG = {
    "KXHIGHNY":  {"name": "New York",       "lat": 40.7128, "lon": -74.0060},
    "KXHIGHCHI": {"name": "Chicago",        "lat": 41.8781, "lon": -87.6298},
    "KXHIGHLA":  {"name": "Los Angeles",    "lat": 34.0522, "lon": -118.2437},
    "KXHIGHMIA": {"name": "Miami",          "lat": 25.7617, "lon": -80.1918},
    "KXHIGHDC":  {"name": "Washington DC",  "lat": 38.9072, "lon": -77.0369},
    "KXHIGHHOU": {"name": "Houston",        "lat": 29.7604, "lon": -95.3698},
    "KXHIGHDAL": {"name": "Dallas",         "lat": 32.7767, "lon": -96.7970},
    "KXHIGHDEN": {"name": "Denver",         "lat": 39.7392, "lon": -104.9903},
    "KXHIGHPHX": {"name": "Phoenix",        "lat": 33.4484, "lon": -112.0740},
    "KXHIGHATL": {"name": "Atlanta",        "lat": 33.7490, "lon": -84.3880},
    "KXHIGHSF":  {"name": "San Francisco",  "lat": 37.7749, "lon": -122.4194},
    "KXHIGHBOS": {"name": "Boston",         "lat": 42.3601, "lon": -71.0589},
    "KXHIGHSEA": {"name": "Seattle",        "lat": 47.6062, "lon": -122.3321},
    "KXHIGHAUS": {"name": "Austin",         "lat": 30.2672, "lon": -97.7431},
}

# Forecast confidence decays with days out
CONFIDENCE_BY_DAYS = {
    0: 0.98, 1: 0.93, 2: 0.82, 3: 0.68, 4: 0.55, 5: 0.45,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelForecast:
    """A single model's forecast for a city."""
    model: str          # "NWS", "GFS", "ECMWF"
    high_f: float
    hours_out: float    # hours until the forecast target period


@dataclass
class EnsembleForecast:
    """Combined forecast from multiple models for a city."""
    city_key: str
    city_name: str
    # Ensemble stats
    mean_high_f: float          # Weighted mean across models
    std_f: float                # Standard deviation across models (disagreement)
    confidence: float           # 0-1, based on model agreement + days out
    n_models: int               # How many models contributed
    models: list[ModelForecast] = field(default_factory=list)
    # Metadata
    forecast_time: str = ""
    days_out: int = 0
    # Per-model raw values
    nws_high: Optional[float] = None
    gfs_high: Optional[float] = None
    ecmwf_high: Optional[float] = None


# ---------------------------------------------------------------------------
# Ensemble forecast fetcher
# ---------------------------------------------------------------------------

class WeatherAgent:
    """Fetches ensemble weather forecasts and generates trading signals."""

    # Model weights for ensemble (NWS is best for US, ECMWF is best globally)
    MODEL_WEIGHTS = {"NWS": 0.45, "ECMWF": 0.35, "GFS": 0.20}

    def __init__(self):
        self.forecasts: dict[str, EnsembleForecast] = {}
        self._forecast_cache_ts: float = 0
        self._grid_cache: dict[str, tuple[str, str, str]] = {}
        self._accuracy_data: list[dict] = []
        self._openmeteo_cache: dict[str, dict] = {}
        self._openmeteo_cache_ts: float = 0
        self._load_accuracy()

    # ------------------------------------------------------------------
    # Accuracy tracking
    # ------------------------------------------------------------------

    def _load_accuracy(self):
        try:
            if ACCURACY_FILE.exists():
                with open(ACCURACY_FILE) as f:
                    self._accuracy_data = json.load(f)
        except Exception:
            self._accuracy_data = []

    def _save_accuracy(self):
        try:
            ACCURACY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(ACCURACY_FILE, "w") as f:
                json.dump(self._accuracy_data[-1000:], f)
        except Exception as e:
            logger.warning("weather.accuracy_save_failed", error=str(e))

    def _get_city_accuracy(self, city_key: str) -> float:
        entries = [e for e in self._accuracy_data if e.get("city_key") == city_key]
        if len(entries) < 5:
            return 0.0
        errors = [abs(e["forecast"] - e["actual"]) for e in entries[-50:]]
        mae = np.mean(errors)
        return max(0, 1 - mae / 10)

    def record_outcome(self, city_key: str, forecast_high: float, actual_high: float):
        self._accuracy_data.append({
            "city_key": city_key,
            "forecast": forecast_high,
            "actual": actual_high,
            "error": abs(forecast_high - actual_high),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._save_accuracy()

    # ------------------------------------------------------------------
    # NWS forecast (existing, best for US)
    # ------------------------------------------------------------------

    async def _get_grid(self, client: httpx.AsyncClient, city_key: str) -> tuple[str, str, str]:
        if city_key in self._grid_cache:
            return self._grid_cache[city_key]
        cfg = CITY_CONFIG[city_key]
        resp = await client.get(
            f"https://api.weather.gov/points/{cfg['lat']},{cfg['lon']}",
            headers={"User-Agent": "APEX-Trading-Bot/1.0 (contact@titanholdings.ai)"},
        )
        resp.raise_for_status()
        props = resp.json()["properties"]
        grid_id = props["gridId"]
        grid_x = str(props["gridX"])
        grid_y = str(props["gridY"])
        self._grid_cache[city_key] = (grid_id, grid_x, grid_y)
        return grid_id, grid_x, grid_y

    async def _fetch_nws(self, client: httpx.AsyncClient, city_key: str) -> Optional[ModelForecast]:
        """Fetch NWS/NOAA forecast for a city."""
        try:
            grid_id, grid_x, grid_y = await self._get_grid(client, city_key)
            resp = await client.get(
                f"https://api.weather.gov/gridpoints/{grid_id}/{grid_x},{grid_y}/forecast",
                headers={"User-Agent": "APEX-Trading-Bot/1.0 (contact@titanholdings.ai)"},
            )
            resp.raise_for_status()
            periods = resp.json().get("properties", {}).get("periods", [])
            if not periods:
                return None

            for i, period in enumerate(periods[:6]):
                if period.get("isDaytime", False):
                    return ModelForecast(
                        model="NWS",
                        high_f=float(period["temperature"]),
                        hours_out=i * 12,  # rough: each period is ~12h
                    )
            # Fallback
            return ModelForecast(model="NWS", high_f=float(periods[0]["temperature"]), hours_out=0)
        except Exception as e:
            logger.debug("weather.nws_failed", city=city_key, error=str(e))
            return None

    # ------------------------------------------------------------------
    # Open-Meteo: GFS + ECMWF (free, no key needed)
    # ------------------------------------------------------------------

    async def _fetch_openmeteo(self, client: httpx.AsyncClient, city_key: str) -> list[ModelForecast]:
        """Fetch GFS and ECMWF forecasts from Open-Meteo API.

        Open-Meteo provides free access to multiple weather models.
        We request daily max temperature for the next 3 days.
        """
        cfg = CITY_CONFIG[city_key]
        results = []

        # GFS model
        try:
            resp = await client.get(
                "https://api.open-meteo.com/v1/gfs",
                params={
                    "latitude": cfg["lat"],
                    "longitude": cfg["lon"],
                    "daily": "temperature_2m_max",
                    "temperature_unit": "fahrenheit",
                    "forecast_days": 3,
                    "timezone": "America/New_York",
                },
            )
            if resp.status_code == 200:
                data = resp.json().get("daily", {})
                temps = data.get("temperature_2m_max", [])
                if temps:
                    results.append(ModelForecast(
                        model="GFS", high_f=float(temps[0]), hours_out=0
                    ))
                    if len(temps) > 1:
                        results.append(ModelForecast(
                            model="GFS", high_f=float(temps[1]), hours_out=24
                        ))
        except Exception as e:
            logger.debug("weather.gfs_failed", city=city_key, error=str(e))

        # ECMWF model
        try:
            resp = await client.get(
                "https://api.open-meteo.com/v1/ecmwf",
                params={
                    "latitude": cfg["lat"],
                    "longitude": cfg["lon"],
                    "daily": "temperature_2m_max",
                    "temperature_unit": "fahrenheit",
                    "forecast_days": 3,
                    "timezone": "America/New_York",
                },
            )
            if resp.status_code == 200:
                data = resp.json().get("daily", {})
                temps = data.get("temperature_2m_max", [])
                if temps:
                    results.append(ModelForecast(
                        model="ECMWF", high_f=float(temps[0]), hours_out=0
                    ))
                    if len(temps) > 1:
                        results.append(ModelForecast(
                            model="ECMWF", high_f=float(temps[1]), hours_out=24
                        ))
        except Exception as e:
            logger.debug("weather.ecmwf_failed", city=city_key, error=str(e))

        # HRRR model (best for <24h, high resolution)
        try:
            resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": cfg["lat"],
                    "longitude": cfg["lon"],
                    "hourly": "temperature_2m",
                    "temperature_unit": "fahrenheit",
                    "forecast_days": 2,
                    "models": "hrrr_seamless",
                    "timezone": "America/New_York",
                },
            )
            if resp.status_code == 200:
                hourly = resp.json().get("hourly", {})
                times = hourly.get("time", [])
                temps = hourly.get("temperature_2m", [])
                # Group by date and find daily max
                daily_max = {}
                for t_str, temp in zip(times, temps):
                    if temp is not None:
                        d = t_str[:10]
                        if d not in daily_max or temp > daily_max[d]:
                            daily_max[d] = temp
                sorted_dates = sorted(daily_max.keys())
                if sorted_dates:
                    results.append(ModelForecast(
                        model="HRRR", high_f=float(daily_max[sorted_dates[0]]),
                        hours_out=0,
                    ))
                    if len(sorted_dates) > 1:
                        results.append(ModelForecast(
                            model="HRRR", high_f=float(daily_max[sorted_dates[1]]),
                            hours_out=24,
                        ))
        except Exception as e:
            logger.debug("weather.hrrr_failed", city=city_key, error=str(e))

        return results

    # ------------------------------------------------------------------
    # Ensemble combination
    # ------------------------------------------------------------------

    def _build_ensemble(self, city_key: str, models: list[ModelForecast]) -> Optional[EnsembleForecast]:
        """Combine multiple model forecasts into a single ensemble forecast.

        Uses weighted average where weights are:
        - NWS: 0.45 (best for US cities, official source)
        - ECMWF: 0.35 (best global model)
        - GFS: 0.20 (decent, widely available)

        Model agreement (low std) = high confidence.
        Model disagreement (high std) = lower confidence, wider sigma.
        """
        if not models:
            return None

        # Group by hours_out (same target period)
        by_horizon: dict[float, list[ModelForecast]] = {}
        for m in models:
            # Round to nearest day
            horizon = round(m.hours_out / 24) * 24
            by_horizon.setdefault(horizon, []).append(m)

        # Use the nearest horizon with at least 1 model
        best_horizon = min(by_horizon.keys())
        horizon_models = by_horizon[best_horizon]

        # Weighted mean
        weights = np.array([self.MODEL_WEIGHTS.get(m.model, 0.1) for m in horizon_models])
        temps = np.array([m.high_f for m in horizon_models])

        # Normalize weights
        weights = weights / weights.sum()

        mean_high = float(np.average(temps, weights=weights))
        std = float(np.std(temps)) if len(temps) > 1 else 2.0  # Default 2°F if single model

        # Confidence: based on days out + model agreement
        days_out = int(best_horizon / 24)
        base_confidence = CONFIDENCE_BY_DAYS.get(days_out, 0.40)

        # Model agreement bonus: if models agree within 2°F, boost confidence
        if std < 1.5:
            agreement_bonus = 0.05  # Very high agreement
        elif std < 3.0:
            agreement_bonus = 0.0   # Normal agreement
        elif std < 5.0:
            agreement_bonus = -0.05  # Some disagreement
        else:
            agreement_bonus = -0.15  # Strong disagreement — be cautious

        # Historical accuracy adjustment
        city_accuracy = self._get_city_accuracy(city_key)
        accuracy_adj = 0.0
        if city_accuracy > 0:
            accuracy_adj = 0.05 * (city_accuracy - 0.5)  # +/- 2.5% based on history

        confidence = np.clip(base_confidence + agreement_bonus + accuracy_adj, 0.2, 0.99)

        # Parse per-model values
        nws = next((m.high_f for m in horizon_models if m.model == "NWS"), None)
        gfs = next((m.high_f for m in horizon_models if m.model == "GFS"), None)
        ecmwf = next((m.high_f for m in horizon_models if m.model == "ECMWF"), None)

        return EnsembleForecast(
            city_key=city_key,
            city_name=CITY_CONFIG[city_key]["name"],
            mean_high_f=round(mean_high, 1),
            std_f=round(std, 2),
            confidence=round(confidence, 3),
            n_models=len(horizon_models),
            models=horizon_models,
            forecast_time=datetime.now(timezone.utc).isoformat(),
            days_out=days_out,
            nws_high=nws,
            gfs_high=gfs,
            ecmwf_high=ecmwf,
        )

    # ------------------------------------------------------------------
    # Combined fetch
    # ------------------------------------------------------------------

    async def fetch_all_forecasts(self, force: bool = False) -> dict[str, EnsembleForecast]:
        """Fetch ensemble forecasts (NWS + GFS + ECMWF) for all cities."""
        now = time.time()
        # Cache for 15 min (faster than before — catch forecast updates quicker)
        if not force and (now - self._forecast_cache_ts) < 900:
            return self.forecasts

        async with httpx.AsyncClient(timeout=20) as client:
            # Fetch all sources concurrently
            tasks = []
            for city_key in CITY_CONFIG:
                tasks.append(self._fetch_city_ensemble(client, city_key))
            results = await asyncio.gather(*tasks, return_exceptions=True)

        forecasts = {}
        for city_key, result in zip(CITY_CONFIG.keys(), results):
            if isinstance(result, EnsembleForecast):
                forecasts[city_key] = result
            elif isinstance(result, Exception):
                logger.debug("weather.ensemble_failed", city=city_key, error=str(result))

        self.forecasts = forecasts
        self._forecast_cache_ts = now

        logger.info(
            "weather.ensemble_fetched",
            n=len(forecasts),
            avg_models=np.mean([f.n_models for f in forecasts.values()]) if forecasts else 0,
        )
        return forecasts

    async def _fetch_city_ensemble(self, client: httpx.AsyncClient, city_key: str) -> Optional[EnsembleForecast]:
        """Fetch all models for a single city and build ensemble."""
        models: list[ModelForecast] = []

        # Fetch NWS + Open-Meteo concurrently
        nws_task = self._fetch_nws(client, city_key)
        om_task = self._fetch_openmeteo(client, city_key)

        nws_result, om_results = await asyncio.gather(nws_task, om_task, return_exceptions=True)

        if isinstance(nws_result, ModelForecast):
            models.append(nws_result)
        if isinstance(om_results, list):
            models.extend(om_results)

        return self._build_ensemble(city_key, models)

    # ------------------------------------------------------------------
    # Market parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_temperature_threshold(question: str) -> Optional[tuple[float, str]]:
        """Parse a Kalshi weather market question.

        Returns (threshold_f, direction) where direction is 'above', 'below', or 'range'.
        """
        import re
        q = question.lower().replace("**", "")

        # ">X°" or "above X°"
        above = re.search(r'(?:above|over)\s*(\d{2,3})\s*°?f?', q)
        if not above:
            above = re.search(r'>\s*(\d{2,3})\s*°?', q)
        if above:
            return float(above.group(1)), "above"

        # "<X°" or "below X°"
        below = re.search(r'(?:below|under)\s*(\d{2,3})\s*°?f?', q)
        if not below:
            below = re.search(r'<\s*(\d{2,3})\s*°?', q)
        if below:
            return float(below.group(1)), "below"

        # "X-Y°" range
        range_match = re.search(r'(\d{2,3})\s*-\s*(\d{2,3})\s*°', q)
        if range_match:
            low = float(range_match.group(1))
            high = float(range_match.group(2))
            return (low + high) / 2, "range"

        return None

    @staticmethod
    def identify_city_key(question: str) -> Optional[str]:
        q = question.lower().replace("**", "")
        city_aliases = {
            "KXHIGHNY": ["new york", "nyc", "manhattan"],
            "KXHIGHCHI": ["chicago", "chi "],
            "KXHIGHLA": ["los angeles", "la ", "l.a."],
            "KXHIGHMIA": ["miami"],
            "KXHIGHDC": ["washington", "dc ", "d.c."],
            "KXHIGHHOU": ["houston"],
            "KXHIGHDAL": ["dallas"],
            "KXHIGHDEN": ["denver"],
            "KXHIGHPHX": ["phoenix"],
            "KXHIGHATL": ["atlanta"],
            "KXHIGHSF": ["san francisco", "sf "],
            "KXHIGHBOS": ["boston"],
            "KXHIGHSEA": ["seattle"],
            "KXHIGHAUS": ["austin"],
        }
        for key, aliases in city_aliases.items():
            for alias in aliases:
                if alias in q:
                    return key
        return None

    # ------------------------------------------------------------------
    # IMPROVEMENT 3: Range market probability with truncated normal
    # ------------------------------------------------------------------

    def _estimate_probability(
        self, mean_high: float, threshold: float, direction: str,
        confidence: float, std: float = 2.0, is_range: bool = False,
        range_low: float = 0, range_high: float = 0,
    ) -> float:
        """Estimate probability using ensemble statistics.

        For range markets (X-Y°F), uses a TRUNCATED NORMAL distribution:
        - The forecast is a point estimate with uncertainty (sigma from ensemble std)
        - We compute P(low <= actual <= high) given the forecast distribution
        - This is much more accurate than treating ranges as simple thresholds

        For above/below markets, uses standard normal CDF.

        Parameters
        ----------
        mean_high : float
            Ensemble mean forecast high temperature
        threshold : float
            For above/below: the temperature threshold
            For range: the midpoint of the range
        direction : str
            "above", "below", or "range"
        confidence : float
            Forecast confidence (0-1), used to blend with 0.5
        std : float
            Ensemble standard deviation (model disagreement in °F)
        is_range : bool
            Whether this is a range market
        range_low : float
            Low end of range (e.g., 87 for "87-88°F")
        range_high : float
            High end of range (e.g., 88 for "87-88°F")
        """
        # Use ensemble std as the base sigma, but ensure minimum
        # Ensemble std represents model disagreement
        # Add base forecast error on top
        base_sigma = max(1.5, std * 1.2)

        # Adjust sigma by confidence (lower confidence = wider distribution)
        sigma = base_sigma / max(confidence, 0.3)
        sigma = max(sigma, 1.0)  # Floor at 1°F

        if direction == "range" and range_high > range_low:
            # TRUNCATED NORMAL: P(range_low <= X <= range_high)
            # = Phi((high - mu) / sigma) - Phi((low - mu) / sigma)
            z_high = (range_high - mean_high) / sigma
            z_low = (range_low - mean_high) / sigma
            phi_high = 0.5 * (1 + _math.erf(z_high / _math.sqrt(2)))
            phi_low = 0.5 * (1 + _math.erf(z_low / _math.sqrt(2)))
            prob = phi_high - phi_low

            # Range markets need a width adjustment
            # A 2°F range at the mean has ~34% probability (not 50%)
            # This is correct behavior — narrow ranges near the mean are good bets
            # Wide ranges are less likely to be exactly right

        elif direction == "above":
            z = (mean_high - threshold) / sigma
            prob = 1 - 0.5 * (1 + _math.erf(-z / _math.sqrt(2)))

        elif direction == "below":
            z = (threshold - mean_high) / sigma
            prob = 1 - 0.5 * (1 + _math.erf(-z / _math.sqrt(2)))

        else:
            prob = 0.5

        # Blend with confidence — low confidence regresses toward 0.5
        prob = 0.5 + (prob - 0.5) * min(confidence, 1.0)

        return np.clip(prob, 0.01, 0.99)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    async def evaluate_weather_signal(
        self, market: dict, forecast: EnsembleForecast
    ) -> Optional[dict]:
        """Evaluate a single Kalshi weather market using enhanced forecaster."""
        question = market.get("question", "")
        market_price = market.get("current_price", 0.5)

        parsed = self.parse_temperature_threshold(question)
        if parsed is None:
            return None

        threshold, direction = parsed

        # Parse range endpoints for range markets
        range_low = range_high = 0
        if direction == "range":
            import re
            q = question.lower().replace("**", "")
            rm = re.search(r'(\d{2,3})\s*-\s*(\d{2,3})\s*°', q)
            if rm:
                range_low = float(rm.group(1))
                range_high = float(rm.group(2))

        # Build enhanced calibrated forecast from source data
        enhancer = get_forecaster()
        source_forecasts = {}
        if forecast.nws_high is not None:
            source_forecasts["nws"] = forecast.nws_high
        if forecast.gfs_high is not None:
            source_forecasts["gfs"] = forecast.gfs_high
        if forecast.ecmwf_high is not None:
            source_forecasts["ecmwf"] = forecast.ecmwf_high
        # Fall back to ensemble mean if individual sources unavailable
        if not source_forecasts:
            source_forecasts["ensemble"] = forecast.mean_high_f

        city_key = self.identify_city_key(question) or forecast.city_name.lower()[:3]
        target_date = market.get("end_date", "")[:10] or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Estimate lead hours from end_date
        end_date = market.get("end_date", "")
        lead_hours = 24.0
        if end_date:
            try:
                exp_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                lead_hours = max((exp_dt - datetime.now(timezone.utc)).total_seconds() / 3600, 1)
            except (ValueError, TypeError):
                pass

        calibrated = enhancer.build_forecast(
            city_key=city_key,
            target_date=target_date,
            source_forecasts=source_forecasts,
            lead_hours=lead_hours,
        )

        # Use enhanced forecaster's trade decision
        signal = enhancer.should_trade(
            forecast=calibrated,
            market_price=market_price,
            threshold=threshold,
            direction=direction,
            range_low=range_low,
            range_high=range_high,
        )

        if signal is None:
            return None

        # Merge with market metadata
        model_str = "/".join(source_forecasts.keys())
        signal.update({
            "market_id": market["market_id"],
            "venue": "kalshi",
            "question": question[:80],
            "strategy": "weather",
            "ensemble_mean_f": calibrated.ensemble_mean_f,
            "ensemble_std_f": calibrated.ensemble_std_f,
            "n_models": calibrated.n_models,
            "models": model_str,
            "threshold_f": threshold,
            "threshold_direction": direction,
            "city": city_key,
            "days_out": max(int(lead_hours / 24), 0),
            "nws_high": forecast.nws_high,
            "gfs_high": forecast.gfs_high,
            "ecmwf_high": forecast.ecmwf_high,
            "end_date": market.get("end_date", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return signal

    async def generate_signals(self, markets: list[dict]) -> list[dict]:
        """Generate trading signals for Kalshi weather markets.

        CONVERGENCE: One trade per city per day. If multiple buckets
        have similar edge, skip — we're uncertain.
        """
        forecasts = await self.fetch_all_forecasts()
        if not forecasts:
            logger.warning("weather.no_forecasts")
            return []

        all_signals = []
        for market in markets:
            ticker = market.get("market_id", "")
            category = market.get("category", "")

            forecast = None
            for key in CITY_CONFIG:
                if key in ticker or key in category:
                    forecast = forecasts.get(key)
                    break

            if forecast is None:
                city_key = self.identify_city_key(market.get("question", ""))
                if city_key:
                    forecast = forecasts.get(city_key)

            if forecast is None:
                continue

            signal = await self.evaluate_weather_signal(market, forecast)
            if signal is not None:
                signal["size_usd"] = 0
                all_signals.append(signal)

        # CONVERGENCE: Group by city + event date, pick best per group
        from collections import defaultdict
        event_groups: dict[str, list[dict]] = defaultdict(list)
        for s in all_signals:
            city = s.get("city", "unknown")
            date = s.get("end_date", "")[:10]
            event_key = f"{city}_{date}"
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
                logger.info("weather.convergence_skip",
                            event_key=event_key, n=len(group),
                            best_edge=best["edge"],
                            gap=edge_gap)
                continue

            if best["direction"] != second["direction"]:
                logger.info("weather.conflict_skip",
                            event_key=event_key,
                            best_dir=best["direction"],
                            second_dir=second["direction"])
                continue

            signals.append(best)

        signals.sort(key=lambda s: abs(s["edge"]), reverse=True)

        logger.info(
            "weather.signals_generated",
            n=len(signals),
            markets_scanned=len(markets),
            forecasts_available=len(forecasts),
            avg_confidence=np.mean([s["confidence"] for s in signals]) if signals else 0,
        )

        return signals
