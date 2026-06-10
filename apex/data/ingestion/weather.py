"""
APEX Weather Ensemble Ingester

Polls multiple free and keyed weather APIs, computes an ensemble
forecast (mean + spread), and publishes to apex:weather Redis stream.
"""

from __future__ import annotations

import asyncio
import statistics
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog

from apex.config import ApexConfig
from apex.data.store import FeatureStore
from apex.data.streams import WEATHER_STREAM, StreamPublisher

logger = structlog.get_logger(__name__)

_POLL_INTERVAL = 300  # 5 minutes

# Default cities to track (can be extended via config)
DEFAULT_CITIES: list[dict[str, Any]] = [
    {"name": "New York", "lat": 40.7128, "lon": -74.0060, "nws_office": "OKX", "nws_grid": "33,37"},
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437, "nws_office": "LOX", "nws_grid": "154,44"},
    {"name": "Chicago", "lat": 41.8781, "lon": -87.6298, "nws_office": "LOT", "nws_grid": "75,72"},
    {"name": "Miami", "lat": 25.7617, "lon": -80.1918, "nws_office": "MFL", "nws_grid": "110,50"},
    {"name": "Phoenix", "lat": 33.4484, "lon": -112.0740, "nws_office": "PSR", "nws_grid": "159,59"},
]


class WeatherIngester:
    """
    Polls multiple weather APIs, computes ensemble forecasts.

    Sources:
    - Open-Meteo (free, no key)
    - NWS (free, no key)
    - OpenWeatherMap (key from env)
    - Visual Crossing (key from env)
    - Tomorrow.io (key from env)

    Lifecycle::

        ingester = WeatherIngester(config)
        await ingester.start()
        await ingester.stop()
    """

    def __init__(self, config: ApexConfig):
        self._redis_url = config.infra.redis_url
        self._db_url = config.infra.database_url
        self._owm_key = config.data_sources.newsapi_ai_key  # will be loaded separately
        self._publisher: Optional[StreamPublisher] = None
        self._store: Optional[FeatureStore] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._cities = DEFAULT_CITIES
        self._running = False
        self._tasks: list[asyncio.Task] = []

        # Load optional weather API keys from env
        import os
        self._owm_key = os.getenv("OPENWEATHERMAP_API_KEY", "")
        self._vc_key = os.getenv("VISUAL_CROSSING_KEY", "")
        self._tomorrow_key = os.getenv("TOMORROW_IO_KEY", "")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._publisher = StreamPublisher(self._redis_url)
        await self._publisher.connect()

        self._store = FeatureStore(self._redis_url, self._db_url)
        await self._store.connect()

        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(20.0),
            headers={"User-Agent": "APEX-Weather/1.0 (contact@apex.trade)"},
        )
        self._running = True

        self._tasks = [
            asyncio.create_task(self._poll_loop(), name="weather-poll"),
        ]
        logger.info("weather_ingester.started", n_cities=len(self._cities))

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
        logger.info("weather_ingester.stopped")

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        while self._running:
            await self._fetch_all_cities()
            await asyncio.sleep(_POLL_INTERVAL)

    async def _fetch_all_cities(self) -> None:
        for city in self._cities:
            try:
                ensemble = await self._build_ensemble(city)
                if ensemble:
                    assert self._publisher is not None and self._store is not None
                    await self._publisher.publish(WEATHER_STREAM, ensemble)
                    await self._store.put(
                        entity_id=f"weather:{city['name']}",
                        feature_set="weather_ensemble",
                        features=ensemble,
                    )
            except Exception:
                logger.warning("weather.city_failed", city=city["name"])

    # ------------------------------------------------------------------
    # Individual source fetchers
    # ------------------------------------------------------------------

    async def _fetch_open_meteo(self, lat: float, lon: float) -> Optional[dict[str, float]]:
        """Open-Meteo: free, no key required."""
        assert self._http is not None
        try:
            resp = await self._http.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": True,
                    "temperature_unit": "fahrenheit",
                    "windspeed_unit": "mph",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            cw = data.get("current_weather", {})
            return {
                "temp_f": cw.get("temperature", 0),
                "wind_mph": cw.get("windspeed", 0),
                "source": "open_meteo",
            }
        except Exception:
            logger.debug("weather.open_meteo_failed")
            return None

    async def _fetch_nws(self, office: str, grid: str) -> Optional[dict[str, float]]:
        """NWS: free, no key required."""
        assert self._http is not None
        try:
            url = f"https://api.weather.gov/gridpoints/{office}/{grid}/forecast"
            resp = await self._http.get(url)
            resp.raise_for_status()
            data = resp.json()
            periods = data.get("properties", {}).get("periods", [])
            if periods:
                current = periods[0]
                temp = current.get("temperature", 0)
                # NWS may return Celsius; check unit
                unit = current.get("temperatureUnit", "F")
                if unit == "C":
                    temp = temp * 9 / 5 + 32
                wind_str = current.get("windSpeed", "0 mph")
                wind_mph = float(wind_str.split()[0]) if wind_str else 0
                return {
                    "temp_f": temp,
                    "wind_mph": wind_mph,
                    "source": "nws",
                }
        except Exception:
            logger.debug("weather.nws_failed")
        return None

    async def _fetch_openweathermap(self, lat: float, lon: float) -> Optional[dict[str, float]]:
        """OpenWeatherMap: requires API key."""
        if not self._owm_key:
            return None
        assert self._http is not None
        try:
            resp = await self._http.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "lat": lat,
                    "lon": lon,
                    "appid": self._owm_key,
                    "units": "imperial",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            main = data.get("main", {})
            wind = data.get("wind", {})
            return {
                "temp_f": main.get("temp", 0),
                "humidity": main.get("humidity", 0),
                "wind_mph": wind.get("speed", 0),
                "source": "openweathermap",
            }
        except Exception:
            logger.debug("weather.owm_failed")
            return None

    async def _fetch_visual_crossing(self, lat: float, lon: float) -> Optional[dict[str, float]]:
        """Visual Crossing: requires API key."""
        if not self._vc_key:
            return None
        assert self._http is not None
        try:
            resp = await self._http.get(
                f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/today",
                params={
                    "unitGroup": "us",
                    "key": self._vc_key,
                    "include": "current",
                    "contentType": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            current = data.get("currentConditions", {})
            return {
                "temp_f": current.get("temp", 0),
                "humidity": current.get("humidity", 0),
                "wind_mph": current.get("windspeed", 0),
                "source": "visual_crossing",
            }
        except Exception:
            logger.debug("weather.vc_failed")
            return None

    async def _fetch_tomorrow_io(self, lat: float, lon: float) -> Optional[dict[str, float]]:
        """Tomorrow.io: requires API key."""
        if not self._tomorrow_key:
            return None
        assert self._http is not None
        try:
            resp = await self._http.get(
                "https://api.tomorrow.io/v4/weather/realtime",
                params={
                    "location": f"{lat},{lon}",
                    "apikey": self._tomorrow_key,
                    "units": "imperial",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            values = data.get("data", {}).get("values", {})
            return {
                "temp_f": values.get("temperature", 0),
                "humidity": values.get("humidity", 0),
                "wind_mph": values.get("windSpeed", 0),
                "source": "tomorrow_io",
            }
        except Exception:
            logger.debug("weather.tomorrow_io_failed")
            return None

    # ------------------------------------------------------------------
    # Ensemble builder
    # ------------------------------------------------------------------

    async def _build_ensemble(self, city: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Fetch from all sources and compute ensemble statistics."""
        lat, lon = city["lat"], city["lon"]

        # Fetch concurrently from all sources
        results = await asyncio.gather(
            self._fetch_open_meteo(lat, lon),
            self._fetch_nws(city.get("nws_office", ""), city.get("nws_grid", "")),
            self._fetch_openweathermap(lat, lon),
            self._fetch_visual_crossing(lat, lon),
            self._fetch_tomorrow_io(lat, lon),
            return_exceptions=True,
        )

        forecasts: list[dict[str, float]] = []
        sources_used: list[str] = []
        for r in results:
            if isinstance(r, dict) and r is not None:
                forecasts.append(r)
                sources_used.append(str(r.get("source", "unknown")))

        if not forecasts:
            logger.warning("weather.no_sources", city=city["name"])
            return None

        temps = [f["temp_f"] for f in forecasts if "temp_f" in f]
        winds = [f["wind_mph"] for f in forecasts if "wind_mph" in f]
        humidities = [f.get("humidity", 0) for f in forecasts if f.get("humidity") is not None]

        mean_temp = statistics.mean(temps) if temps else 0
        temp_spread = statistics.stdev(temps) if len(temps) >= 2 else 0
        mean_wind = statistics.mean(winds) if winds else 0
        mean_humidity = statistics.mean(humidities) if humidities else 0

        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "city": city["name"],
            "lat": lat,
            "lon": lon,
            "ensemble_temp_f": round(mean_temp, 1),
            "temp_spread_f": round(temp_spread, 2),
            "ensemble_wind_mph": round(mean_wind, 1),
            "ensemble_humidity_pct": round(mean_humidity, 1),
            "n_sources": len(forecasts),
            "sources": sources_used,
            "individual_forecasts": forecasts,
        }
