#!/usr/bin/env python3
"""Weather Data Collector for APEX V2.

Collects and stores:
1. Historical Kalshi weather market settlements (actual temperatures)
2. Historical weather forecasts from multiple sources
3. Forecast accuracy metrics (model bias, RMSE by city/lead-time)
4. Climate normals (baseline distributions)

Uses SQLite for storage. Run daily to build training dataset.
"""

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import structlog

logger = structlog.get_logger()

DB_PATH = Path(__file__).parent.parent / "data" / "weather" / "weather_history.db"

CITIES = {
    "nyc": {
        "name": "New York", "lat": 40.78, "lon": -73.97,
        "nws_id": "OKX", "grid": (33, 37), "series": "KXHIGHNY",
    },
    "chi": {
        "name": "Chicago", "lat": 41.88, "lon": -87.63,
        "nws_id": "LOT", "grid": (76, 73), "series": "KXHIGHCHI",
    },
    "la": {
        "name": "Los Angeles", "lat": 34.05, "lon": -118.24,
        "nws_id": "LOX", "grid": (155, 44), "series": "KXHIGHLA",
    },
    "mia": {
        "name": "Miami", "lat": 25.76, "lon": -80.19,
        "nws_id": "MFL", "grid": (107, 50), "series": "KXHIGHMIA",
    },
    "dc": {
        "name": "Washington DC", "lat": 38.91, "lon": -77.04,
        "nws_id": "LWX", "grid": (96, 72), "series": "KXHIGHDC",
    },
    "hou": {
        "name": "Houston", "lat": 29.76, "lon": -95.37,
        "nws_id": "HGX", "grid": (64, 96), "series": "KXHIGHHOU",
    },
    "dal": {
        "name": "Dallas", "lat": 32.78, "lon": -96.80,
        "nws_id": "FWD", "grid": (34, 69), "series": "KXHIGHDAL",
    },
    "den": {
        "name": "Denver", "lat": 39.74, "lon": -104.99,
        "nws_id": "BOU", "grid": (61, 58), "series": "KXHIGHDEN",
    },
    "phx": {
        "name": "Phoenix", "lat": 33.45, "lon": -112.07,
        "nws_id": "PSR", "grid": (156, 56), "series": "KXHIGHPHX",
    },
    "atl": {
        "name": "Atlanta", "lat": 33.75, "lon": -84.39,
        "nws_id": "FFC", "grid": (52, 87), "series": "KXHIGHATL",
    },
    "sf": {
        "name": "San Francisco", "lat": 37.77, "lon": -122.42,
        "nws_id": "MTR", "grid": (85, 105), "series": "KXHIGHSF",
    },
    "bos": {
        "name": "Boston", "lat": 42.36, "lon": -71.06,
        "nws_id": "BOX", "grid": (71, 64), "series": "KXHIGHBOS",
    },
    "sea": {
        "name": "Seattle", "lat": 47.61, "lon": -122.33,
        "nws_id": "SEW", "grid": (125, 68), "series": "KXHIGHSEA",
    },
    "aus": {
        "name": "Austin", "lat": 30.27, "lon": -97.74,
        "nws_id": "EWX", "grid": (164, 82), "series": "KXHIGHAUS",
    },
}


def init_db():
    """Create the weather history database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS settlements (
            ticker TEXT PRIMARY KEY,
            city_key TEXT NOT NULL,
            event_date TEXT NOT NULL,
            actual_high_f REAL NOT NULL,
            result TEXT NOT NULL,
            market_type TEXT,
            threshold REAL,
            range_low REAL,
            range_high REAL,
            close_time TEXT,
            expiration_value REAL,
            collected_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city_key TEXT NOT NULL,
            target_date TEXT NOT NULL,
            forecast_time TEXT NOT NULL,
            source TEXT NOT NULL,
            forecast_high_f REAL NOT NULL,
            lead_hours REAL,
            UNIQUE(city_key, target_date, forecast_time, source)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS forecast_accuracy (
            city_key TEXT NOT NULL,
            source TEXT NOT NULL,
            lead_hours_bin TEXT NOT NULL,
            bias_f REAL,
            mae_f REAL,
            rmse_f REAL,
            n_samples INTEGER,
            pct_within_1f REAL,
            pct_within_2f REAL,
            pct_within_3f REAL,
            last_updated TEXT,
            PRIMARY KEY (city_key, source, lead_hours_bin)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS climate_normals (
            city_key TEXT NOT NULL,
            month INTEGER NOT NULL,
            day INTEGER NOT NULL,
            normal_high_f REAL,
            std_high_f REAL,
            p10_high_f REAL,
            p90_high_f REAL,
            n_years INTEGER,
            PRIMARY KEY (city_key, month, day)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS weather_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            city_key TEXT,
            event_date TEXT,
            direction TEXT,
            entry_price REAL,
            true_prob REAL,
            edge REAL,
            ensemble_mean_f REAL,
            ensemble_std_f REAL,
            nws_forecast REAL,
            gfs_forecast REAL,
            ecmwf_forecast REAL,
            hrrr_forecast REAL,
            nam_forecast REAL,
            actual_high_f REAL,
            outcome TEXT,
            pnl REAL,
            confidence REAL,
            regime TEXT,
            days_out INTEGER,
            collected_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    return conn


async def collect_kalshi_settlements(conn, days_back=60):
    """Collect historical weather settlements from Kalshi API."""
    c = conn.cursor()
    collected = 0

    async with httpx.AsyncClient(timeout=30) as client:
        for city_key, city_info in CITIES.items():
            series = city_info["series"]
            cursor = ""

            for status in ["settled", "finalized"]:
                page = 0
                while page < 10:
                    try:
                        params = {
                            "limit": 100,
                            "series_ticker": series,
                            "status": status,
                        }
                        if cursor:
                            params["cursor"] = cursor

                        resp = await client.get(
                            "https://api.elections.kalshi.com"
                            "/trade-api/v2/markets",
                            params=params,
                        )
                        if resp.status_code != 200:
                            break

                        data = resp.json()
                        markets = data.get("markets", [])
                        if not markets:
                            break

                        for m in markets:
                            ticker = m.get("ticker", "")
                            result = (m.get("result") or "").lower()
                            exp_val = m.get("expiration_value")

                            if not result or result not in ("yes", "no"):
                                continue
                            if not exp_val:
                                continue

                            parts = ticker.split("-")
                            if len(parts) < 3:
                                continue

                            date_str = parts[1]
                            try:
                                event_date = datetime.strptime(
                                    date_str, "%y%b%d"
                                ).strftime("%Y-%m-%d")
                            except ValueError:
                                continue

                            market_part = parts[2]
                            market_type = "unknown"
                            threshold = None
                            range_low = None
                            range_high = None

                            if market_part.startswith("T"):
                                try:
                                    threshold = float(market_part[1:])
                                    title = m.get("title", "").lower()
                                    if ("above" in title
                                            or ">" in title):
                                        market_type = "above"
                                    else:
                                        market_type = "below"
                                except ValueError:
                                    pass
                            elif market_part.startswith("B"):
                                market_type = "range"
                                try:
                                    mid = float(market_part[1:])
                                    range_low = mid - 0.5
                                    range_high = mid + 0.5
                                    threshold = mid
                                except ValueError:
                                    pass

                            c.execute("""
                                INSERT OR REPLACE INTO settlements
                                (ticker, city_key, event_date,
                                 actual_high_f, result,
                                 market_type, threshold,
                                 range_low, range_high,
                                 close_time, expiration_value)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                ticker, city_key, event_date,
                                float(exp_val), result, market_type,
                                threshold, range_low, range_high,
                                m.get("close_time", ""),
                                float(exp_val),
                            ))
                            collected += 1

                        cursor = data.get("cursor", "")
                        if not cursor:
                            break
                        page += 1
                        await asyncio.sleep(0.5)

                    except Exception as e:
                        logger.warning(
                            "weather.settlement_fetch_error",
                            city=city_key, error=str(e),
                        )
                        break

    conn.commit()
    logger.info("weather.settlements_collected", count=collected)
    return collected


async def collect_open_meteo_historical(conn, days_back=60):
    """Collect historical actual temperatures from Open-Meteo."""
    c = conn.cursor()
    now = datetime.now(timezone.utc)
    end_date = now.date() - timedelta(days=1)
    start_date = end_date - timedelta(days=days_back)

    async with httpx.AsyncClient(timeout=30) as client:
        for city_key, city_info in CITIES.items():
            try:
                resp = await client.get(
                    "https://archive-api.open-meteo.com/v1/archive",
                    params={
                        "latitude": city_info["lat"],
                        "longitude": city_info["lon"],
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "daily": "temperature_2m_max",
                        "temperature_unit": "fahrenheit",
                        "timezone": "America/New_York",
                    },
                )
                if resp.status_code != 200:
                    continue

                data = resp.json()
                dates = data.get("daily", {}).get("time", [])
                temps = data.get("daily", {}).get(
                    "temperature_2m_max", []
                )

                inserted = 0
                for date_str, temp in zip(dates, temps):
                    if temp is None:
                        continue
                    c.execute("""
                        INSERT OR IGNORE INTO forecasts
                        (city_key, target_date, forecast_time,
                         source, forecast_high_f, lead_hours)
                        VALUES (?, ?, ?, 'actual', ?, 0)
                    """, (city_key, date_str, date_str, float(temp)))
                    inserted += 1

                logger.info(
                    "weather.historical_collected",
                    city=city_key, days=inserted,
                )
            except Exception as e:
                logger.warning(
                    "weather.historical_error",
                    city=city_key, error=str(e),
                )
            await asyncio.sleep(1)

    conn.commit()


async def collect_nws_forecasts(conn):
    """Collect current NWS forecasts."""
    c = conn.cursor()
    now = datetime.now(timezone.utc)

    async with httpx.AsyncClient(timeout=30) as client:
        for city_key, city_info in CITIES.items():
            try:
                grid_id = city_info["nws_id"]
                gx, gy = city_info["grid"]

                resp = await client.get(
                    f"https://api.weather.gov/gridpoints/"
                    f"{grid_id}/{gx},{gy}/forecast",
                    headers={"User-Agent": "APEX-WeatherBot/1.0"},
                )
                if resp.status_code != 200:
                    continue

                data = resp.json()
                periods = data.get(
                    "properties", {}
                ).get("periods", [])

                for period in periods[:7]:
                    if period.get("isDaytime", False):
                        temp = period.get("temperature")
                        start = period.get("startTime", "")
                        if temp and start:
                            try:
                                start_dt = datetime.fromisoformat(
                                    start.replace("Z", "+00:00")
                                )
                                target_date = (
                                    start_dt.strftime("%Y-%m-%d")
                                )
                                lead_h = (
                                    (start_dt - now).total_seconds()
                                    / 3600
                                )
                                c.execute("""
                                    INSERT OR REPLACE INTO forecasts
                                    (city_key, target_date,
                                     forecast_time, source,
                                     forecast_high_f, lead_hours)
                                    VALUES (?, ?, ?, 'nws', ?, ?)
                                """, (
                                    city_key, target_date,
                                    now.isoformat(), float(temp),
                                    lead_h,
                                ))
                            except (ValueError, TypeError):
                                pass
            except Exception as e:
                logger.warning(
                    "weather.nws_error",
                    city=city_key, error=str(e),
                )
            await asyncio.sleep(0.5)

    conn.commit()


async def collect_open_meteo_forecasts(conn):
    """Collect GFS and ECMWF forecasts from Open-Meteo."""
    c = conn.cursor()
    now = datetime.now(timezone.utc)

    models_map = {
        "gfs_seamless_temperature_2m_max": "gfs",
        "ecmwf_ifs025_temperature_2m_max": "ecmwf",
        "temperature_2m_max": "open_meteo_avg",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        for city_key, city_info in CITIES.items():
            try:
                resp = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": city_info["lat"],
                        "longitude": city_info["lon"],
                        "daily": "temperature_2m_max",
                        "temperature_unit": "fahrenheit",
                        "forecast_days": 7,
                        "models": "gfs_seamless,ecmwf_ifs025",
                        "timezone": "America/New_York",
                    },
                )
                if resp.status_code != 200:
                    continue

                data = resp.json()
                daily = data.get("daily", {})
                dates = daily.get("time", [])

                for model_key, source in models_map.items():
                    temps = daily.get(model_key, [])
                    if not temps:
                        continue

                    for date_str, temp in zip(dates, temps):
                        if temp is None:
                            continue
                        try:
                            target_dt = datetime.strptime(
                                date_str, "%Y-%m-%d"
                            )
                            lead_h = (
                                (target_dt - now.replace(tzinfo=None))
                                .total_seconds() / 3600
                            )
                            c.execute("""
                                INSERT OR REPLACE INTO forecasts
                                (city_key, target_date,
                                 forecast_time, source,
                                 forecast_high_f, lead_hours)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (
                                city_key, date_str,
                                now.isoformat(), source,
                                float(temp), lead_h,
                            ))
                        except (ValueError, TypeError):
                            pass

            except Exception as e:
                logger.warning(
                    "weather.forecast_error",
                    city=city_key, error=str(e),
                )
            await asyncio.sleep(0.5)

    conn.commit()


async def collect_hrrr_forecasts(conn):
    """Collect HRRR forecasts via Open-Meteo."""
    c = conn.cursor()
    now = datetime.now(timezone.utc)

    async with httpx.AsyncClient(timeout=30) as client:
        for city_key, city_info in CITIES.items():
            try:
                resp = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": city_info["lat"],
                        "longitude": city_info["lon"],
                        "hourly": "temperature_2m",
                        "temperature_unit": "fahrenheit",
                        "forecast_days": 2,
                        "models": "hrrr_seamless",
                        "timezone": "America/New_York",
                    },
                )
                if resp.status_code != 200:
                    continue

                data = resp.json()
                hourly = data.get("hourly", {})
                times = hourly.get("time", [])
                temps = hourly.get("temperature_2m", [])

                daily_max = {}
                for t_str, temp in zip(times, temps):
                    if temp is not None:
                        d = t_str[:10]
                        if d not in daily_max or temp > daily_max[d]:
                            daily_max[d] = temp

                for date_str, max_temp in daily_max.items():
                    try:
                        target_dt = datetime.strptime(
                            date_str, "%Y-%m-%d"
                        )
                        lead_h = (
                            (target_dt - now.replace(tzinfo=None))
                            .total_seconds() / 3600
                        )
                        c.execute("""
                            INSERT OR REPLACE INTO forecasts
                            (city_key, target_date,
                             forecast_time, source,
                             forecast_high_f, lead_hours)
                            VALUES (?, ?, ?, 'hrrr', ?, ?)
                        """, (
                            city_key, date_str,
                            now.isoformat(), max_temp, lead_h,
                        ))
                    except (ValueError, TypeError):
                        pass

            except Exception as e:
                logger.warning(
                    "weather.hrrr_error",
                    city=city_key, error=str(e),
                )
            await asyncio.sleep(0.5)

    conn.commit()


def compute_accuracy_metrics(conn):
    """Compute forecast accuracy by city, source, and lead-time bin."""
    c = conn.cursor()

    c.execute("""
        SELECT f.city_key, f.source, f.lead_hours,
               f.forecast_high_f, a.forecast_high_f as actual_high
        FROM forecasts f
        JOIN forecasts a ON f.city_key = a.city_key
            AND f.target_date = a.target_date
            AND a.source = 'actual'
        WHERE f.source != 'actual'
          AND f.forecast_high_f IS NOT NULL
          AND a.forecast_high_f IS NOT NULL
    """)
    rows = c.fetchall()

    bins = {
        "0-6h": (0, 6), "6-12h": (6, 12),
        "12-24h": (12, 24), "24-48h": (24, 48),
        "48-72h": (48, 72), "72-168h": (72, 168),
    }

    from collections import defaultdict
    groups = defaultdict(list)
    for city, source, lead, forecast, actual in rows:
        if lead is None:
            continue
        for bin_name, (lo, hi) in bins.items():
            if lo <= lead < hi:
                groups[(city, source, bin_name)].append(
                    (forecast, actual)
                )
                break

    import math
    for (city, source, bin_name), values in groups.items():
        if len(values) < 3:
            continue

        errors = [f - a for f, a in values]
        abs_errors = [abs(e) for e in errors]

        bias = sum(errors) / len(errors)
        mae = sum(abs_errors) / len(abs_errors)
        rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
        within_1 = (
            sum(1 for e in abs_errors if e <= 1.0) / len(abs_errors)
        )
        within_2 = (
            sum(1 for e in abs_errors if e <= 2.0) / len(abs_errors)
        )
        within_3 = (
            sum(1 for e in abs_errors if e <= 3.0) / len(abs_errors)
        )

        c.execute("""
            INSERT OR REPLACE INTO forecast_accuracy
            (city_key, source, lead_hours_bin, bias_f, mae_f, rmse_f,
             n_samples, pct_within_1f, pct_within_2f,
             pct_within_3f, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            city, source, bin_name,
            round(bias, 2), round(mae, 2), round(rmse, 2),
            len(values),
            round(within_1, 3), round(within_2, 3),
            round(within_3, 3),
            datetime.now(timezone.utc).isoformat(),
        ))

    conn.commit()
    logger.info("weather.accuracy_computed", groups=len(groups))


async def run_full_collection():
    """Run all data collection tasks."""
    conn = init_db()

    print("=" * 60)
    print("  APEX Weather Data Collector")
    print("=" * 60)

    print("\n1. Collecting Kalshi weather settlements...")
    settlements = await collect_kalshi_settlements(conn, days_back=60)
    print(f"   Collected {settlements} settlement records")

    print("\n2. Collecting historical actual temperatures...")
    await collect_open_meteo_historical(conn, days_back=60)
    print("   Done")

    print("\n3. Collecting NWS forecasts...")
    await collect_nws_forecasts(conn)
    print("   Done")

    print("\n4. Collecting GFS/ECMWF forecasts...")
    await collect_open_meteo_forecasts(conn)
    print("   Done")

    print("\n5. Collecting HRRR forecasts...")
    await collect_hrrr_forecasts(conn)
    print("   Done")

    print("\n6. Computing accuracy metrics...")
    compute_accuracy_metrics(conn)
    print("   Done")

    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM settlements")
    n_settle = c.fetchone()[0]
    c.execute(
        "SELECT COUNT(*) FROM forecasts WHERE source != 'actual'"
    )
    n_forecast = c.fetchone()[0]
    c.execute(
        "SELECT COUNT(*) FROM forecasts WHERE source = 'actual'"
    )
    n_actual = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM forecast_accuracy")
    n_accuracy = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT city_key) FROM settlements")
    n_cities = c.fetchone()[0]

    print(f"\n{'=' * 60}")
    print("  Database Summary:")
    print(f"    Settlements:     {n_settle} ({n_cities} cities)")
    print(f"    Forecasts:       {n_forecast}")
    print(f"    Actual temps:    {n_actual}")
    print(f"    Accuracy rows:   {n_accuracy}")
    print(f"    DB path:         {DB_PATH}")
    print(f"{'=' * 60}")

    conn.close()


if __name__ == "__main__":
    asyncio.run(run_full_collection())
