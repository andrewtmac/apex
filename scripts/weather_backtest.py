#!/usr/bin/env python3
"""Backtest the Enhanced Weather Forecaster against historical Kalshi settlements.

Tests: given historical forecasts and actual outcomes, what would our
trading P&L and win rate have been?
"""

import sqlite3
from pathlib import Path

import structlog

from enhanced_weather import EnhancedWeatherForecaster

logger = structlog.get_logger()

DB_PATH = Path(__file__).parent.parent / "data" / "weather" / "weather_history.db"


def run_backtest():
    """Run backtest against historical settlements."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    forecaster = EnhancedWeatherForecaster()

    # Get all settlements with market data
    c.execute("""
        SELECT s.city_key, s.event_date, s.ticker, s.market_type,
               s.threshold, s.range_low, s.range_high, s.result,
               s.actual_high_f
        FROM settlements s
        WHERE s.market_type IN ('range', 'above', 'below')
        ORDER BY s.event_date, s.city_key, s.threshold
    """)
    settlements = c.fetchall()

    # Get all actual temperatures
    c.execute("""
        SELECT city_key, target_date, forecast_high_f
        FROM forecasts WHERE source = 'actual'
    """)
    actuals = {(r[0], r[1]): r[2] for r in c.fetchall()}

    # Get all forecasts
    c.execute("""
        SELECT city_key, target_date, source, forecast_high_f, lead_hours
        FROM forecasts WHERE source != 'actual'
        ORDER BY target_date, city_key, source
    """)
    forecast_rows = c.fetchall()

    # Group forecasts by (city, date)
    forecast_groups: dict[tuple, dict[str, float]] = {}
    for city, date, source, temp, lead in forecast_rows:
        key = (city, date)
        if key not in forecast_groups:
            forecast_groups[key] = {}
        forecast_groups[key][source] = temp

    # Simulate trading
    trades = []
    total_pnl = 0
    wins = 0
    losses = 0

    # Group settlements by (city, date) to evaluate each day
    from collections import defaultdict
    daily_markets = defaultdict(list)
    for row in settlements:
        city, date, ticker, mtype, thresh, rlow, rhigh, result, actual = row
        daily_markets[(city, date)].append(row)

    for (city, date), markets in sorted(daily_markets.items()):
        # Check if we have actual temperature
        actual = actuals.get((city, date))
        if actual is None:
            # Use expiration_value from settlement
            actual = markets[0][8] if markets else None
        if actual is None:
            continue

        # Get forecasts for this day
        sources = forecast_groups.get((city, date), {})
        if not sources:
            # Use climatology
            clim = forecaster.get_climatology(city, date)
            if clim:
                sources = {"climatology": clim}
            else:
                continue

        # Estimate lead time (assume 24h for historical)
        lead_hours = 24.0

        # Build calibrated forecast
        forecast = forecaster.build_forecast(
            city_key=city,
            target_date=date,
            source_forecasts=sources,
            lead_hours=lead_hours,
        )

        # Evaluate each market
        best_trade = None
        best_edge = 0

        for mkt in markets:
            _, _, ticker, mtype, thresh, rlow, rhigh, result, _ = mkt

            # Get market price (we don't have historical prices, so estimate)
            # For range markets: ~20% base rate, so assume 0.20
            # For above/below: vary by distance from mean
            if mtype == "range":
                # Estimate price based on probability
                true_prob = forecaster.estimate_probability(
                    forecast, thresh, "range", rlow or 0, rhigh or 0
                )
                # Assume market price is roughly uniform across buckets
                market_price = 0.20  # Typical range market price
            elif mtype == "above":
                true_prob = forecaster.estimate_probability(
                    forecast, thresh, "above"
                )
                market_price = max(0.05, min(0.95, 0.3 + (thresh - 80) * 0.02))
            elif mtype == "below":
                true_prob = forecaster.estimate_probability(
                    forecast, thresh, "below"
                )
                market_price = max(0.05, min(0.95, 0.3 - (thresh - 80) * 0.02))
            else:
                continue

            edge = true_prob - market_price

            # Only trade if edge is significant
            if abs(edge) > abs(best_edge) and abs(edge) > 0.05:
                best_trade = {
                    "ticker": ticker,
                    "city": city,
                    "date": date,
                    "mtype": mtype,
                    "threshold": thresh,
                    "range_low": rlow,
                    "range_high": rhigh,
                    "direction": "BUY" if edge > 0 else "SELL",
                    "true_prob": true_prob,
                    "market_price": market_price,
                    "edge": edge,
                    "result": result,
                    "actual": actual,
                    "ensemble_mean": forecast.ensemble_mean_f,
                    "calibrated_sigma": forecast.calibrated_sigma,
                    "confidence": forecast.confidence,
                    "regime": forecast.regime,
                }
                best_edge = edge

        if best_trade is None:
            continue

        # Calculate P&L
        t = best_trade
        won = False
        if t["direction"] == "BUY":
            won = t["result"] == "yes"
            pnl = (1.0 - t["market_price"]) if won else -t["market_price"]
        else:
            won = t["result"] == "no"
            pnl = t["market_price"] if won else -(1.0 - t["market_price"])

        total_pnl += pnl
        if won:
            wins += 1
        else:
            losses += 1

        t["won"] = won
        t["pnl"] = round(pnl, 4)
        trades.append(t)

    conn.close()

    # Report
    print("\n" + "=" * 70)
    print("  APEX Enhanced Weather Forecaster — Backtest Results")
    print("=" * 70)

    if not trades:
        print("  No trades generated (insufficient forecast data)")
        return

    total = wins + losses
    win_rate = wins / total if total > 0 else 0

    print(f"\n  Total trades:    {total}")
    print(f"  Wins:            {wins}")
    print(f"  Losses:          {losses}")
    print(f"  Win rate:        {win_rate:.1%}")
    print(f"  Total P&L:       ${total_pnl:+.2f} (per $1 contract)")
    print(f"  Avg P&L/trade:   ${total_pnl/total:+.4f}")

    # By market type
    for mtype in ["range", "above", "below"]:
        type_trades = [t for t in trades if t["mtype"] == mtype]
        if type_trades:
            tw = sum(1 for t in type_trades if t["won"])
            tl = len(type_trades)
            tp = sum(t["pnl"] for t in type_trades)
            print(f"\n  {mtype.upper()} markets: {tl} trades, {tw/tl:.1%} win rate, P&L ${tp:+.2f}")

    # By city
    print("\n  By city:")
    cities = sorted(set(t["city"] for t in trades))
    for city in cities:
        ct = [t for t in trades if t["city"] == city]
        cw = sum(1 for t in ct if t["won"])
        cp = sum(t["pnl"] for t in ct)
        print(f"    {city:5s}: {len(ct):3d} trades, {cw/len(ct):.1%} win, P&L ${cp:+.2f}")

    # By regime
    print("\n  By regime:")
    for regime in ["STABLE", "NORMAL", "UNCERTAIN", "VOLATILE"]:
        rt = [t for t in trades if t.get("regime") == regime]
        if rt:
            rw = sum(1 for t in rt if t["won"])
            rp = sum(t["pnl"] for t in rt)
            print(f"    {regime:10s}: {len(rt):3d} trades, {rw/len(rt):.1%} win, P&L ${rp:+.2f}")

    # By confidence
    print("\n  By confidence bucket:")
    for low, high, label in [(0, 0.5, "low (0-50%)"), (0.5, 0.7, "mid (50-70%)"),
                              (0.7, 0.85, "high (70-85%)"), (0.85, 1.0, "v.high (85%+)")]:
        ct = [t for t in trades if low <= t.get("confidence", 0) < high]
        if ct:
            cw = sum(1 for t in ct if t["won"])
            cp = sum(t["pnl"] for t in ct)
            print(f"    {label:16s}: {len(ct):3d} trades, {cw/len(ct):.1%} win, P&L ${cp:+.2f}")

    # Sample trades
    print("\n  Sample winning trades:")
    winners = [t for t in trades if t["won"]][:5]
    for t in winners:
        print(f"    {t['date']} {t['city']} {t['mtype']} {t['threshold']:.1f}°F "
              f"edge={t['edge']:+.1%} actual={t['actual']:.0f}°F "
              f"σ={t['calibrated_sigma']:.1f}°F conf={t['confidence']:.0%}")

    print("\n  Sample losing trades:")
    losers = [t for t in trades if not t["won"]][:5]
    for t in losers:
        print(f"    {t['date']} {t['city']} {t['mtype']} {t['threshold']:.1f}°F "
              f"edge={t['edge']:+.1%} actual={t['actual']:.0f}°F "
              f"σ={t['calibrated_sigma']:.1f}°F conf={t['confidence']:.0%}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_backtest()
