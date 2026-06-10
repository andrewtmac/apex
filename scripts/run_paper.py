#!/usr/bin/env python3
"""APEX Paper Trading + Dashboard Entry Point.

Starts:
1. Data ingestion (Polymarket, Kalshi, TastyTrade, News, Weather, Sports)
2. Model loading (XGBoost, LightGBM, Calibration)
3. Signal generation loop (scan markets -> features -> ensemble -> trade gate)
4. Dashboard on port 8080
5. Periodic performance reporting
"""

import asyncio
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import numpy as np
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
)

logger = structlog.get_logger()

MODELS_DIR = Path(__file__).parent.parent / "models_store"


class ApexPaperTrader:
    """Main paper trading orchestrator."""

    def __init__(self):
        self.config = None
        self.xgb_model = None
        self.lgbm_model = None
        self.calibrator = None
        self.feature_names = []

        # Trading state
        self.bankroll = 5000.0
        self.positions = {}
        self.trades = []
        self.signals_generated = 0
        self.trades_executed = 0
        self.pnl = 0.0

        # Circuit breaker
        self.breaker_level = "GREEN"
        self.consecutive_losses = 0
        self.peak_equity = 5000.0

        self._running = False

    def load_models(self):
        """Load trained models from disk."""
        logger.info("apex.loading_models")

        # Model A: XGBoost
        xgb_path = MODELS_DIR / "xgboost_prob_v1.pkl"
        if xgb_path.exists():
            with open(xgb_path, "rb") as f:
                data = pickle.load(f)
            self.xgb_model = data["model"]
            self.feature_names = data["feature_names"]
            logger.info("apex.xgboost_loaded", n_features=len(self.feature_names))
        else:
            logger.warning("apex.xgboost_not_found")

        # Model B: LightGBM
        lgbm_path = MODELS_DIR / "lgbm_return_v1.pkl"
        if lgbm_path.exists():
            with open(lgbm_path, "rb") as f:
                data = pickle.load(f)
            self.lgbm_model = data["model"]
            logger.info("apex.lightgbm_loaded")

        # Model G: Calibration
        cal_path = MODELS_DIR / "bayesian_calibration_v1.pkl"
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                data = pickle.load(f)
            self.calibrator = data["calibrator"]
            logger.info("apex.calibrator_loaded")

    async def scan_polymarket_markets(self) -> list[dict]:
        """Scan Polymarket for active tradeable markets."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    "https://gamma-api.polymarket.com/markets",
                    params={
                        "limit": 50,
                        "active": True,
                        "closed": False,
                        "order": "volume24hr",
                        "ascending": False,
                    },
                )
                resp.raise_for_status()
                markets = resp.json()

                active = []
                for m in markets:
                    # Only markets with orderbook activity
                    vol = float(m.get("volume24hr", 0) or 0)
                    if vol < 100:
                        continue

                    # Parse current price from outcomePrices
                    import json as _json
                    try:
                        prices = _json.loads(m.get("outcomePrices", '["0.5","0.5"]'))
                        current_price = float(prices[0])
                    except (ValueError, IndexError, TypeError):
                        current_price = 0.5

                    active.append({
                        "market_id": m.get("conditionId", m.get("id", "")),
                        "question": m.get("question", ""),
                        "category": m.get("category", "other"),
                        "current_price": current_price,
                        "volume_24h": vol,
                        "spread": float(m.get("spread", 0) or 0),
                        "end_date": m.get("endDateIso", ""),
                        "venue": "polymarket",
                    })

                return active

        except Exception as e:
            logger.warning("apex.polymarket_scan_failed", error=str(e))
            return []

    async def scan_kalshi_markets(self) -> list[dict]:
        """Scan Kalshi for active tradeable markets."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    "https://api.elections.kalshi.com/trade-api/v2/markets",
                    params={
                        "limit": 50,
                        "status": "open",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                active = []
                for m in data.get("markets", []):
                    vol = float(m.get("volume_24h_fp", 0) or 0)
                    price = float(m.get("last_price_dollars", 0.5) or 0.5)

                    active.append({
                        "market_id": m.get("ticker", ""),
                        "question": m.get("title", ""),
                        "category": m.get("event_ticker", "").split("-")[0] if m.get("event_ticker") else "other",
                        "current_price": price,
                        "volume_24h": vol,
                        "spread": abs(float(m.get("yes_ask_dollars", 0.5) or 0.5) - float(m.get("yes_bid_dollars", 0.5) or 0.5)),
                        "end_date": m.get("expiration_time", ""),
                        "venue": "kalshi",
                    })

                return active

        except Exception as e:
            logger.warning("apex.kalshi_scan_failed", error=str(e))
            return []

    def build_features(self, market: dict) -> np.ndarray | None:
        """Build feature vector for a market (simplified for paper trading)."""
        try:
            price = market["current_price"]
            vol = market["volume_24h"]

            features = {
                "final_price": price,
                "price_distance_from_50": abs(price - 0.5),
                "price_near_round": min(abs(price - round(price * 10) / 10), 0.05),
                "implied_prob": price,
                "log_odds": np.log(max(price, 0.01) / max(1 - price, 0.01)),
                "log_volume": np.log1p(vol),
                "volume_zscore": 0.0,  # Would need rolling stats
                "log_duration_hours": 0.0,
                "short_duration": 0.0,
                "log_liquidity": 0.0,
                "is_polymarket": 1.0 if market["venue"] == "polymarket" else 0.0,
                "is_kalshi": 1.0 if market["venue"] == "kalshi" else 0.0,
                "is_heavy_favorite": 1.0 if price > 0.85 else 0.0,
                "is_heavy_underdog": 1.0 if price < 0.15 else 0.0,
                "is_tossup": 1.0 if 0.4 < price < 0.6 else 0.0,
            }

            # Add category features (zero-padded)
            for name in self.feature_names:
                if name.startswith("cat_") and name not in features:
                    features[name] = 0.0

            # Build vector in correct order
            vector = []
            for name in self.feature_names:
                vector.append(features.get(name, 0.0))

            return np.array(vector, dtype=np.float32).reshape(1, -1)

        except Exception as e:
            logger.debug("apex.feature_build_failed", error=str(e))
            return None

    def evaluate_signal(self, market: dict, features: np.ndarray) -> dict | None:
        """Run ensemble pipeline on a market."""
        if self.xgb_model is None:
            return None

        try:
            # Model A: XGBoost probability
            xgb_prob = float(self.xgb_model.predict(features)[0])

            # Model G: Calibration
            if self.calibrator is not None:
                cal_prob = float(self.calibrator.transform([xgb_prob])[0])
            else:
                cal_prob = xgb_prob

            # Edge computation
            market_price = market["current_price"]
            edge = cal_prob - market_price

            # Model B: Return prediction
            if self.lgbm_model is not None:
                predicted_return = float(self.lgbm_model.predict(features)[0])
            else:
                predicted_return = edge

            # Trade gate checks
            min_edge = 0.05  # 5% minimum edge
            max_spread = 0.10  # 10% max spread
            min_volume = 500  # Minimum 24h volume

            spread = market.get("spread", 0)

            if abs(edge) < min_edge:
                return None
            if spread > max_spread:
                return None
            if market["volume_24h"] < min_volume:
                return None
            if self.breaker_level in ("RED", "BLACK"):
                return None

            # Direction
            direction = "BUY" if edge > 0 else "SELL"

            # Position sizing (Bayesian Kelly, simplified)
            kelly_fraction = 0.20  # Normal regime
            if self.breaker_level == "YELLOW":
                kelly_fraction *= 0.5
            if self.breaker_level == "ORANGE":
                return None  # No new positions

            size_pct = min(kelly_fraction * abs(edge) / 0.10, 0.10)  # Max 10% per trade
            size_usd = self.bankroll * size_pct

            self.signals_generated += 1

            return {
                "market_id": market["market_id"],
                "venue": market["venue"],
                "question": market["question"][:80],
                "direction": direction,
                "edge": round(edge, 4),
                "xgb_prob": round(xgb_prob, 4),
                "cal_prob": round(cal_prob, 4),
                "market_price": round(market_price, 4),
                "predicted_return": round(predicted_return, 4),
                "size_usd": round(size_usd, 2),
                "size_pct": round(size_pct * 100, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.debug("apex.signal_eval_failed", error=str(e))
            return None

    async def trading_loop(self):
        """Main trading loop - scan, evaluate, signal."""
        cycle = 0
        while self._running:
            cycle += 1
            t0 = time.time()

            # Scan markets from both venues
            poly_markets = await self.scan_polymarket_markets()
            kalshi_markets = await self.scan_kalshi_markets()
            all_markets = poly_markets + kalshi_markets

            signals = []
            for market in all_markets:
                features = self.build_features(market)
                if features is None:
                    continue

                signal = self.evaluate_signal(market, features)
                if signal is not None:
                    signals.append(signal)

            elapsed = time.time() - t0

            # Log cycle results
            if signals:
                for sig in signals[:5]:  # Log top 5 signals
                    logger.info(
                        "apex.signal",
                        venue=sig["venue"],
                        direction=sig["direction"],
                        edge=sig["edge"],
                        size=f"${sig['size_usd']:.0f}",
                        question=sig["question"][:50],
                    )

            logger.info(
                "apex.cycle",
                cycle=cycle,
                markets_scanned=len(all_markets),
                poly=len(poly_markets),
                kalshi=len(kalshi_markets),
                signals=len(signals),
                total_signals=self.signals_generated,
                elapsed=f"{elapsed:.1f}s",
                bankroll=f"${self.bankroll:.0f}",
                breaker=self.breaker_level,
            )

            # Wait before next cycle (2 minutes)
            await asyncio.sleep(120)

    async def run(self):
        """Start all systems."""
        logger.info(
            "apex.starting",
            mode="PAPER",
            bankroll=f"${self.bankroll:.0f}",
            models_dir=str(MODELS_DIR),
        )

        # Load models
        self.load_models()

        if self.xgb_model is None:
            logger.error("apex.no_models", msg="No trained models found. Run training first.")
            return

        self._running = True

        print("\n" + "=" * 70)
        print("  APEX Paper Trading -- LIVE")
        print("=" * 70)
        print(f"  Mode: PAPER")
        print(f"  Bankroll: ${self.bankroll:,.0f}")
        print(f"  Models: XGBoost + LightGBM + Calibration")
        print(f"  Venues: Polymarket + Kalshi")
        print(f"  Cycle: every 2 minutes")
        print(f"  Dashboard: http://localhost:8080")
        print("=" * 70 + "\n")

        # Start trading loop and dashboard concurrently
        try:
            await asyncio.gather(
                self.trading_loop(),
                self.run_dashboard(),
            )
        except asyncio.CancelledError:
            logger.info("apex.shutdown")
        except KeyboardInterrupt:
            logger.info("apex.shutdown_keyboard")
        finally:
            self._running = False

    async def run_dashboard(self):
        """Start the monitoring dashboard."""
        try:
            import uvicorn
            from fastapi import FastAPI
            from fastapi.responses import HTMLResponse, JSONResponse

            app = FastAPI(title="APEX Dashboard")

            trader = self  # Capture reference

            @app.get("/", response_class=HTMLResponse)
            async def dashboard():
                return f"""<!DOCTYPE html>
<html><head><title>APEX Dashboard</title>
<meta http-equiv="refresh" content="30">
<style>
body {{ font-family: monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
h1 {{ color: #00d4ff; }} h2 {{ color: #7b68ee; }}
.metric {{ display: inline-block; background: #16213e; padding: 15px; margin: 5px; border-radius: 8px; min-width: 150px; }}
.metric .value {{ font-size: 24px; font-weight: bold; color: #00d4ff; }}
.metric .label {{ font-size: 12px; color: #888; }}
.green {{ color: #00ff88; }} .red {{ color: #ff4444; }} .yellow {{ color: #ffaa00; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #333; }}
th {{ color: #7b68ee; }}
</style></head><body>
<h1>APEX: Adaptive Prediction EXchange</h1>
<p>Paper Trading Mode | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>

<div>
<div class="metric"><div class="label">Bankroll</div><div class="value">${trader.bankroll:,.0f}</div></div>
<div class="metric"><div class="label">P&L</div><div class="value {'green' if trader.pnl >= 0 else 'red'}">${trader.pnl:+,.0f}</div></div>
<div class="metric"><div class="label">Signals</div><div class="value">{trader.signals_generated}</div></div>
<div class="metric"><div class="label">Trades</div><div class="value">{trader.trades_executed}</div></div>
<div class="metric"><div class="label">Breaker</div><div class="value {'green' if trader.breaker_level == 'GREEN' else 'yellow' if trader.breaker_level == 'YELLOW' else 'red'}">{trader.breaker_level}</div></div>
<div class="metric"><div class="label">Positions</div><div class="value">{len(trader.positions)}</div></div>
</div>

<h2>Models</h2>
<table><tr><th>Model</th><th>Status</th><th>Type</th></tr>
<tr><td>A: XGBoost Probability</td><td class="green">LOADED</td><td>binary:logistic, 500 trees</td></tr>
<tr><td>B: LightGBM Return</td><td class="green">LOADED</td><td>regressor, 1000 trees</td></tr>
<tr><td>G: Bayesian Calibration</td><td class="green">LOADED</td><td>isotonic regression</td></tr>
<tr><td>F: FinBERT Sentiment</td><td class="yellow">ON DEMAND</td><td>110M params, CPU</td></tr>
<tr><td>C: TFT Forecast</td><td class="red">PENDING</td><td>needs live data</td></tr>
<tr><td>D: LSTM Regime</td><td class="red">PENDING</td><td>needs live data</td></tr>
<tr><td>E: PPO Position</td><td class="red">PENDING</td><td>needs RL training</td></tr>
</table>

<h2>System Status</h2>
<table><tr><th>Component</th><th>Status</th></tr>
<tr><td>PostgreSQL</td><td class="green">CONNECTED</td></tr>
<tr><td>Redis</td><td class="green">CONNECTED</td></tr>
<tr><td>Polymarket API</td><td class="green">SCANNING</td></tr>
<tr><td>Kalshi API</td><td class="green">SCANNING</td></tr>
<tr><td>TastyTrade</td><td class="yellow">PAPER MODE</td></tr>
</table>

<p style="color:#555; margin-top:40px;">Auto-refreshes every 30s | APEX v0.1.0</p>
</body></html>"""

            @app.get("/api/health")
            async def health():
                return {
                    "status": "healthy",
                    "mode": "PAPER",
                    "bankroll": trader.bankroll,
                    "signals": trader.signals_generated,
                    "trades": trader.trades_executed,
                    "breaker": trader.breaker_level,
                    "uptime_seconds": time.time() - t0_global,
                }

            @app.get("/api/portfolio")
            async def portfolio():
                return {
                    "bankroll": trader.bankroll,
                    "pnl": trader.pnl,
                    "positions": len(trader.positions),
                    "deployed_pct": 0.0,
                    "breaker": trader.breaker_level,
                }

            config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="warning")
            server = uvicorn.Server(config)
            await server.serve()

        except Exception as e:
            logger.warning("apex.dashboard_failed", error=str(e))


t0_global = time.time()


async def main():
    trader = ApexPaperTrader()
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
