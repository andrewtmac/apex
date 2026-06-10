# APEX: Adaptive Prediction EXchange

## What This Is

ML-driven multi-venue trading bot built on NautilusTrader. Fundamentally different from epik-trade:
- **Trained ML models** (XGBoost, LightGBM, LSTM, PPO) instead of LLM prompting
- **Python-first** on NautilusTrader's Cython/Rust execution engine
- **7-model ensemble** with Bayesian calibration and Thompson sampling
- **165+ engineered features** across 7 domain groups

## Platforms

- Polymarket (prediction markets)
- Kalshi (event contracts)
- TastyTrade (options, equities, futures)

## Quick Start

```bash
# Start infrastructure
docker compose up -d

# Install dependencies
uv sync

# Collect historical data
apex collect

# Train models
apex train

# Start paper trading
apex paper

# Start dashboard
apex dashboard
```

## Key Commands

- **Build/check:** `uv run python -m pytest tests/`
- **Type check:** `uv run mypy apex/`
- **Lint:** `uv run ruff check apex/`
- **Dashboard:** `uv run apex dashboard` (port 8080)

## Architecture

```
apex/
  config.py           -- Central config (env vars, risk params)
  data/ingestion/     -- Real-time data collectors (8 sources)
  data/features/      -- Feature engineering pipeline (165+ features)
  data/store.py       -- Feature store (TimescaleDB + Redis)
  data/streams.py     -- Redis Streams pub/sub bus
  models/             -- 7 ML models + registry
  ensemble/           -- Signal combination + trade gate
  strategies/         -- NautilusTrader strategy implementations
  risk/               -- Position sizing, CVaR, circuit breakers
  research/           -- Alpha discovery, backtesting
  training/           -- Model retraining pipeline
  monitoring/         -- Dashboard, alerts, performance
  node.py             -- Main TradingNode entry point
  cli.py              -- CLI interface
```

## Infrastructure

- TimescaleDB on port 5433 (separate from epik-trade's 5432)
- Redis on port 6380 (separate from epik-trade's 6379)
- Both via docker-compose

## Risk Rules (NEVER modify without explicit approval)

- Tiered circuit breakers: GREEN -> YELLOW -> ORANGE -> RED -> BLACK
- Max single bet: 12% (CALM regime), scales down to 3% (CRISIS)
- Max deployed: 85% (CALM), scales down to 25% (CRISIS)
- Drawdown halt: 30% (CALM), scales down to 10% (CRISIS)
- Portfolio CVaR limit: 5% daily at 95% confidence
- All new strategies must pass walk-forward validation (Sharpe > 1.0 OOS)

## Git Workflow

Push directly to main. Single-operator bot.
