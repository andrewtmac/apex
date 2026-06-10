# APEX: Adaptive Prediction EXchange

ML-driven multi-venue trading system built on NautilusTrader.

## Quick Start

```bash
docker compose up -d    # TimescaleDB + Redis
uv sync                 # Install dependencies
apex collect            # Collect historical data
apex train              # Train models
apex paper              # Start paper trading
```
