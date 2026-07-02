# APEX V2: Kalshi Autonomous Trading System
## Mission: $1,000 → $1,000,000 in 365 Days

---

## Executive Summary

APEX V2 is a multi-agent autonomous trading system built for Kalshi event
contracts. It builds on the existing APEX infrastructure (Kalshi API integration,
circuit breaker, dashboard, Telegram) but adds specialized strategy agents,
a self-improvement learning system, and hourly reporting.

**Starting capital:** $1,000 (paper) → live next week
**Target:** $1,000,000 (1000x in 365 days)
**Required daily compound rate:** ~1.9%/day (ln(1000)/365)
**Primary edges:** Weather forecasting, crypto momentum, macro events

---

## Architecture: Multi-Agent System

```
┌─────────────────────────────────────────────────────┐
│                 SUPERVISOR AGENT                     │
│  Orchestrates all agents, manages state, persists    │
├─────────┬──────────┬──────────┬──────────┬──────────┤
│ SCANNER │ ANALYZER │ EXECUTOR │  RISK    │ LEARNER  │
│ Agent   │ Agent    │ Agent    │ Agent    │ Agent    │
├─────────┴──────────┴──────────┴──────────┴──────────┤
│              SHARED STATE (paper_state.json)          │
├──────────────────────────────────────────────────────┤
│         KALSHI API  │  WEATHER APIs  │  CRYPTO APIs  │
└──────────────────────────────────────────────────────┘
         │                                    │
    ┌────┴────┐                          ┌────┴────┐
    │Dashboard│                          │Telegram │
    │  :8080  │                          │ Hourly  │
    └─────────┘                          └─────────┘
```

### Agent Responsibilities

1. **Supervisor** — Main loop, state persistence, agent coordination
2. **Scanner** — Continuously scans Kalshi for tradeable markets
3. **Analyzer** — Evaluates signals using multi-signal fusion
4. **Executor** — Manages position sizing, entry, exit, and orders
5. **Risk** — Circuit breaker, drawdown limits, portfolio constraints
6. **Learner** — Tracks outcomes, adjusts strategy weights, improves over time

---

## Strategy Deep Dive

### STRATEGY 1: Weather Temperature (PRIMARY — highest edge)

**Why it works:**
- Kalshi weather markets (KXHIGHNY, KXHIGHCHI, etc.) settle on actual
  NOAA temperature readings — deterministic resolution
- NWS/NOAA forecasts are freely available and highly accurate (±1-2°F
  within 24h, ±3-5°F within 72h)
- Market prices often lag forecast updates by 30-60 minutes
- Volume is $300K+/day across weather series — deep liquidity

**Signal generation:**
1. Fetch NWS forecast for each city (api.weather.gov)
2. Compare forecast high temp to market range prices
3. If forecast says 88°F but market for "85-90°F" is priced at 0.40,
   and "above 85°F" is at 0.55, the edge = forecast confidence - market price
4. Weight by: forecast confidence, days until resolution, historical accuracy

**Data sources:**
- NWS API (api.weather.gov) — free, unlimited
- Open-Meteo API — free, historical + forecast
- Weather Underground — backup
- Historical accuracy tracking: compare past forecasts to actuals

**Edge window:** 24-72 hours before resolution (forecasts most accurate)

**Position sizing:** Up to 12% of bankroll per weather trade (highest confidence)

### STRATEGY 2: Crypto Range Markets (SECONDARY)

**Why it works:**
- Kalshi BTC/ETH range markets settle on CoinGecko/CoinMarketCap prices
- Crypto has strong momentum and mean-reversion patterns
- On-chain data (whale movements, exchange flows) leads price by hours
- Fear & Greed index, funding rates provide additional signals

**Signal generation:**
1. Fetch BTC/ETH price from CoinGecko API
2. Calculate momentum indicators (RSI, MACD, Bollinger Bands)
3. Check on-chain metrics (exchange net flow, whale alerts)
4. Cross-reference with Fear & Greed index
5. Compare model probability to Kalshi range market prices

**Data sources:**
- CoinGecko API (free tier: 10-30 calls/min)
- Alternative.me Fear & Greed Index (free)
- Blockchain.com exchange flow data
- Binance funding rates (free)

**Edge window:** 1-24 hours (crypto moves fast)

**Position sizing:** Up to 8% of bankroll per crypto trade

### STRATEGY 3: Macro Events (TERTIARY)

**Why it works:**
- Fed rate decisions, CPI, GDP, jobless claims have predictable
  analyst consensus before release
- Markets often misprice tail outcomes
- Resolution is binary and fast

**Signal generation:**
1. Track FedWatch tool probabilities
2. Compare consensus estimates to market prices
3. Look for overpriced/underpriced tail outcomes
4. Use historical surprise distributions

**Position sizing:** Up to 5% of bankroll per macro trade

### STRATEGY 4: Sports (EXISTING — maintain, don't expand)

- Keep existing sports validation from ESPN + Odds API
- Sports are lower edge, use as portfolio diversifier only
- Max 3% of bankroll per sports trade

---

## Self-Improvement System

### How the Bot Learns

The Learner agent maintains a **strategy performance database** that tracks:

1. **Per-strategy metrics:**
   - Win rate, avg profit, avg loss, Sharpe ratio
   - Best/worst market conditions
   - Time-of-day performance
   - Edge accuracy (predicted vs actual)

2. **Auto-adjustments (every 24h):**
   - Increase weight for strategies with Sharpe > 1.5
   - Decrease weight for strategies with Sharpe < 0.5
   - Adjust MIN_EDGE threshold based on recent accuracy
   - Tighten/loosen stop-loss based on volatility regime

3. **Regime detection:**
   - HIGH_VOL: widen stops, reduce sizing, favor crypto
   - LOW_VOL: tighten stops, increase sizing, favor weather
   - TRENDING: momentum strategies get priority
   - MEAN_REVERTING: contrarian strategies get priority

4. **Feature importance tracking:**
   - Which signals actually predict outcomes?
   - Drop features that don't contribute
   - Add new signals from successful trades

### Learning Loop

```
Every 24 hours:
  1. Collect all resolved trades from last 24h
  2. Calculate per-strategy metrics
  3. Update strategy weights via Thompson Sampling
  4. Retrain edge calibration model if 50+ new data points
  5. Log learnings and send summary to Telegram
  6. Persist updated weights to strategy_weights.json
```

---

## Risk Management

### Position Limits
- Max single position: 12% of bankroll (weather, highest confidence)
- Max single position: 8% (crypto)
- Max single position: 5% (macro)
- Max single position: 3% (sports)
- Max open positions: 15
- Max deployed capital: 70% of equity

### Circuit Breaker (existing, tuned)
- GREEN: Normal (0-10% drawdown)
- YELLOW: 10% DD or 5 consec losses — 50% sizing reduction
- ORANGE: 20% DD or 8 consec losses — stop new positions
- RED: 30% DD — close non-hedged
- BLACK: 40% DD — emergency liquidation

### Daily Limits
- Max daily loss: 15% of bankroll → pause until next day
- Max daily trades: 20 (prevent overtrading)
- Cooldown after 3 consecutive losses: 2 hours

### Kelly Criterion (Modified)
- Use fractional Kelly (0.25x) for position sizing
- Cap at 12% regardless of Kelly output
- Scale by circuit breaker multiplier
- Scale by strategy confidence weight

---

## Data Pipeline

### Real-Time Feeds (every 2 minutes)
- Kalshi market scan (all series)
- Weather forecasts (NWS API)
- Crypto prices (CoinGecko)
- Live ESPN scores (sports validation)

### Periodic Feeds (every 30 minutes)
- Bookmaker odds (The Odds API)
- On-chain metrics
- Fear & Greed index

### Daily Feeds
- Historical weather actuals vs forecasts (accuracy tracking)
- Kalshi market resolutions (outcome tracking)
- Strategy performance recalculation

---

## Telegram Reporting

### Hourly Update Format
```
📊 APEX Hourly Update
━━━━━━━━━━━━━━━━━━━
💰 Equity: $1,234 (+23.4%)
💵 Cash: $456 | Deployed: $778 (63%)
📈 Today: +$45 (+3.8%) | 3W-1L

🔥 Top Positions:
  BUY  NYC high >85°F  +$23
  SELL BTC >$110K      +$12
  BUY  Fed holds       -$8

🧠 Learnings:
  Weather 24h edge improved 2%
  Crypto momentum signals weakening

⚠️ Risk: GREEN | DD: 2.1%
⏱ Cycle: 847 | Signals: 1,234
```

### Daily Digest (midnight UTC)
- Full P&L breakdown
- Best/worst trades
- Strategy performance comparison
- Learning insights
- Next day's focus areas

---

## Implementation Phases

### Phase 1: Foundation (NOW — paper trading week)
- [x] Existing Kalshi API integration
- [x] Circuit breaker system
- [x] Dashboard infrastructure
- [ ] Weather strategy agent (NWS API integration)
- [ ] Crypto strategy agent (CoinGecko + indicators)
- [ ] Self-improvement learner
- [ ] Hourly Telegram reporting
- [ ] New dashboard with strategy breakdown
- [ ] Reset to $1,000 bankroll

### Phase 2: Optimization (days 3-7)
- [ ] Tune weather signal accuracy
- [ ] Add on-chain crypto signals
- [ ] Thompson sampling for strategy weights
- [ ] Regime detection
- [ ] Walk-forward validation on historical data

### Phase 3: Live Trading (week 2)
- [ ] Switch to live Kalshi API
- [ ] Start with $500 live + $500 paper
- [ ] Monitor slippage and execution quality
- [ ] Gradual capital increase as confidence grows

### Phase 4: Scale (weeks 3+)
- [ ] Add more Kalshi series as volume grows
- [ ] Implement cross-market arbitrage
- [ ] Add limit orders for better fills
- [ ] Scale position sizes with bankroll growth

---

## Key Metrics to Track

| Metric | Target | Measurement |
|--------|--------|-------------|
| Daily ROI | 1.9%/day | equity change / equity |
| Win rate | >60% | wins / total trades |
| Sharpe ratio | >2.0 | annualized |
| Max drawdown | <20% | peak-to-trough |
| Avg edge accuracy | >5% | predicted - actual |
| Trades/day | 5-15 | prevent overtrading |
| Weather accuracy | >65% | correct forecasts / total |
| Crypto accuracy | >55% | correct signals / total |

---

## Files to Create

```
scripts/
  apex_v2.py          — Main V2 orchestrator (Supervisor)
  weather_agent.py    — Weather strategy with NWS API
  crypto_agent.py     — Crypto strategy with indicators
  learner.py          — Self-improvement system
  reporter.py         — Hourly Telegram reporting
  dashboard_v2.html   — New dashboard frontend
```

## API Keys Needed (all free)
- NWS API: None required (free, unlimited)
- CoinGecko: Free tier (10-30 calls/min)
- Alternative.me: None required (free)
- Kalshi: Already configured
- Telegram: Already configured
