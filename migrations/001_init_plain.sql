-- APEX Database Schema (Plain PostgreSQL -- no TimescaleDB required)
-- Time-series tables use BRIN indexes for efficient range scans

-- ============================================================
-- Core Trading Tables
-- ============================================================

CREATE TABLE IF NOT EXISTS markets (
    id              TEXT PRIMARY KEY,
    venue           TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    title           TEXT,
    category        TEXT,
    resolution_date TIMESTAMPTZ,
    status          TEXT DEFAULT 'active',
    outcome         SMALLINT,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_markets_venue_status ON markets (venue, status);
CREATE INDEX IF NOT EXISTS idx_markets_category ON markets (category);

CREATE TABLE IF NOT EXISTS signals (
    id              BIGSERIAL PRIMARY KEY,
    time            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    market_id       TEXT NOT NULL,
    venue           TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    direction       TEXT NOT NULL,
    edge            DOUBLE PRECISION NOT NULL,
    edge_ci_lower   DOUBLE PRECISION,
    edge_ci_upper   DOUBLE PRECISION,
    ensemble_score  DOUBLE PRECISION,
    regime          TEXT,
    model_outputs   JSONB DEFAULT '{}',
    accepted        BOOLEAN DEFAULT FALSE,
    reject_reason   TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_time ON signals USING BRIN (time);
CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals (strategy, time DESC);

CREATE TABLE IF NOT EXISTS trades (
    id              TEXT PRIMARY KEY,
    time            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    market_id       TEXT NOT NULL,
    venue           TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    direction       TEXT NOT NULL,
    side            TEXT NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    quantity        DOUBLE PRECISION NOT NULL,
    cost            DOUBLE PRECISION NOT NULL,
    fee             DOUBLE PRECISION DEFAULT 0,
    external_id     TEXT,
    signal_id       BIGINT,
    metadata        JSONB DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_trades_time ON trades USING BRIN (time);
CREATE INDEX IF NOT EXISTS idx_trades_venue ON trades (venue, time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades (strategy, time DESC);

CREATE TABLE IF NOT EXISTS positions (
    id              TEXT PRIMARY KEY,
    market_id       TEXT NOT NULL,
    venue           TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    direction       TEXT NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    current_price   DOUBLE PRECISION,
    quantity        DOUBLE PRECISION NOT NULL,
    cost_basis      DOUBLE PRECISION NOT NULL,
    unrealized_pnl  DOUBLE PRECISION DEFAULT 0,
    realized_pnl    DOUBLE PRECISION DEFAULT 0,
    status          TEXT DEFAULT 'open',
    opened_at       TIMESTAMPTZ DEFAULT NOW(),
    closed_at       TIMESTAMPTZ,
    exit_price      DOUBLE PRECISION,
    metadata        JSONB DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions (status, venue);

-- ============================================================
-- Time-Series Tables (with BRIN indexes for range queries)
-- ============================================================

CREATE TABLE IF NOT EXISTS price_ticks (
    time            TIMESTAMPTZ NOT NULL,
    venue           TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    bid             DOUBLE PRECISION,
    ask             DOUBLE PRECISION,
    mid             DOUBLE PRECISION NOT NULL,
    volume          DOUBLE PRECISION DEFAULT 0,
    metadata        JSONB
);
CREATE INDEX IF NOT EXISTS idx_price_ticks_time ON price_ticks USING BRIN (time);
CREATE INDEX IF NOT EXISTS idx_price_ticks_lookup ON price_ticks (venue, symbol, time DESC);

CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    time            TIMESTAMPTZ NOT NULL,
    venue           TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    bids            JSONB NOT NULL,
    asks            JSONB NOT NULL,
    spread          DOUBLE PRECISION,
    imbalance       DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_orderbook_time ON orderbook_snapshots USING BRIN (time);

CREATE TABLE IF NOT EXISTS feature_store (
    time            TIMESTAMPTZ NOT NULL,
    entity_id       TEXT NOT NULL,
    feature_set     TEXT NOT NULL,
    features        JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feature_store_time ON feature_store USING BRIN (time);
CREATE INDEX IF NOT EXISTS idx_feature_store_entity ON feature_store (entity_id, feature_set, time DESC);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    time            TIMESTAMPTZ NOT NULL,
    total_equity    DOUBLE PRECISION NOT NULL,
    poly_equity     DOUBLE PRECISION DEFAULT 0,
    kalshi_equity   DOUBLE PRECISION DEFAULT 0,
    tt_equity       DOUBLE PRECISION DEFAULT 0,
    open_positions  INT DEFAULT 0,
    deployed_pct    DOUBLE PRECISION DEFAULT 0,
    drawdown_pct    DOUBLE PRECISION DEFAULT 0,
    cvar_95         DOUBLE PRECISION,
    regime          TEXT,
    breaker_level   TEXT DEFAULT 'GREEN'
);
CREATE INDEX IF NOT EXISTS idx_portfolio_time ON portfolio_snapshots USING BRIN (time);

CREATE TABLE IF NOT EXISTS weather_observations (
    time            TIMESTAMPTZ NOT NULL,
    city            TEXT NOT NULL,
    station         TEXT,
    temp_f          DOUBLE PRECISION NOT NULL,
    humidity        DOUBLE PRECISION,
    wind_mph        DOUBLE PRECISION,
    source          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_weather_time ON weather_observations USING BRIN (time);

CREATE TABLE IF NOT EXISTS model_performance (
    time            TIMESTAMPTZ NOT NULL,
    model_name      TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    metric_name     TEXT NOT NULL,
    metric_value    DOUBLE PRECISION NOT NULL,
    metadata        JSONB DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_model_perf_time ON model_performance USING BRIN (time);
CREATE INDEX IF NOT EXISTS idx_model_perf_lookup ON model_performance (model_name, metric_name, time DESC);

CREATE TABLE IF NOT EXISTS strategy_performance (
    time            TIMESTAMPTZ NOT NULL,
    strategy        TEXT NOT NULL,
    venue           TEXT NOT NULL,
    trades_count    INT DEFAULT 0,
    wins            INT DEFAULT 0,
    losses          INT DEFAULT 0,
    gross_pnl       DOUBLE PRECISION DEFAULT 0,
    fees            DOUBLE PRECISION DEFAULT 0,
    net_pnl         DOUBLE PRECISION DEFAULT 0,
    sharpe          DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION DEFAULT 0,
    avg_edge        DOUBLE PRECISION,
    brier_score     DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_strategy_perf_time ON strategy_performance USING BRIN (time);

-- ============================================================
-- Historical resolved markets (for training)
-- ============================================================

CREATE TABLE IF NOT EXISTS resolved_markets (
    id              TEXT PRIMARY KEY,
    venue           TEXT NOT NULL,
    slug            TEXT,
    title           TEXT NOT NULL,
    category        TEXT,
    outcome         SMALLINT NOT NULL,       -- 1=YES, 0=NO
    final_price     DOUBLE PRECISION,
    volume          DOUBLE PRECISION DEFAULT 0,
    created_at      TIMESTAMPTZ,
    resolved_at     TIMESTAMPTZ,
    metadata        JSONB DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_resolved_venue ON resolved_markets (venue, category);
CREATE INDEX IF NOT EXISTS idx_resolved_time ON resolved_markets (resolved_at DESC);
