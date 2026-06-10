"""
APEX Central Configuration

Loads environment variables from .env and exposes typed dataclass configs
for every subsystem: venues, risk regimes, data sources, infrastructure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TradingMode(str, Enum):
    LIVE = "LIVE"
    PAPER = "PAPER"
    DRY_RUN = "DRY_RUN"


class Regime(str, Enum):
    CALM = "CALM"
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    CRISIS = "CRISIS"


# ---------------------------------------------------------------------------
# Venue configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolymarketConfig:
    api_key: str = ""
    private_key: str = ""
    proxy_wallet: str = ""
    clob_url: str = "https://clob.polymarket.com"
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass(frozen=True)
class KalshiConfig:
    api_key: str = ""
    private_key_path: str = ""  # path to RSA PEM file


@dataclass(frozen=True)
class TastyTradeConfig:
    username: str = ""
    password: str = ""
    sandbox: bool = True

    @property
    def base_url(self) -> str:
        return (
            "https://api.cert.tastyworks.com"
            if self.sandbox
            else "https://api.tastyworks.com"
        )


@dataclass(frozen=True)
class VenueConfig:
    """Aggregated venue configuration."""

    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    tastytrade: TastyTradeConfig = field(default_factory=TastyTradeConfig)


# ---------------------------------------------------------------------------
# Risk regime parameters
# ---------------------------------------------------------------------------

_REGIME_DEFAULTS: dict[Regime, dict[str, float]] = {
    Regime.CALM: {
        "kelly_fraction": 0.25,
        "max_deployed_pct": 0.85,
        "max_single_bet_pct": 0.12,
        "drawdown_halt_pct": 0.30,
    },
    Regime.NORMAL: {
        "kelly_fraction": 0.20,
        "max_deployed_pct": 0.75,
        "max_single_bet_pct": 0.10,
        "drawdown_halt_pct": 0.25,
    },
    Regime.ELEVATED: {
        "kelly_fraction": 0.10,
        "max_deployed_pct": 0.50,
        "max_single_bet_pct": 0.06,
        "drawdown_halt_pct": 0.15,
    },
    Regime.CRISIS: {
        "kelly_fraction": 0.05,
        "max_deployed_pct": 0.25,
        "max_single_bet_pct": 0.03,
        "drawdown_halt_pct": 0.10,
    },
}


@dataclass(frozen=True)
class RegimeRiskParams:
    """Risk parameters for a single regime."""

    kelly_fraction: float
    max_deployed_pct: float
    max_single_bet_pct: float
    drawdown_halt_pct: float


@dataclass(frozen=True)
class RiskConfig:
    """Risk configuration across all regimes."""

    regimes: dict[Regime, RegimeRiskParams] = field(default_factory=dict)

    def params_for(self, regime: Regime) -> RegimeRiskParams:
        return self.regimes[regime]


def _build_risk_config() -> RiskConfig:
    regimes = {
        regime: RegimeRiskParams(**params)
        for regime, params in _REGIME_DEFAULTS.items()
    }
    return RiskConfig(regimes=regimes)


# ---------------------------------------------------------------------------
# Data source configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSourceConfig:
    newsapi_ai_key: str = ""
    odds_api_key: str = ""
    polygon_rpc_url: str = ""
    anthropic_api_key: str = ""


# ---------------------------------------------------------------------------
# Infrastructure configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InfraConfig:
    redis_url: str = "redis://localhost:6380"
    database_url: str = "postgresql://apex:apex@localhost:5433/apex"
    log_level: str = "info"


@dataclass(frozen=True)
class NotificationConfig:
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApexConfig:
    """Master configuration combining all sub-configs."""

    trading_mode: TradingMode = TradingMode.PAPER
    venues: VenueConfig = field(default_factory=VenueConfig)
    risk: RiskConfig = field(default_factory=_build_risk_config)
    data_sources: DataSourceConfig = field(default_factory=DataSourceConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)

    @property
    def is_live(self) -> bool:
        return self.trading_mode == TradingMode.LIVE


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes")


def load_config(env_path: Optional[str | Path] = None) -> ApexConfig:
    """
    Load configuration from environment / .env file.

    Parameters
    ----------
    env_path : path to .env file.  If *None* the function walks up from CWD.
    """
    if env_path:
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(override=True)

    polymarket = PolymarketConfig(
        api_key=_env("POLYMARKET_API_KEY"),
        private_key=_env("POLYMARKET_PRIVATE_KEY"),
        proxy_wallet=_env("POLYMARKET_PROXY_WALLET"),
        clob_url=_env("POLYMARKET_CLOB_URL", "https://clob.polymarket.com"),
        ws_url=_env("POLYMARKET_WS_URL", "wss://ws-subscriptions-clob.polymarket.com/ws/market"),
    )

    kalshi = KalshiConfig(
        api_key=_env("KALSHI_API_KEY"),
        private_key_path=_env("KALSHI_PRIVATE_KEY"),
    )

    tastytrade = TastyTradeConfig(
        username=_env("TASTYTRADE_USERNAME"),
        password=_env("TASTYTRADE_PASSWORD"),
        sandbox=_env_bool("TASTYTRADE_SANDBOX", default=True),
    )

    venues = VenueConfig(
        polymarket=polymarket,
        kalshi=kalshi,
        tastytrade=tastytrade,
    )

    data_sources = DataSourceConfig(
        newsapi_ai_key=_env("NEWSAPI_AI_KEY"),
        odds_api_key=_env("ODDS_API_KEY"),
        polygon_rpc_url=_env("POLYGON_RPC_URL"),
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
    )

    infra = InfraConfig(
        redis_url=_env("REDIS_URL", "redis://localhost:6380"),
        database_url=_env("DATABASE_URL", "postgresql://apex:apex@localhost:5433/apex"),
        log_level=_env("LOG_LEVEL", "info"),
    )

    notifications = NotificationConfig(
        telegram_bot_token=_env("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_env("TELEGRAM_CHAT_ID"),
    )

    mode_str = _env("TRADING_MODE", "PAPER").upper()
    try:
        trading_mode = TradingMode(mode_str)
    except ValueError:
        trading_mode = TradingMode.PAPER

    return ApexConfig(
        trading_mode=trading_mode,
        venues=venues,
        risk=_build_risk_config(),
        data_sources=data_sources,
        infra=infra,
        notifications=notifications,
    )
