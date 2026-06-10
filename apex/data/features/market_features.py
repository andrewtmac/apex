"""
Market Microstructure Features (40 features)

Computes price returns, volatility estimators, orderbook metrics, volume
signals, technical indicators, momentum, and trend features from raw
price/orderbook data.

All calculations are pure NumPy/SciPy -- no TA-Lib dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as sp_stats

from apex.data.features.builder import FeatureExtractor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-12  # avoid division by zero


# ---------------------------------------------------------------------------
# Pure-numpy technical indicator helpers
# ---------------------------------------------------------------------------


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(x, _EPS))


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average (full-length output)."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _sma(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average using cumsum for speed."""
    if len(arr) < window:
        return np.full_like(arr, np.nan)
    cs = np.cumsum(arr)
    cs[window:] = cs[window:] - cs[:-window]
    result = np.full_like(arr, np.nan)
    result[window - 1 :] = cs[window - 1 :] / window
    return result


def _rsi(close: np.ndarray, period: int = 14) -> float:
    """Relative Strength Index."""
    if len(close) < period + 1:
        return 50.0
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = _ema(gains, span=period)[-1]
    avg_loss = _ema(losses, span=period)[-1]
    if avg_loss < _EPS:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _macd(close: np.ndarray) -> tuple[float, float, float]:
    """MACD line, signal, histogram."""
    if len(close) < 26:
        return 0.0, 0.0, 0.0
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    signal = _ema(macd_line, 9)
    hist = macd_line - signal
    return float(macd_line[-1]), float(signal[-1]), float(hist[-1])


def _bollinger(close: np.ndarray, period: int = 20) -> tuple[float, float]:
    """Upper and lower Bollinger bands (distance from mid as fraction)."""
    if len(close) < period:
        return 0.0, 0.0
    sma = np.mean(close[-period:])
    std = np.std(close[-period:], ddof=1)
    if sma < _EPS:
        return 0.0, 0.0
    upper = (sma + 2 * std - close[-1]) / sma
    lower = (close[-1] - (sma - 2 * std)) / sma
    return float(upper), float(lower)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Average True Range."""
    if len(close) < period + 1:
        return 0.0
    prev_close = close[:-1]
    h = high[1:]
    lo = low[1:]
    tr = np.maximum(h - lo, np.maximum(np.abs(h - prev_close), np.abs(lo - prev_close)))
    atr_arr = _ema(tr, span=period)
    return float(atr_arr[-1])


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Average Directional Index."""
    n = len(close)
    if n < period + 2:
        return 0.0
    up_move = np.diff(high)
    down_move = -np.diff(low)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_val = _atr(high, low, close, period)
    if atr_val < _EPS:
        return 0.0
    plus_di = float(_ema(plus_dm, period)[-1]) / atr_val * 100
    minus_di = float(_ema(minus_dm, period)[-1]) / atr_val * 100
    denom = plus_di + minus_di
    if denom < _EPS:
        return 0.0
    dx = abs(plus_di - minus_di) / denom * 100
    return float(dx)


def _cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> float:
    """Commodity Channel Index."""
    if len(close) < period:
        return 0.0
    tp = (high + low + close) / 3.0
    tp_window = tp[-period:]
    sma = np.mean(tp_window)
    mad = np.mean(np.abs(tp_window - sma))
    if mad < _EPS:
        return 0.0
    return float((tp[-1] - sma) / (0.015 * mad))


def _stochastic(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3
) -> tuple[float, float]:
    """Stochastic %K and %D."""
    if len(close) < k_period:
        return 50.0, 50.0
    hh = np.max(high[-k_period:])
    ll = np.min(low[-k_period:])
    denom = hh - ll
    if denom < _EPS:
        k = 50.0
    else:
        k = float((close[-1] - ll) / denom * 100)
    # %D is SMA of %K -- approximate with just %K here for single point
    # In practice we'd compute a series; for a snapshot feature this is fine.
    d = k  # simplified
    return k, d


def _williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Williams %R."""
    if len(close) < period:
        return -50.0
    hh = np.max(high[-period:])
    ll = np.min(low[-period:])
    denom = hh - ll
    if denom < _EPS:
        return -50.0
    return float((hh - close[-1]) / denom * -100)


# ---------------------------------------------------------------------------
# Volatility estimators
# ---------------------------------------------------------------------------


def _realized_vol(log_returns: np.ndarray, window: int) -> float:
    if len(log_returns) < window:
        return 0.0
    return float(np.std(log_returns[-window:], ddof=1))


def _parkinson_vol(high: np.ndarray, low: np.ndarray, window: int | None = None) -> float:
    """Parkinson (1980) high-low volatility estimator."""
    if window:
        high = high[-window:]
        low = low[-window:]
    if len(high) < 2:
        return 0.0
    hl = _safe_log(high) - _safe_log(low)
    return float(np.sqrt(np.mean(hl**2) / (4.0 * np.log(2.0))))


def _garman_klass_vol(
    o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray, window: int | None = None
) -> float:
    """Garman-Klass (1980) OHLC volatility estimator."""
    if window:
        o, h, lo, c = o[-window:], h[-window:], lo[-window:], c[-window:]
    if len(c) < 2:
        return 0.0
    log_hl = _safe_log(h) - _safe_log(lo)
    log_co = _safe_log(c) - _safe_log(o)
    gk = 0.5 * log_hl**2 - (2.0 * np.log(2.0) - 1.0) * log_co**2
    return float(np.sqrt(np.mean(gk)))


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

# Approximate bar counts at 1-minute resolution
_BARS_1M = 1
_BARS_5M = 5
_BARS_1H = 60
_BARS_4H = 240
_BARS_24H = 1440


class MarketFeatureExtractor(FeatureExtractor):
    """Computes 40 market-microstructure features.

    Expected keys in *raw_data*::

        close : list[float]        # 1-min close prices (most recent last)
        high  : list[float]        # 1-min highs
        low   : list[float]        # 1-min lows
        open  : list[float]        # 1-min opens
        volume: list[float]        # 1-min volumes
        timestamps: list[float]    # epoch seconds per bar

        # Orderbook snapshot
        best_bid: float
        best_ask: float
        bid_sizes: list[float]     # sizes at top N bid levels
        ask_sizes: list[float]     # sizes at top N ask levels
        bid_prices: list[float]    # prices at top N bid levels
        ask_prices: list[float]    # prices at top N ask levels
    """

    _NAMES: list[str] = [
        # Returns (5)
        "log_ret_1m",
        "log_ret_5m",
        "log_ret_1h",
        "log_ret_4h",
        "log_ret_24h",
        # Volatility (4)
        "realized_vol_1h",
        "realized_vol_24h",
        "parkinson_vol",
        "garman_klass_vol",
        # Orderbook (5)
        "bid_ask_spread",
        "mid_price",
        "book_imbalance_L1",
        "weighted_mid",
        "microprice",
        # Volume (4)
        "volume_ratio_5m_1h",
        "vwap_deviation",
        "obv_slope",
        "volume_zscore",
        # Technical (11)
        "rsi_14",
        "macd_signal",
        "macd_histogram",
        "bollinger_upper",
        "bollinger_lower",
        "atr_14",
        "adx_14",
        "cci_20",
        "stochastic_k",
        "stochastic_d",
        "williams_r",
        # Momentum (4)
        "momentum_5m",
        "momentum_1h",
        "momentum_4h",
        "rate_of_change_1h",
        # Trend (4)
        "sma_20_deviation",
        "ema_12_deviation",
        "ema_26_deviation",
        "trend_strength",
        # Misc (3)
        "bar_count_1h",
        "time_since_last_trade",
        "high_low_range",
    ]

    def feature_names(self) -> list[str]:
        return list(self._NAMES)

    async def extract(
        self,
        market_id: str,
        venue: str,
        raw_data: dict[str, Any],
    ) -> dict[str, float]:
        close = np.asarray(raw_data.get("close", []), dtype=np.float64)
        high = np.asarray(raw_data.get("high", []), dtype=np.float64)
        low = np.asarray(raw_data.get("low", []), dtype=np.float64)
        opn = np.asarray(raw_data.get("open", []), dtype=np.float64)
        volume = np.asarray(raw_data.get("volume", []), dtype=np.float64)
        timestamps = np.asarray(raw_data.get("timestamps", []), dtype=np.float64)

        n = len(close)
        feat: dict[str, float] = {}

        # ----- Returns -----
        log_close = _safe_log(close) if n > 0 else np.array([])

        def _log_ret(bars: int) -> float:
            if n < bars + 1:
                return 0.0
            return float(log_close[-1] - log_close[-1 - bars])

        feat["log_ret_1m"] = _log_ret(_BARS_1M)
        feat["log_ret_5m"] = _log_ret(_BARS_5M)
        feat["log_ret_1h"] = _log_ret(_BARS_1H)
        feat["log_ret_4h"] = _log_ret(_BARS_4H)
        feat["log_ret_24h"] = _log_ret(_BARS_24H)

        # ----- Volatility -----
        log_returns = np.diff(log_close) if n > 1 else np.array([])
        feat["realized_vol_1h"] = _realized_vol(log_returns, _BARS_1H)
        feat["realized_vol_24h"] = _realized_vol(log_returns, _BARS_24H)
        feat["parkinson_vol"] = _parkinson_vol(high, low, window=_BARS_24H)
        feat["garman_klass_vol"] = _garman_klass_vol(opn, high, low, close, window=_BARS_24H)

        # ----- Orderbook -----
        best_bid = float(raw_data.get("best_bid", 0.0))
        best_ask = float(raw_data.get("best_ask", 0.0))
        bid_sizes = np.asarray(raw_data.get("bid_sizes", [0.0]), dtype=np.float64)
        ask_sizes = np.asarray(raw_data.get("ask_sizes", [0.0]), dtype=np.float64)
        bid_prices = np.asarray(raw_data.get("bid_prices", [best_bid]), dtype=np.float64)
        ask_prices = np.asarray(raw_data.get("ask_prices", [best_ask]), dtype=np.float64)

        mid = (best_bid + best_ask) / 2.0 if (best_bid + best_ask) > 0 else (close[-1] if n else 0.0)
        spread = best_ask - best_bid

        feat["bid_ask_spread"] = spread / mid if mid > _EPS else 0.0
        feat["mid_price"] = mid

        total_bid_size = float(np.sum(bid_sizes))
        total_ask_size = float(np.sum(ask_sizes))
        imbal_denom = total_bid_size + total_ask_size
        feat["book_imbalance_L1"] = (
            (total_bid_size - total_ask_size) / imbal_denom if imbal_denom > _EPS else 0.0
        )

        # Weighted mid: weigh each side by opposite size
        if imbal_denom > _EPS:
            feat["weighted_mid"] = (
                best_bid * total_ask_size + best_ask * total_bid_size
            ) / imbal_denom
        else:
            feat["weighted_mid"] = mid

        # Microprice (volume-weighted mid using top-of-book)
        l1_bid_sz = float(bid_sizes[0]) if len(bid_sizes) > 0 else 0.0
        l1_ask_sz = float(ask_sizes[0]) if len(ask_sizes) > 0 else 0.0
        l1_denom = l1_bid_sz + l1_ask_sz
        feat["microprice"] = (
            (best_bid * l1_ask_sz + best_ask * l1_bid_sz) / l1_denom
            if l1_denom > _EPS
            else mid
        )

        # ----- Volume -----
        if n >= _BARS_1H and np.sum(volume[-_BARS_1H:]) > _EPS:
            vol_5m = np.sum(volume[-_BARS_5M:]) if n >= _BARS_5M else 0.0
            vol_1h = np.sum(volume[-_BARS_1H:])
            feat["volume_ratio_5m_1h"] = vol_5m / vol_1h * (_BARS_1H / _BARS_5M)
        else:
            feat["volume_ratio_5m_1h"] = 1.0

        # VWAP deviation
        if n >= _BARS_1H and np.sum(volume[-_BARS_1H:]) > _EPS:
            vwap = np.sum(close[-_BARS_1H:] * volume[-_BARS_1H:]) / np.sum(volume[-_BARS_1H:])
            feat["vwap_deviation"] = (close[-1] - vwap) / vwap if vwap > _EPS else 0.0
        else:
            feat["vwap_deviation"] = 0.0

        # OBV slope
        if n > 1:
            obv = np.cumsum(np.where(np.diff(close) > 0, volume[1:], -volume[1:]))
            if len(obv) >= 10:
                x = np.arange(10, dtype=np.float64)
                slope, _, _, _, _ = sp_stats.linregress(x, obv[-10:])
                feat["obv_slope"] = float(slope)
            else:
                feat["obv_slope"] = 0.0
        else:
            feat["obv_slope"] = 0.0

        # Volume z-score (current bar vs 1h mean/std)
        if n >= _BARS_1H:
            vol_window = volume[-_BARS_1H:]
            vol_mean = np.mean(vol_window)
            vol_std = np.std(vol_window, ddof=1)
            feat["volume_zscore"] = (
                float((volume[-1] - vol_mean) / vol_std) if vol_std > _EPS else 0.0
            )
        else:
            feat["volume_zscore"] = 0.0

        # ----- Technical indicators -----
        feat["rsi_14"] = _rsi(close, 14)

        _macd_line, _macd_sig, _macd_hist = _macd(close)
        feat["macd_signal"] = _macd_sig
        feat["macd_histogram"] = _macd_hist

        boll_up, boll_lo = _bollinger(close, 20)
        feat["bollinger_upper"] = boll_up
        feat["bollinger_lower"] = boll_lo

        feat["atr_14"] = _atr(high, low, close, 14)
        feat["adx_14"] = _adx(high, low, close, 14)
        feat["cci_20"] = _cci(high, low, close, 20)

        stoch_k, stoch_d = _stochastic(high, low, close)
        feat["stochastic_k"] = stoch_k
        feat["stochastic_d"] = stoch_d
        feat["williams_r"] = _williams_r(high, low, close)

        # ----- Momentum -----
        def _momentum(bars: int) -> float:
            if n < bars + 1 or close[-1 - bars] < _EPS:
                return 0.0
            return float((close[-1] - close[-1 - bars]) / close[-1 - bars])

        feat["momentum_5m"] = _momentum(_BARS_5M)
        feat["momentum_1h"] = _momentum(_BARS_1H)
        feat["momentum_4h"] = _momentum(_BARS_4H)

        if n >= _BARS_1H + 1 and close[-1 - _BARS_1H] > _EPS:
            feat["rate_of_change_1h"] = float(
                (close[-1] / close[-1 - _BARS_1H] - 1.0) * 100
            )
        else:
            feat["rate_of_change_1h"] = 0.0

        # ----- Trend -----
        if n >= 26:
            sma20 = _sma(close, 20)
            ema12 = _ema(close, 12)
            ema26 = _ema(close, 26)
            cur = close[-1]

            feat["sma_20_deviation"] = (
                float((cur - sma20[-1]) / sma20[-1]) if np.isfinite(sma20[-1]) and sma20[-1] > _EPS else 0.0
            )
            feat["ema_12_deviation"] = (
                float((cur - ema12[-1]) / ema12[-1]) if ema12[-1] > _EPS else 0.0
            )
            feat["ema_26_deviation"] = (
                float((cur - ema26[-1]) / ema26[-1]) if ema26[-1] > _EPS else 0.0
            )

            # Trend strength: slope of SMA20 over last 10 bars
            sma20_tail = sma20[-10:]
            if len(sma20_tail) == 10 and np.all(np.isfinite(sma20_tail)):
                x = np.arange(10, dtype=np.float64)
                slope, _, _, _, _ = sp_stats.linregress(x, sma20_tail)
                feat["trend_strength"] = float(slope / (np.mean(sma20_tail) + _EPS))
            else:
                feat["trend_strength"] = 0.0
        else:
            feat["sma_20_deviation"] = 0.0
            feat["ema_12_deviation"] = 0.0
            feat["ema_26_deviation"] = 0.0
            feat["trend_strength"] = 0.0

        # ----- Misc -----
        feat["bar_count_1h"] = float(min(n, _BARS_1H))

        if len(timestamps) >= 2:
            feat["time_since_last_trade"] = float(timestamps[-1] - timestamps[-2])
        else:
            feat["time_since_last_trade"] = 0.0

        if n > 0:
            range_window = min(n, _BARS_24H)
            feat["high_low_range"] = float(
                (np.max(high[-range_window:]) - np.min(low[-range_window:]))
                / mid
                if mid > _EPS
                else 0.0
            )
        else:
            feat["high_low_range"] = 0.0

        return feat
