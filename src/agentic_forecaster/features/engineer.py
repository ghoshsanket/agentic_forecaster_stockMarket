"""Technical-indicator feature engineering.

All indicators are computed per ticker on the OHLCV frame.  The target is the
next-day close-to-close direction: 1 if ``close[t+1] > close[t]`` else 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(50.0)


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def bollinger(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: upper, lower, width."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    width = (upper - lower) / sma.replace(0.0, np.nan)
    return upper, lower, width


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def build_feature_frame(
    df: pd.DataFrame,
    indicators: list[str] | None = None,
) -> pd.DataFrame:
    """Build the full feature frame for one ticker.

    Parameters
    ----------
    df : DataFrame with columns ``date, open, high, low, close, volume``.
    indicators : optional list of indicator names to compute.  If ``None``,
        all available indicators are computed.
    """
    out = pd.DataFrame()
    out["date"] = df["date"]
    close = df["close"].astype(float)

    all_indicators = {
        "rsi_14": lambda: rsi(close, 14),
        "macd": lambda: macd(close)[0],
        "macd_signal": lambda: macd(close)[1],
        "macd_hist": lambda: macd(close)[2],
        "atr_14": lambda: atr(df["high"], df["low"], close, 14),
        "volatility_20": lambda: close.pct_change().rolling(20).std(),
        "sma_20": lambda: close.rolling(20).mean(),
        "sma_50": lambda: close.rolling(50).mean(),
        "ema_12": lambda: close.ewm(span=12, adjust=False).mean(),
        "ema_26": lambda: close.ewm(span=26, adjust=False).mean(),
        "bb_upper": lambda: bollinger(close)[0],
        "bb_lower": lambda: bollinger(close)[1],
        "bb_width": lambda: bollinger(close)[2],
        "obv": lambda: obv(close, df["volume"].astype(float)),
        "returns_1": lambda: close.pct_change(1),
        "returns_5": lambda: close.pct_change(5),
        "returns_10": lambda: close.pct_change(10),
        "log_volume": lambda: np.log1p(df["volume"].astype(float)),
    }

    selected = indicators if indicators is not None else list(all_indicators)
    for name in selected:
        if name not in all_indicators:
            raise ValueError(f"Unknown indicator: {name!r}")
        out[name] = all_indicators[name]()

    # Target: next-day direction
    out["target"] = (close.shift(-1) > close).astype(float)
    out.loc[out.index[-1], "target"] = np.nan
    return out
