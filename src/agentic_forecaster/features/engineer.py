"""Technical-indicator feature engineering.

All indicators are computed per ticker on the DAILY OHLCV frame.  The target
is the next-day close-to-close direction: 1 if ``close[t+1] > close[t]`` else
0.  The final daily observation has no label (NA).

Phase-1 canonical feature set (11 features):
    open, high, low, close, volume,
    log_return, realized_volatility_20, rsi_14,
    macd, macd_signal, macd_histogram, atr_14

Extra indicators (SMA, EMA, Bollinger, OBV, multi-period returns) remain
implemented for future experimentation but are NOT enabled in
``configs/paper.yaml``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PHASE1_FEATURES = [
    "open", "high", "low", "close", "volume",
    "log_return", "realized_volatility_20", "rsi_14",
    "macd", "macd_signal", "macd_histogram", "atr_14",
]


def log_return(close: pd.Series) -> pd.Series:
    """Log return: log(close[t] / close[t-1])."""
    return np.log(close / close.shift(1))


def realized_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation of LOG returns."""
    return log_return(close).rolling(window).std()


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI with Wilder's / RMA smoothing (alpha = 1/period).

    This is the textbook definition and the default used by the Phase-1
    reconstruction.  It is retained under an explicit name because the
    publication's equations are ambiguous and the recovery search must be able
    to compare RSI variants.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return _rsi_from_averages(avg_gain, avg_loss)


def rsi_rolling(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI from a SIMPLE rolling average of gains and losses.

    This matches the displayed equations in the publication literally: a rolling
    arithmetic mean of the up-moves and the down-moves over ``period`` bars,
    rather than an exponentially smoothed average.  Causal: only past and
    current bars are used.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    return _rsi_from_averages(avg_gain, avg_loss)


def rsi_ema(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI from an EMA-smoothed average of gains and losses."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    return _rsi_from_averages(avg_gain, avg_loss)


def _rsi_from_averages(avg_gain: pd.Series, avg_loss: pd.Series) -> pd.Series:
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(50.0)


#: RSI variants offered to the recovery search.  R1 is the Phase-1 default.
RSI_METHODS: dict[str, str] = {
    "R0_rolling": "rsi_rolling",
    "R1_wilder": "rsi_wilder",
    "R2_ema": "rsi_ema",
}


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing) - the Phase-1 default."""
    return rsi_wilder(close, period)


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


def bollinger_percent_b(close: pd.Series, period: int = 20,
                        num_std: float = 2.0) -> pd.Series:
    """Bollinger %B: where the close sits inside the band, 0 = lower, 1 = upper.

    %B = (close - lower) / (upper - lower).  Causal; uses a trailing window.
    """
    mid = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std(ddof=0)
    upper, lower = mid + num_std * sd, mid - num_std * sd
    width = (upper - lower).replace(0.0, np.nan)
    return ((close - lower) / width).fillna(0.5)


def sma5_minus_sma20(close: pd.Series) -> pd.Series:
    """Distance between the 5- and 20-bar simple moving averages, normalised by
    the 20-bar average so the scale is comparable across price levels."""
    fast = close.rolling(5, min_periods=5).mean()
    slow = close.rolling(20, min_periods=20).mean()
    return (fast - slow) / slow.replace(0.0, np.nan)


def build_feature_frame(
    df: pd.DataFrame,
    indicators: list[str] | None = None,
    use_ohlcv: bool = True,
) -> pd.DataFrame:
    """Build the feature frame for one ticker from DAILY OHLCV.

    Parameters
    ----------
    df : DataFrame with columns ``date, open, high, low, close, volume``.
    indicators : optional list of indicator names to compute.  If ``None``,
        the Phase-1 canonical set is used.
    use_ohlcv : if True, include raw OHLCV columns as features.
    """
    out = pd.DataFrame()
    out["date"] = df["date"]
    close = df["close"].astype(float)

    all_indicators = {
        # --- RSI variants (recovery search) ---
        "rsi_14": lambda: rsi_wilder(close, 14),
        "rsi_14_rolling": lambda: rsi_rolling(close, 14),
        "rsi_14_ema": lambda: rsi_ema(close, 14),
        "macd": lambda: macd(close)[0],
        "macd_signal": lambda: macd(close)[1],
        "macd_histogram": lambda: macd(close)[2],
        "atr_14": lambda: atr(df["high"], df["low"], close, 14),
        "realized_volatility_20": lambda: realized_volatility(close, 20),
        "log_return": lambda: log_return(close),
        "sma_5": lambda: close.rolling(5, min_periods=5).mean(),
        "sma_20": lambda: close.rolling(20, min_periods=20).mean(),
        "sma_50": lambda: close.rolling(50, min_periods=50).mean(),
        "sma_5_minus_sma_20": lambda: sma5_minus_sma20(close),
        "ema_12": lambda: close.ewm(span=12, adjust=False).mean(),
        "ema_26": lambda: close.ewm(span=26, adjust=False).mean(),
        "bb_upper": lambda: bollinger(close)[0],
        "bb_lower": lambda: bollinger(close)[1],
        "bb_width": lambda: bollinger(close)[2],
        "bb_percent_b": lambda: bollinger_percent_b(close),
        "obv": lambda: obv(close, df["volume"].astype(float)),
        "returns_1": lambda: close.pct_change(1),
        "returns_5": lambda: close.pct_change(5),
        "returns_10": lambda: close.pct_change(10),
        "log_volume": lambda: np.log1p(df["volume"].astype(float)),
    }

    if use_ohlcv:
        for col in ("open", "high", "low", "close", "volume"):
            out[col] = df[col].astype(float)

    selected = indicators if indicators is not None else PHASE1_FEATURES
    for name in selected:
        if name not in all_indicators:
            raise ValueError(f"Unknown indicator: {name!r}")
        out[name] = all_indicators[name]()

    # Target: next-day direction (PAPER-DEFINED)
    out["target"] = (close.shift(-1) > close).astype(float)
    out.loc[out.index[-1], "target"] = np.nan
    return out
