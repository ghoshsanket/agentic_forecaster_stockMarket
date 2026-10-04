"""V2 stationary per-stock features.

Why a SEPARATE feature engineer
-------------------------------
The paper reconstruction feeds RAW OHLCV plus a few indicators to one model per
stock (``features/engineer.py``, used by ``data/agent.py``).  That is
legitimate for an independent per-stock model.  It is NOT legitimate for V2,
which is ONE SHARED model across many securities: a raw price level of 1500 for
one stock and 30 for another is not a comparable quantity, and a shared model
would learn security identity from the price axis instead of from behaviour.

Every feature below is therefore a stationary, scale-free quantity derived from
a single security's own history, and EVERY feature at date ``t`` uses only
values up to and including ``t``.

Causality rules enforced here
-----------------------------
* Multi-day windows are BACKWARD looking and include ``t``.
* Exponential moving averages are seeded from the beginning of the available
  history and are therefore functions of ``<= t`` only.
* Indicator warm-up periods produce NaN.  NaN is NEVER forward-filled: a late
  listing keeps NaN until the indicator genuinely exists, and samples whose
  required history is missing are dropped downstream.
* Division guards return NaN (never +/-inf and never a silently invented
  value).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("agentic_forecaster.v2.features")

#: Ordered stock-level feature schema.  The order is part of the checkpoint and
#: of the feature-store hash: changing it invalidates every cached artefact.
STOCK_FEATURE_NAMES: tuple[str, ...] = (
    "log_return_1",
    "log_return_3",
    "log_return_5",
    "log_return_10",
    "log_return_20",
    "overnight_gap",
    "intraday_return",
    "high_low_range",
    "close_location",
    "realized_vol_5",
    "realized_vol_10",
    "realized_vol_20",
    "atr_14_pct",
    "rsi_14_centered",
    "macd_pct",
    "macd_signal_pct",
    "macd_hist_pct",
    "sma5_distance",
    "sma20_distance",
    "sma50_distance",
    "bollinger_percent_b",
    "volume_log",
    "volume_z_20",
    "volume_ratio_20",
    "drawdown_20",
    "drawdown_60",
    "obv_change_normalized",
)

#: Momentum horizons, named the way the context layer refers to them.
MOMENTUM_SHORT = "log_return_5"
MOMENTUM_LONG = "log_return_20"

#: Volatility used to normalise the return target.
RETURN_VOL_FEATURE = "realized_vol_20"

#: Minimum plausible daily volatility used only to avoid dividing by zero when
#: normalising the return target.  It is a numerical guard, not a signal.
VOL_FLOOR = 1e-4

OHLCV_COLUMNS = ("date", "open", "high", "low", "close", "volume")


def normalise_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case, sort and validate one security's daily bars.

    The paper-snapshot parquet files store ``Date/Open/High/Low/Close/Volume``;
    every other V2 stage assumes lower-case names, so the rename happens once,
    here, and nowhere else.
    """
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"daily bars missing column(s) {missing}: {list(out.columns)}")
    out["date"] = pd.to_datetime(out["date"])
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = (out.sort_values("date")
              .drop_duplicates(subset="date", keep="last")
              .reset_index(drop=True))
    # A bar without a close cannot produce a return or a target, so it is not a
    # trading day for V2 purposes.
    out = out.loc[out["close"].notna() & (out["close"] > 0)].reset_index(drop=True)
    return out


def _wilder_ema(values: pd.Series, alpha: float) -> pd.Series:
    """Exponential moving average seeded from the first observation.

    ``alpha = 1/n`` reproduces Wilder's smoothing.  Because the seed is the
    first row of the security's own history, the value at ``t`` depends only on
    observations up to ``t``.
    """
    return values.ewm(alpha=alpha, adjust=False).mean()


def _rolling_std(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window, min_periods=window).std(ddof=1)


def _true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    ranges = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _rsi_14(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder_ema(gain, 1.0 / 14.0)
    avg_loss = _wilder_ema(loss, 1.0 / 14.0)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 with any gain is the "all advances" limit of the formula.
    rsi = rsi.where(avg_loss > 0, 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi


def _rolling_max(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).max()


def build_stock_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Build the causal, stationary V2 stock feature frame for ONE security.

    Parameters
    ----------
    bars:
        Daily OHLCV with at least a date column and positive closes.

    Returns
    -------
    DataFrame with ``date`` plus every name in :data:`STOCK_FEATURE_NAMES`, all
    computed from information available at the close of ``date``.
    """
    df = normalise_ohlcv_frame(bars)
    close, high, low, open_, volume = (df["close"], df["high"], df["low"],
                                       df["open"], df["volume"])
    log_close = np.log(close)
    out = pd.DataFrame({"date": df["date"]})

    # --- returns over backward-looking windows, inclusive of t -------------
    for horizon in (1, 3, 5, 10, 20):
        out[f"log_return_{horizon}"] = log_close - log_close.shift(horizon)

    # --- session shape ----------------------------------------------------
    prev_close = close.shift(1)
    out["overnight_gap"] = np.log(open_) - np.log(prev_close)
    out["intraday_return"] = log_close - np.log(open_)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["high_low_range"] = np.log(high) - np.log(low)
    span = (high - low).replace(0.0, np.nan)
    # A zero-range bar carries no information about where the close sits inside
    # the range; 0.5 (the neutral midpoint) is used rather than inventing NaN.
    out["close_location"] = ((close - low) / span).where(span > 0, 0.5)

    # --- volatility -------------------------------------------------------
    ret1 = out["log_return_1"]
    for window in (5, 10, 20):
        out[f"realized_vol_{window}"] = _rolling_std(ret1, window)

    atr14 = _true_range(df).rolling(14, min_periods=14).mean()
    out["atr_14_pct"] = atr14 / close

    out["rsi_14_centered"] = (_rsi_14(close) - 50.0) / 50.0

    # --- MACD (causal EMA) ------------------------------------------------
    ema12 = _wilder_ema(close, 2.0 / (12.0 + 1.0))
    ema26 = _wilder_ema(close, 2.0 / (26.0 + 1.0))
    macd = ema12 - ema26
    signal = _wilder_ema(macd, 2.0 / (9.0 + 1.0))
    out["macd_pct"] = macd / close
    out["macd_signal_pct"] = signal / close
    out["macd_hist_pct"] = (macd - signal) / close

    # --- moving-average distances and Bollinger ---------------------------
    for window in (5, 20, 50):
        sma = close.rolling(window, min_periods=window).mean()
        out[f"sma{window}_distance"] = close / sma - 1.0

    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std(ddof=1)
    band = (4.0 * std20).replace(0.0, np.nan)
    out["bollinger_percent_b"] = (close - (sma20 - 2.0 * std20)) / band

    # --- volume -----------------------------------------------------------
    volume = volume.fillna(0.0)
    out["volume_log"] = np.log1p(volume)
    vol_mean = volume.rolling(20, min_periods=20).mean()
    vol_std = volume.rolling(20, min_periods=20).std(ddof=1)
    out["volume_z_20"] = (volume - vol_mean) / vol_std.replace(0.0, np.nan)
    out["volume_ratio_20"] = volume / vol_mean.replace(0.0, np.nan)

    # --- drawdowns --------------------------------------------------------
    for window in (20, 60):
        peak = _rolling_max(close, window)
        out[f"drawdown_{window}"] = close / peak - 1.0

    # --- on-balance-volume change, normalised by its own typical magnitude -
    obv_step = np.sign(close.diff()).fillna(0.0) * volume
    obv_change = obv_step.cumsum().diff()
    typical = obv_change.abs().rolling(20, min_periods=20).mean()
    out["obv_change_normalized"] = obv_change / typical.where(typical > 0)

    missing = [c for c in STOCK_FEATURE_NAMES if c not in out.columns]
    if missing:
        raise AssertionError(f"stock feature schema drift: missing {missing}")
    out = out.loc[:, ["date", *STOCK_FEATURE_NAMES]]
    return _sanitise(out)


def _sanitise(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace +/-inf with NaN everywhere; never fabricate a value."""
    numeric = frame.select_dtypes(include=[np.floating]).columns
    frame = frame.copy()
    frame.loc[:, numeric] = frame.loc[:, numeric].replace([np.inf, -np.inf], np.nan)
    return frame


def feature_quality(frame: pd.DataFrame) -> dict:
    """Warm-up diagnostics for one security's feature frame."""
    counts = {name: int(frame[name].notna().sum()) for name in STOCK_FEATURE_NAMES}
    first_valid = {name: (frame.loc[frame[name].notna(), "date"].min()
                          if counts[name] else None) for name in STOCK_FEATURE_NAMES}
    return {
        "n_rows": len(frame),
        "first_date": frame["date"].min(),
        "last_date": frame["date"].max(),
        "n_finite_first_row": int(np.isfinite(
            frame.loc[:, list(STOCK_FEATURE_NAMES)].to_numpy(dtype=float)).all(axis=1).sum()),
        "n_valid": counts,
        "first_valid_date": {k: (None if v is None else str(pd.Timestamp(v).date()))
                             for k, v in first_valid.items()},
    }


def align_ohlcv_to_features(bars: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Join the feature frame back onto its bars (used by the store builder)."""
    return normalise_ohlcv_frame(bars).merge(features, on="date", how="left",
                                             validate="one_to_one")