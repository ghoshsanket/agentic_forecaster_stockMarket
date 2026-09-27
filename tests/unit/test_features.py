"""Unit tests for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from agentic_forecaster.features.engineer import (
    atr,
    bollinger,
    build_feature_frame,
    macd,
    obv,
    rsi,
)


def _frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2023-01-01", periods=n),
            "open": close + rng.normal(0, 0.5, n),
            "high": close + abs(rng.normal(0, 1, n)),
            "low": close - abs(rng.normal(0, 1, n)),
            "close": close,
            "volume": rng.integers(1000, 10000, n).astype(float),
        }
    )


def test_rsi_bounds():
    s = rsi(_frame()["close"])
    assert s.dropna().between(0, 100).all()


def test_macd_shapes():
    df = _frame()
    m, sig, hist = macd(df["close"])
    assert len(m) == len(sig) == len(hist) == len(df)


def test_atr_positive():
    df = _frame()
    a = atr(df["high"], df["low"], df["close"])
    assert (a.dropna() >= 0).all()


def test_bollinger_order():
    df = _frame()
    upper, lower, _width = bollinger(df["close"])
    valid = upper.dropna().index
    assert (upper.loc[valid] >= lower.loc[valid]).all()


def test_obv_cumulative():
    df = _frame()
    o = obv(df["close"], df["volume"])
    assert len(o) == len(df)


def test_build_feature_frame_columns():
    df = _frame()
    ff = build_feature_frame(df, indicators=["rsi_14", "macd", "atr_14", "volatility_20", "returns_1"])
    for col in ("rsi_14", "macd", "atr_14", "volatility_20", "returns_1", "target", "date"):
        assert col in ff.columns
    assert ff["target"].dropna().isin([0.0, 1.0]).all()
