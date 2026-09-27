"""Tests for next-day target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from agentic_forecaster.features.engineer import build_feature_frame


def test_target_next_day_known_closes():
    """With closes 100, 101, 99 the targets must be 1, 0, NA."""
    df = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=3),
        "open": [100.0, 101.0, 99.0],
        "high": [101.0, 102.0, 100.0],
        "low": [99.0, 100.0, 98.0],
        "close": [100.0, 101.0, 99.0],
        "volume": [1000, 1000, 1000],
    })
    ff = build_feature_frame(df, indicators=["rsi_14"])
    assert ff["target"].iloc[0] == 1.0
    assert ff["target"].iloc[1] == 0.0
    assert np.isnan(ff["target"].iloc[2])


def test_target_is_next_day_not_next_minute():
    """Target must be based on daily closes, not intraday."""
    df = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=5),
        "open": [100.0] * 5,
        "high": [101.0] * 5,
        "low": [99.0] * 5,
        "close": [100.0, 100.0, 100.0, 100.0, 100.0],
        "volume": [1000] * 5,
    })
    ff = build_feature_frame(df, indicators=["rsi_14"])
    assert ff["target"].iloc[0] == 0.0
