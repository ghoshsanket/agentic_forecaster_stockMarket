"""Tests for intraday -> daily resampling."""

from __future__ import annotations

import pandas as pd

from agentic_forecaster.data.resampling import resample_intraday_to_daily


def test_minute_to_daily_produces_one_row_per_date():
    rows = []
    for hour in range(9, 12):
        rows.append({
            "date": f"2024-01-02 {hour:02d}:15:00",
            "open": 100.0 + hour,
            "high": 101.0 + hour,
            "low": 99.0 + hour,
            "close": 100.5 + hour,
            "volume": 1000,
        })
    rows.append({
        "date": "2024-01-03 09:15:00",
        "open": 110.0, "high": 111.0, "low": 109.0, "close": 110.5, "volume": 2000,
    })
    df = pd.DataFrame(rows)
    daily = resample_intraday_to_daily(df)
    assert len(daily) == 2
    assert daily.iloc[0]["open"] == 109.0
    assert daily.iloc[0]["high"] == 112.0
    assert daily.iloc[0]["low"] == 108.0
    assert daily.iloc[0]["close"] == 111.5
    assert daily.iloc[0]["volume"] == 3000
    assert daily.iloc[1]["open"] == 110.0


def test_daily_ohlcv_correctness():
    df = pd.DataFrame({
        "date": ["2024-01-02 09:15:00", "2024-01-02 10:00:00", "2024-01-02 15:30:00"],
        "open": [100.0, 101.0, 102.0],
        "high": [101.0, 103.0, 104.0],
        "low": [99.0, 100.0, 101.0],
        "close": [100.5, 102.0, 103.0],
        "volume": [100, 200, 300],
    })
    daily = resample_intraday_to_daily(df)
    assert len(daily) == 1
    row = daily.iloc[0]
    assert row["open"] == 100.0
    assert row["high"] == 104.0
    assert row["low"] == 99.0
    assert row["close"] == 103.0
    assert row["volume"] == 600
