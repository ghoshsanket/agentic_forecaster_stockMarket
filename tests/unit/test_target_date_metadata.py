"""Tests for target_date metadata integrity (item 5)."""

from __future__ import annotations

import pandas as pd

from agentic_forecaster.data.agent import DataAgent


def _config():
    return {
        "experiment": {"seed": 42},
        "data": {
            "synthetic": True, "n_tickers": 1, "n_days": 120,
            "sequence_length": 10,
            "train_start": "2020-01-01", "train_end": "2020-03-31",
            "val_start": "2020-04-01", "val_end": "2020-04-30",
            "test_start": "2020-05-01", "test_end": "2020-06-30",
        },
        "features": {
            "indicators": ["rsi_14", "macd", "macd_signal", "macd_histogram",
                           "atr_14", "realized_volatility_20", "log_return"],
            "use_ohlcv": True, "drop_na": True,
        },
    }


def test_no_labeled_sample_has_nat_target_date():
    ds = DataAgent(_config()).run(ticker="SYN00")
    for name in ("train", "val", "test"):
        split = getattr(ds, name)
        if len(split.dates) == 0:
            continue
        targets = pd.to_datetime(split.target_dates)
        assert not targets.isna().any(), f"{name} contains a NaT target_date"


def test_final_valid_labeled_sample_has_real_target_date():
    """The LAST labeled sample must still carry a real next trading date."""
    ds = DataAgent(_config()).run(ticker="SYN00")
    split = ds.test
    assert len(split.dates) > 0
    last_origin = pd.Timestamp(split.dates[-1])
    last_target = pd.Timestamp(split.target_dates[-1])
    assert not pd.isna(last_target)
    assert last_target > last_origin


def test_target_date_is_next_actual_trading_day():
    ds = DataAgent(_config()).run(ticker="SYN00")
    split = ds.train
    origins = pd.to_datetime(split.dates)
    targets = pd.to_datetime(split.target_dates)
    for origin, target in zip(origins[:20], targets[:20]):
        assert target > origin
        # consecutive within this per-ticker series
        assert (target - origin).days >= 1
