"""Split-boundary label-leakage tests (item 6)."""

from __future__ import annotations

import pandas as pd

from agentic_forecaster.data.agent import DataAgent

BOUNDARIES = {
    "train_start": "2016-01-01",
    "train_end": "2020-12-31",
    "val_start": "2021-01-01",
    "val_end": "2021-12-31",
    "test_start": "2022-01-01",
    "test_end": "2022-12-31",
}


def _config():
    return {
        "experiment": {"seed": 42},
        "data": {
            "synthetic": True, "n_tickers": 1, "n_days": 1600,
            "sequence_length": 10,
            "train_start": BOUNDARIES["train_start"],
            "train_end": BOUNDARIES["train_end"],
            "val_start": BOUNDARIES["val_start"],
            "val_end": BOUNDARIES["val_end"],
            "test_start": BOUNDARIES["test_start"],
            "test_end": BOUNDARIES["test_end"],
        },
        "features": {
            "indicators": ["rsi_14", "macd", "macd_signal", "macd_histogram",
                           "atr_14", "realized_volatility_20", "log_return"],
            "use_ohlcv": True, "drop_na": True,
        },
    }


def test_train_targets_never_reach_into_validation():
    ds = DataAgent(_config()).run(ticker="SYN00")
    if not len(ds.train.dates):
        return
    targets = pd.to_datetime(ds.train.target_dates)
    origins = pd.to_datetime(ds.train.dates)
    assert targets.max() <= pd.Timestamp(BOUNDARIES["train_end"]), (
        "a training label resolves after the training window (leakage)"
    )
    assert origins.max() <= pd.Timestamp(BOUNDARIES["train_end"])


def test_validation_targets_never_reach_into_test():
    ds = DataAgent(_config()).run(ticker="SYN00")
    if not len(ds.val.dates):
        return
    targets = pd.to_datetime(ds.val.target_dates)
    origins = pd.to_datetime(ds.val.dates)
    assert targets.max() <= pd.Timestamp(BOUNDARIES["val_end"])
    assert targets.min() >= pd.Timestamp(BOUNDARIES["val_start"])
    assert origins.max() <= pd.Timestamp(BOUNDARIES["val_end"])


def test_test_targets_stay_inside_test_window():
    ds = DataAgent(_config()).run(ticker="SYN00")
    if not len(ds.test.dates):
        return
    targets = pd.to_datetime(ds.test.target_dates)
    origins = pd.to_datetime(ds.test.dates)
    assert targets.max() <= pd.Timestamp(BOUNDARIES["test_end"])
    assert origins.min() >= pd.Timestamp(BOUNDARIES["test_start"])


def test_no_overlap_between_splits():
    ds = DataAgent(_config()).run(ticker="SYN00")
    train = set(pd.to_datetime(ds.train.target_dates))
    val = set(pd.to_datetime(ds.val.target_dates))
    test = set(pd.to_datetime(ds.test.target_dates))
    assert not (train & val)
    assert not (val & test)
    assert not (train & test)
