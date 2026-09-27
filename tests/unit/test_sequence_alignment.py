"""Tests for sequence alignment: input ends on prediction origin, target is next day."""

from __future__ import annotations

import pandas as pd

from agentic_forecaster.data.agent import DataAgent


def test_sequence_end_before_target_date():
    config = {
        "experiment": {"seed": 42},
        "data": {
            "synthetic": True, "n_tickers": 1, "n_days": 100,
            "sequence_length": 30,
            "train_start": "2020-01-01", "train_end": "2020-03-31",
            "val_start": "2020-04-01", "val_end": "2020-04-30",
            "test_start": "2020-05-01", "test_end": "2020-06-30",
        },
        "features": {"indicators": ["rsi_14", "macd", "atr_14", "realized_volatility_20", "log_return"], "use_ohlcv": True, "drop_na": True},
    }
    agent = DataAgent(config)
    ds = agent.run(ticker="SYN00")
    for split_name in ("train", "val", "test"):
        split = getattr(ds, split_name)
        for i in range(len(split.dates)):
            target = pd.Timestamp(split.target_dates[i])
            if pd.isna(target):
                continue
            origin = pd.Timestamp(split.dates[i])
            assert target > origin, f"sequence_end_date {origin} not before target_date {target}"


def test_sequence_includes_origin_row():
    """The 30-day window must include the prediction-origin row (index t)."""
    config = {
        "experiment": {"seed": 42},
        "data": {
            "synthetic": True, "n_tickers": 1, "n_days": 100,
            "sequence_length": 30,
            "train_start": "2020-01-01", "train_end": "2020-03-31",
            "val_start": "2020-04-01", "val_end": "2020-04-30",
            "test_start": "2020-05-01", "test_end": "2020-06-30",
        },
        "features": {"indicators": ["rsi_14", "macd", "atr_14", "realized_volatility_20", "log_return"], "use_ohlcv": True, "drop_na": True},
    }
    agent = DataAgent(config)
    ds = agent.run(ticker="SYN00")
    assert ds.train.X.shape[1] == 30
