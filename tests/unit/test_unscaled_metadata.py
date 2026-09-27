"""Unscaled metadata must come from the retained snapshot, not inverse transform."""

from __future__ import annotations

import numpy as np

from agentic_forecaster.data.agent import DataAgent

REQUIRED = ("close", "atr_14", "rsi_14", "macd", "macd_signal",
            "realized_volatility_20")


def _config():
    return {
        "experiment": {"seed": 42},
        "data": {
            "synthetic": True, "n_tickers": 1, "n_days": 300,
            "sequence_length": 10,
            "train_fraction": 0.7, "val_fraction": 0.15,
        },
        "features": {
            "indicators": ["rsi_14", "macd", "macd_signal", "macd_histogram",
                           "atr_14", "realized_volatility_20", "log_return"],
            "use_ohlcv": True, "drop_na": True,
        },
    }


def test_snapshot_contains_required_reporting_features():
    agent = DataAgent(_config())
    ds = agent.run(ticker="SYN00")
    snap = agent.origin_snapshot(ds.test, -1)
    for name in REQUIRED:
        assert name in snap, f"missing unscaled {name}"
        assert np.isfinite(snap[name])


def test_snapshot_is_unscaled_not_standardised():
    agent = DataAgent(_config())
    ds = agent.run(ticker="SYN00")
    snap = agent.origin_snapshot(ds.test, -1)
    # A StandardScaler would place the training mean near 0 and the scale near 1.
    # Real prices/volatilities/ATR are on their natural scales.
    assert snap["close"] > 1.0
    assert snap["atr_14"] > 0.0
    assert 0.0 <= snap["rsi_14"] <= 100.0
    assert snap["realized_volatility_20"] < 1.0


def test_snapshot_corresponds_to_origin_date():
    agent = DataAgent(_config())
    ds = agent.run(ticker="SYN00")
    index = 0
    snap = agent.origin_snapshot(ds.test, index)
    assert snap  # non-empty
    # The snapshot row is the feature row AT the origin, not at origin+1.
    raw_row = ds.test.unscaled[index]
    close_idx = ds.test.unscaled_feature_names.index("close")
    assert snap["close"] == pytest_approx(float(raw_row[close_idx]))


def pytest_approx(value: float, tol: float = 1e-9):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) <= tol
    return _Approx()
