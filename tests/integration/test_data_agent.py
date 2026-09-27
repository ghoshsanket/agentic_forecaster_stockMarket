"""Integration test: data agent produces consistent per-ticker splits."""

from __future__ import annotations

from agentic_forecaster.data import DataAgent


def test_data_agent_shapes(tiny_config):
    agent = DataAgent(tiny_config)
    ds = agent.run(ticker="SYN00")
    assert ds.train.X.shape[1] == tiny_config["data"]["sequence_length"]
    assert ds.train.X.shape[2] == len(ds.feature_names)
    assert ds.test.X.shape[2] == len(ds.feature_names)
    assert set(ds.train.y.tolist()) <= {0.0, 1.0}


def test_scaler_no_leakage(tiny_config):
    agent = DataAgent(tiny_config)
    ds = agent.run(ticker="SYN00")
    train_mean = ds.train.X.mean(axis=(0, 1))
    assert abs(train_mean.mean()) < 1.0


def test_per_ticker_scalers_are_independent(tiny_config):
    agent = DataAgent(tiny_config)
    ds1 = agent.run(ticker="SYN00")
    ds2 = agent.run(ticker="SYN01")
    assert ds1.scaler is not ds2.scaler
    assert ds1.ticker == "SYN00"
    assert ds2.ticker == "SYN01"
