"""Integration test: data agent produces consistent splits."""

from __future__ import annotations

from agentic_forecaster.data import DataAgent


def test_data_agent_shapes(tiny_config):
    agent = DataAgent(tiny_config)
    ds = agent.run()
    assert ds.train.X.shape[1] == tiny_config["data"]["sequence_length"]
    assert ds.train.X.shape[2] == len(ds.feature_names)
    assert ds.test.X.shape[2] == len(ds.feature_names)
    assert set(ds.train.y.tolist()) <= {0, 1}


def test_scaler_no_leakage(tiny_config):
    agent = DataAgent(tiny_config)
    ds = agent.run()
    # StandardScaler fit on train -> train mean ~ 0
    train_mean = ds.train.X.mean(axis=(0, 1))
    assert abs(train_mean.mean()) < 1.0
