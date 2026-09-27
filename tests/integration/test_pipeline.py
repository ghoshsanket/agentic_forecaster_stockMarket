"""Integration test: per-ticker model training on synthetic data."""

from __future__ import annotations

from agentic_forecaster.agents.model_agent import ModelAgent
from agentic_forecaster.data import DataAgent


def test_per_ticker_training(tiny_config):
    data_agent = DataAgent(tiny_config)
    model_agent = ModelAgent(tiny_config)
    ds = data_agent.run(ticker="SYN00")
    fitted = model_agent.train_ticker("SYN00", ds, device="cpu")
    assert fitted.ticker == "SYN00"
    assert fitted.model is not None
    assert "accuracy" in fitted.metrics
    assert fitted.temperature > 0


def test_different_tickers_different_models(tiny_config):
    data_agent = DataAgent(tiny_config)
    model_agent = ModelAgent(tiny_config)
    ds1 = data_agent.run(ticker="SYN00")
    ds2 = data_agent.run(ticker="SYN01")
    f1 = model_agent.train_ticker("SYN00", ds1, device="cpu")
    f2 = model_agent.train_ticker("SYN01", ds2, device="cpu")
    assert f1.model is not f2.model
    assert f1.ticker == "SYN00"
    assert f2.ticker == "SYN01"
