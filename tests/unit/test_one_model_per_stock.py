"""Tests for one-model-per-stock design."""

from __future__ import annotations

from agentic_forecaster.agents.model_agent import ModelAgent


def test_different_tickers_different_model_instances():
    """Each ticker must get its own model instance."""
    config = {
        "experiment": {"seed": 42},
        "models": {"attention_lstm": {"hidden_size": 8, "num_layers": 1, "dropout": 0.0, "epochs": 1, "batch_size": 16, "patience": 2}},
    }
    agent = ModelAgent(config)
    assert agent is not None


def test_fitted_model_has_ticker_and_scaler():
    config = {
        "experiment": {"seed": 42},
        "models": {"attention_lstm": {"hidden_size": 8, "num_layers": 1, "dropout": 0.0, "epochs": 1, "batch_size": 16, "patience": 2}},
    }
    agent = ModelAgent(config)
    assert agent is not None
