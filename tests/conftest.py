"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture
def synthetic_dataset():
    from agentic_forecaster.data import make_synthetic_dataset

    return make_synthetic_dataset(n_tickers=2, n_days=300, seed=42)


@pytest.fixture
def tiny_config():
    return {
        "experiment": {"name": "test", "seed": 42},
        "data": {
            "synthetic": True,
            "n_tickers": 2,
            "n_days": 300,
            "sequence_length": 10,
            "train_fraction": 0.7,
            "val_fraction": 0.15,
        },
        "features": {
            "indicators": ["rsi_14", "macd", "macd_signal", "macd_histogram", "atr_14", "realized_volatility_20", "log_return"],
            "use_ohlcv": True,
            "drop_na": True,
        },
        "models": {
            "attention_lstm": {"hidden_size": 8, "num_layers": 1, "dropout": 0.0,
                               "learning_rate": 1e-3, "batch_size": 16, "epochs": 2, "patience": 2},
            "lstm": {"hidden_size": 8, "num_layers": 1, "dropout": 0.0,
                     "learning_rate": 1e-3, "batch_size": 16, "epochs": 2, "patience": 2},
            "random_forest": {"n_estimators": 10, "max_depth": 3},
            "logistic_regression": {"C": 1.0, "max_iter": 100},
            "majority": {},
        },
        "calibration": {"method": "temperature"},
        "evaluation": {"ece_bins": 5},
        "explainability": {"background_samples": 10, "top_k_features": 3},
        "risk": {"kelly_fraction": 0.25, "max_position_pct": 0.10},
        "reporting": {"formats": ["html"], "llm": {"enabled": False}},
    }
