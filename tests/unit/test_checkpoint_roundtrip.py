"""Tests for checkpoint save/load round trip."""

from __future__ import annotations

import numpy as np
import torch

from agentic_forecaster.agents.model_agent import FittedModel
from agentic_forecaster.models import AttentionLSTM
from agentic_forecaster.models.checkpoint import load_model_bundle, save_model_bundle


def test_checkpoint_roundtrip(tmp_path):
    model = AttentionLSTM(input_size=5, hidden_size=8, num_layers=1, dropout=0.0)
    fitted = FittedModel(
        name="attention_lstm", kind="torch", model=model,
        ticker="TEST", fold="fold_0",
        metrics={"accuracy": 0.5},
        calibration={"method": "temperature", "temperature": 1.5},
        temperature=1.5,
        feature_names=["a", "b", "c", "d", "e"],
        train_config={"hidden_size": 8, "num_layers": 1, "dropout": 0.0},
    )
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(np.random.randn(10, 5))
    fitted._scaler = scaler

    save_model_bundle(fitted, tmp_path / "bundle")
    loaded = load_model_bundle(tmp_path / "bundle")

    assert loaded.ticker == "TEST"
    assert loaded.temperature == 1.5
    assert loaded.feature_names == ["a", "b", "c", "d", "e"]

    x = torch.randn(2, 30, 5)
    p_orig = fitted.predict_proba(x.numpy())
    p_loaded = loaded.predict_proba(x.numpy())
    assert np.allclose(p_orig, p_loaded, atol=1e-5)
