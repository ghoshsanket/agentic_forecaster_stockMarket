"""Tests that training uses max 3 epochs and device propagates."""

from __future__ import annotations

import numpy as np
import torch

from agentic_forecaster.models import AttentionLSTM
from agentic_forecaster.training import Trainer, resolve_device


def test_resolve_device_cpu():
    device = resolve_device("cpu")
    assert device == torch.device("cpu")


def test_resolve_device_auto():
    device = resolve_device("auto")
    assert device.type in ("cpu", "cuda")


def test_training_runs_3_epochs_max():
    model = AttentionLSTM(input_size=5, hidden_size=8, num_layers=1, dropout=0.0)
    trainer = Trainer(model, epochs=3, batch_size=16, patience=10, device="cpu")
    X = np.random.randn(100, 30, 5).astype(np.float32)
    y = np.random.randint(0, 2, 100).astype(np.float32)
    result = trainer.fit(X, y, X[:20], y[:20])
    assert result.best_epoch < 3
    assert len(result.history["train_loss"]) <= 3
