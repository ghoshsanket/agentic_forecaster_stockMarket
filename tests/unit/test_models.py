"""Unit tests for model definitions (binary logit output)."""

from __future__ import annotations

import torch

from agentic_forecaster.models import AttentionLSTM, PlainLSTM


def test_attention_lstm_forward():
    model = AttentionLSTM(input_size=5, hidden_size=8, num_layers=1, dropout=0.0)
    x = torch.randn(4, 10, 5)
    logits, weights = model(x, return_attention=True)
    assert logits.shape == (4,)
    assert weights.shape == (4, 10)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)


def test_plain_lstm_forward():
    model = PlainLSTM(input_size=5, hidden_size=8, num_layers=1, dropout=0.0)
    x = torch.randn(4, 10, 5)
    logits = model(x)
    assert logits.shape == (4,)


def test_attention_lstm_backward():
    model = AttentionLSTM(input_size=5, hidden_size=8, num_layers=1, dropout=0.0)
    x = torch.randn(4, 10, 5)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_predict_proba_sigmoid():
    model = AttentionLSTM(input_size=5, hidden_size=8, num_layers=1, dropout=0.0)
    x = torch.randn(4, 10, 5)
    p = model.predict_proba(x)
    assert p.shape == (4,)
    assert (p >= 0).all() and (p <= 1).all()
