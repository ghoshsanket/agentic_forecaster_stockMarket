"""Unit tests for evaluation metrics."""

from __future__ import annotations

import numpy as np

from agentic_forecaster.evaluation.metrics import (
    compute_metrics,
    expected_calibration_error,
)


def test_perfect_predictions():
    rng = np.random.default_rng(0)
    n = 2000
    p = rng.uniform(0.01, 0.99, n)
    y = (rng.uniform(size=n) < p).astype(int)
    m = compute_metrics(y, p, n_bins=10)
    assert m["accuracy"] > 0.6
    assert m["brier"] < 0.25
    assert m["ece"] < 0.05


def test_ece_zero_when_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.3, 0.7, 10000)
    y = (rng.uniform(size=10000) < p).astype(int)
    ece = expected_calibration_error(y, p, n_bins=10)
    assert ece < 0.05


def test_compute_metrics_returns_all_keys():
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    m = compute_metrics(y, p)
    assert "accuracy" in m
    assert "brier" in m
    assert "ece" in m
    assert "f1" in m
