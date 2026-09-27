"""Unit tests for temperature calibration."""

from __future__ import annotations

import numpy as np

from agentic_forecaster.calibration import TemperatureCalibrator


def test_temperature_reduces_nll():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    logits = rng.normal(0, 3, (500, 2))  # over-confident
    cal = TemperatureCalibrator().fit(logits, y)
    assert cal.temperature > 1.0  # over-confident -> T > 1


def test_calibrate_preserves_ranking():
    logits = np.array([[0.1, 0.9], [0.8, 0.2], [0.5, 0.5]])
    cal = TemperatureCalibrator()
    cal.temperature = 2.0
    out = cal.calibrate(logits)
    assert (np.argsort(out, axis=1) == np.argsort(logits, axis=1)).all()
