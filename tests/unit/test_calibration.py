"""Unit tests for temperature calibration (binary logit)."""

from __future__ import annotations

import numpy as np

from agentic_forecaster.calibration import TemperatureCalibrator


def test_temperature_reduces_nll():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    logits = rng.normal(0, 3, 500)
    cal = TemperatureCalibrator().fit(logits, y)
    assert cal.temperature > 1.0


def test_calibrate_preserves_ranking():
    p = np.array([0.1, 0.5, 0.9])
    cal = TemperatureCalibrator()
    cal.temperature = 2.0
    out = cal.calibrate_proba(p)
    assert (np.argsort(out) == np.argsort(p)).all()
