"""Tests that calibration is actually applied at inference."""

from __future__ import annotations

import numpy as np

from agentic_forecaster.calibration import TemperatureCalibrator


def test_calibrated_differs_when_T_not_1():
    cal = TemperatureCalibrator()
    cal.temperature = 2.0
    p = np.array([0.5, 0.7, 0.9])
    p_cal = cal.calibrate_proba(p)
    assert not np.allclose(p, p_cal)
    assert p_cal[2] < p[2]
    assert p_cal[1] < p[1]


def test_calibrated_equals_raw_when_T_1():
    cal = TemperatureCalibrator()
    cal.temperature = 1.0
    p = np.array([0.5, 0.7, 0.9])
    p_cal = cal.calibrate_proba(p)
    assert np.allclose(p, p_cal)
