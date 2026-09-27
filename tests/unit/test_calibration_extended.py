import numpy as np
import pytest

from agentic_forecaster.calibration import TemperatureCalibrator


@pytest.fixture
def sample_logits():
    return np.random.randn(100, 2).astype(np.float32)


@pytest.fixture
def sample_labels():
    return np.random.randint(0, 2, size=100)


class TestTemperatureCalibrator:
    def test_fit_returns_self(self, sample_logits, sample_labels):
        calibrator = TemperatureCalibrator()
        result = calibrator.fit(sample_logits, sample_labels)
        assert result is calibrator

    def test_produces_valid_probabilities(self, sample_logits, sample_labels):
        calibrator = TemperatureCalibrator()
        calibrator.fit(sample_logits, sample_labels)
        calibrated = calibrator.calibrate(sample_logits)
        assert calibrated.shape == sample_logits.shape
        assert not np.isnan(calibrated).any()

    def test_positive_temperature(self, sample_logits, sample_labels):
        calibrator = TemperatureCalibrator()
        calibrator.fit(sample_logits, sample_labels)
        assert calibrator.temperature > 0

    def test_save_and_load(self, sample_logits, sample_labels):
        calibrator = TemperatureCalibrator()
        calibrator.fit(sample_logits, sample_labels)
        d = calibrator.to_dict()
        loaded = TemperatureCalibrator.from_dict(d)
        np.testing.assert_allclose(calibrator.temperature, loaded.temperature)

    def test_before_fit_uses_identity(self, sample_logits):
        calibrator = TemperatureCalibrator()
        result = calibrator.calibrate(sample_logits)
        np.testing.assert_allclose(result, sample_logits)

    def test_extreme_logits(self, sample_labels):
        logits = np.array([[100.0, -100.0]] * len(sample_labels), dtype=np.float32)
        calibrator = TemperatureCalibrator()
        calibrator.fit(logits, sample_labels)
        assert calibrator.temperature > 0

    def test_uniform_logits(self, sample_labels):
        logits = np.zeros((len(sample_labels), 2), dtype=np.float32)
        calibrator = TemperatureCalibrator()
        calibrator.fit(logits, sample_labels)
        assert calibrator.temperature > 0
