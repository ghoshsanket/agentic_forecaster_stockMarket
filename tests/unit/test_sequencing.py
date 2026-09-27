import numpy as np
import pandas as pd
import pytest

from agentic_forecaster.data.sequencing import build_sequences


@pytest.fixture
def sample_ohlcv():
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "date": idx,
            "close": np.random.uniform(90, 110, 100),
            "volume": np.random.randint(1000, 5000, 100),
            "target": np.random.randint(0, 2, 100),
        }
    )


class TestBuildSequences:
    def test_feature_shape(self, sample_ohlcv):
        seq_length = 10
        X, _, _ = build_sequences(
            sample_ohlcv, seq_length=seq_length, target_col="target"
        )
        assert X.ndim == 3
        assert X.shape[1] == seq_length

    def test_number_of_sequences(self, sample_ohlcv):
        seq_length = 10
        X, _, _ = build_sequences(
            sample_ohlcv, seq_length=seq_length, target_col="target"
        )
        expected = len(sample_ohlcv) - seq_length
        assert len(X) == expected

    def test_target_values_match(self, sample_ohlcv):
        seq_length = 5
        _, y, _ = build_sequences(
            sample_ohlcv, seq_length=seq_length, target_col="target"
        )
        expected_targets = sample_ohlcv["target"].values[seq_length:]
        np.testing.assert_array_equal(y, expected_targets)

    def test_single_step_sequence(self, sample_ohlcv):
        X, _, _ = build_sequences(
            sample_ohlcv, seq_length=1, target_col="target"
        )
        assert X.shape[1] == 1
        assert len(X) == len(sample_ohlcv) - 1

    def test_dates_align_with_targets(self, sample_ohlcv):
        seq_length = 5
        _, _, dates = build_sequences(
            sample_ohlcv, seq_length=seq_length, target_col="target"
        )
        expected_dates = sample_ohlcv["date"].values[seq_length:]
        np.testing.assert_array_equal(dates, expected_dates)
