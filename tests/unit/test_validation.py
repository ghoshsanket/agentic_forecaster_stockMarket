import numpy as np
import pandas as pd

from agentic_forecaster.data.validation import (
    validate_no_leakage,
    validate_ohlcv,
    validate_target_alignment,
)


def make_valid_ohlcv(n=100):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = np.random.uniform(95, 105, n)
    high = close + np.random.uniform(1, 5, n)
    low = close - np.random.uniform(1, 5, n)
    open_ = low + np.random.uniform(0, (high - low).min() * 0.5, n)
    return pd.DataFrame(
        {
            "ticker": ["TEST"] * n,
            "date": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 5000, n),
        }
    )


class TestValidateOHLCV:
    def test_valid_data_passes(self):
        df = make_valid_ohlcv()
        errors = validate_ohlcv(df)
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"open": [1, 2], "close": [1, 2]})
        errors = validate_ohlcv(df)
        assert len(errors) > 0
        assert any("Missing" in e for e in errors)

    def test_non_positive_prices_raise(self):
        df = make_valid_ohlcv()
        df.loc[0, "close"] = -1
        errors = validate_ohlcv(df)
        assert any("Non-positive" in e for e in errors)

    def test_zero_prices_raise(self):
        df = make_valid_ohlcv()
        df.loc[0, "close"] = 0
        errors = validate_ohlcv(df)
        assert any("Non-positive" in e for e in errors)

    def test_high_less_than_low_raises(self):
        df = make_valid_ohlcv()
        df.loc[0, "high"] = df.loc[0, "low"] - 1
        errors = validate_ohlcv(df)
        assert any("high" in e.lower() for e in errors)

    def test_nan_values_raise(self):
        df = make_valid_ohlcv()
        df.loc[0, "close"] = np.nan
        errors = validate_ohlcv(df)
        assert any("NaN" in e for e in errors)

    def test_unsorted_dates_raise(self):
        df = make_valid_ohlcv()
        df = df.sort_values("date", ascending=False)
        errors = validate_ohlcv(df)
        assert any("sorted" in e.lower() or "chronological" in e.lower() for e in errors)


class TestValidateNoLeakage:
    def test_no_overlap_passes(self):
        train = pd.date_range("2024-01-01", periods=50, freq="D")
        val = pd.date_range("2024-02-20", periods=20, freq="D")
        test = pd.date_range("2024-03-15", periods=10, freq="D")
        errors = validate_no_leakage(train, val, test)
        assert len(errors) == 0

    def test_overlap_raises(self):
        train = pd.date_range("2024-01-01", periods=50, freq="D")
        val = pd.date_range("2024-01-25", periods=20, freq="D")
        test = pd.date_range("2024-03-15", periods=10, freq="D")
        errors = validate_no_leakage(train, val, test)
        assert len(errors) > 0

    def test_train_after_test_raises(self):
        train = pd.date_range("2024-03-01", periods=50, freq="D")
        val = pd.date_range("2024-02-01", periods=20, freq="D")
        test = pd.date_range("2024-01-01", periods=10, freq="D")
        errors = validate_no_leakage(train, val, test)
        assert len(errors) > 0


class TestValidateTargetAlignment:
    def test_aligned_data_passes(self):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        targets = pd.date_range("2024-01-02", periods=50, freq="D")
        errors = validate_target_alignment(dates, targets, seq_len=30)
        assert len(errors) == 0

    def test_length_mismatch_raises(self):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        targets = pd.date_range("2024-01-02", periods=40, freq="D")
        errors = validate_target_alignment(dates, targets, seq_len=30)
        assert len(errors) > 0

    def test_not_enough_samples_raises(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        targets = pd.date_range("2024-01-02", periods=10, freq="D")
        errors = validate_target_alignment(dates, targets, seq_len=30)
        assert len(errors) > 0
