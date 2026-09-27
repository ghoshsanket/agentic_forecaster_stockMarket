import numpy as np
import pandas as pd
import pytest

from agentic_forecaster.data.resampling import resample_intraday_to_daily, resample_ohlcv


@pytest.fixture
def intraday_data():
    dates = pd.date_range("2024-01-01 09:30", periods=100, freq="1min")
    return pd.DataFrame(
        {
            "date": dates,
            "open": np.random.uniform(90, 100, 100),
            "high": np.random.uniform(100, 110, 100),
            "low": np.random.uniform(80, 90, 100),
            "close": np.random.uniform(95, 105, 100),
            "volume": np.random.randint(100, 500, 100),
        }
    )


class TestResampleIntradayToDaily:
    def test_basic_resampling(self, intraday_data):
        result = resample_intraday_to_daily(intraday_data)
        assert isinstance(result, pd.DataFrame)
        assert "date" in result.columns
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns

    def test_single_day(self):
        dates = pd.date_range("2024-01-01 09:30", periods=60, freq="1min")
        df = pd.DataFrame(
            {
                "date": dates,
                "open": [100.0] * 60,
                "high": [110.0] * 60,
                "low": [90.0] * 60,
                "close": [105.0] * 60,
                "volume": [1000] * 60,
            }
        )
        result = resample_intraday_to_daily(df)
        assert len(result) == 1
        assert result.iloc[0]["open"] == 100.0
        assert result.iloc[0]["high"] == 110.0
        assert result.iloc[0]["low"] == 90.0
        assert result.iloc[0]["close"] == 105.0
        assert result.iloc[0]["volume"] == 60000

    def test_multiple_days(self):
        dates = pd.date_range("2024-01-01 09:30", periods=600, freq="1min")
        df = pd.DataFrame(
            {
                "date": dates,
                "open": np.random.uniform(90, 100, 600),
                "high": np.random.uniform(100, 110, 600),
                "low": np.random.uniform(80, 90, 600),
                "close": np.random.uniform(95, 105, 600),
                "volume": np.random.randint(100, 500, 600),
            }
        )
        result = resample_intraday_to_daily(df)
        assert len(result) >= 1

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        result = resample_intraday_to_daily(df)
        assert len(result) == 0

    def test_missing_date_col_raises(self):
        df = pd.DataFrame({"open": [1], "close": [1]})
        with pytest.raises(ValueError):
            resample_intraday_to_daily(df)


class TestResampleOHLCV:
    def test_auto_detect_columns(self):
        dates = pd.date_range("2024-01-01 09:30", periods=60, freq="1min")
        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": [100.0] * 60,
                "High": [110.0] * 60,
                "Low": [90.0] * 60,
                "Close": [105.0] * 60,
                "Volume": [1000] * 60,
            }
        )
        result = resample_ohlcv(df)
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns
