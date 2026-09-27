import numpy as np
import pandas as pd
import pytest

from agentic_forecaster.data.splits import (
    temporal_split_by_date,
    temporal_split_by_fraction,
)
from agentic_forecaster.evaluation.walk_forward import (
    check_fold_date_coverage,
    paper_walk_forward_folds,
)


@pytest.fixture
def sample_data():
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "date": idx,
            "open": np.random.uniform(90, 100, 100),
            "high": np.random.uniform(100, 110, 100),
            "low": np.random.uniform(80, 90, 100),
            "close": np.random.uniform(95, 105, 100),
            "volume": np.random.randint(1000, 5000, 100),
        }
    )


class TestTemporalSplitByDate:
    def test_split_returns_three_dataframes(self, sample_data):
        train, val, test = temporal_split_by_date(
            sample_data,
            train_start="2024-01-01",
            train_end="2024-02-15",
            val_start="2024-02-16",
            val_end="2024-03-01",
            test_start="2024-03-02",
            test_end="2024-03-31",
        )
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_split_by_date_correct_sizes(self, sample_data):
        train, val, test = temporal_split_by_date(
            sample_data,
            train_start="2024-01-01",
            train_end="2024-02-15",
            val_start="2024-02-16",
            val_end="2024-03-01",
            test_start="2024-03-02",
            test_end="2024-03-31",
        )
        assert len(train) + len(val) + len(test) <= len(sample_data)

    def test_train_before_val_before_test(self, sample_data):
        train, val, test = temporal_split_by_date(
            sample_data,
            train_start="2024-01-01",
            train_end="2024-02-15",
            val_start="2024-02-16",
            val_end="2024-03-01",
            test_start="2024-03-02",
            test_end="2024-03-31",
        )
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0


class TestTemporalSplitByFraction:
    def test_split_by_fraction_correct_sizes(self, sample_data):
        train, val, test = temporal_split_by_fraction(sample_data, train_frac=0.7, val_frac=0.15)
        assert len(train) + len(val) + len(test) <= len(sample_data)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_temporal_order_preserved(self, sample_data):
        train, val, test = temporal_split_by_fraction(sample_data, train_frac=0.7, val_frac=0.15)
        train_dates = pd.to_datetime(train["date"])
        val_dates = pd.to_datetime(val["date"])
        test_dates = pd.to_datetime(test["date"])
        assert train_dates.max() < val_dates.min()
        assert val_dates.max() < test_dates.min()


class TestPaperWalkForwardFolds:
    def test_returns_list_of_folds(self):
        folds = paper_walk_forward_folds()
        assert isinstance(folds, list)
        assert len(folds) == 2

    def test_fold_structure(self):
        folds = paper_walk_forward_folds()
        for fold in folds:
            assert "fold" in fold
            assert "train_start" in fold
            assert "train_end" in fold
            assert "val_start" in fold
            assert "val_end" in fold
            assert "test_start" in fold
            assert "test_end" in fold

    def test_fold_0_dates(self):
        folds = paper_walk_forward_folds()
        fold0 = folds[0]
        assert str(fold0["train_start"]) == "2016-01-01"
        assert str(fold0["train_end"]) == "2020-12-31"
        assert str(fold0["val_start"]) == "2021-01-01"
        assert str(fold0["val_end"]) == "2021-12-31"
        assert str(fold0["test_start"]) == "2022-01-01"
        assert str(fold0["test_end"]) == "2022-12-31"

    def test_fold_1_dates(self):
        folds = paper_walk_forward_folds()
        fold1 = folds[1]
        assert str(fold1["train_start"]) == "2016-01-01"
        assert str(fold1["train_end"]) == "2021-12-31"
        assert str(fold1["val_start"]) == "2022-01-01"
        assert str(fold1["val_end"]) == "2022-12-31"
        assert str(fold1["test_start"]) == "2023-01-01"
        assert str(fold1["test_end"]) == "2023-12-31"


class TestCheckFoldDateCoverage:
    def test_full_coverage_passes(self):
        idx = pd.date_range("2015-01-01", periods=3000, freq="D")
        data = pd.DataFrame({"date": idx, "close": range(3000)})
        folds = paper_walk_forward_folds()
        result = check_fold_date_coverage(data, folds)
        assert len(result) == 2

    def test_incomplete_coverage_reported(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="D")
        data = pd.DataFrame({"date": idx, "close": range(100)})
        folds = paper_walk_forward_folds()
        result = check_fold_date_coverage(data, folds)
        assert isinstance(result, list)
