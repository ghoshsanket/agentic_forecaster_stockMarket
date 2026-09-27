import numpy as np
import pandas as pd
import pytest

from agentic_forecaster.data.splits import walk_forward_folds


@pytest.fixture
def sample_walk_forward_data():
    idx = pd.date_range("2015-01-01", periods=3000, freq="D")
    return pd.DataFrame(
        {
            "date": idx,
            "close": np.random.uniform(90, 110, 3000),
            "ticker": ["TEST"] * 3000,
        }
    )


class TestWalkForwardIntegration:
    def test_folds_generated(self, sample_walk_forward_data):
        folds = walk_forward_folds(sample_walk_forward_data)
        assert isinstance(folds, list)
        assert len(folds) > 0

    def test_each_fold_train_before_test(self, sample_walk_forward_data):
        folds = walk_forward_folds(sample_walk_forward_data)
        for fold in folds:
            assert fold["train_start"] < fold["test_start"]

    def test_train_size_increases(self, sample_walk_forward_data):
        folds = walk_forward_folds(sample_walk_forward_data)
        train_sizes = [fold["train_end"] - fold["train_start"] for fold in folds]
        from itertools import pairwise
        assert all(a <= b for a, b in pairwise(train_sizes))

    def test_test_sets_do_not_overlap(self, sample_walk_forward_data):
        folds = walk_forward_folds(sample_walk_forward_data)
        for i in range(len(folds) - 1):
            assert folds[i]["test_end"] <= folds[i + 1]["test_start"]

    def test_no_temporal_leakage(self, sample_walk_forward_data):
        folds = walk_forward_folds(sample_walk_forward_data)
        for fold in folds:
            assert fold["train_end"] <= fold["val_start"]
            assert fold["val_end"] <= fold["test_start"]

    def test_combined_test_coverage(self, sample_walk_forward_data):
        folds = walk_forward_folds(sample_walk_forward_data)
        test_starts = [fold["test_start"] for fold in folds]
        test_ends = [fold["test_end"] for fold in folds]
        assert test_starts == sorted(test_starts)
        assert test_ends == sorted(test_ends)


class TestFoldAvailability:
    def test_check_fold_available(self, sample_walk_forward_data):
        folds = walk_forward_folds(sample_walk_forward_data)
        for fold in folds:
            assert "available" in fold

    def test_fold_count_one(self):
        idx = pd.date_range("2016-01-01", periods=500, freq="D")
        data = pd.DataFrame({"date": idx, "close": range(500)})
        folds = walk_forward_folds(data)
        assert len(folds) >= 1

    def test_minimum_data_requirement(self):
        idx = pd.date_range("2016-01-01", periods=100, freq="D")
        data = pd.DataFrame({"date": idx, "close": range(100)})
        folds = walk_forward_folds(data)
        for fold in folds:
            assert "available" in fold

    def test_request_more_folds_than_possible(self):
        idx = pd.date_range("2016-01-01", periods=200, freq="D")
        data = pd.DataFrame({"date": idx, "close": range(200)})
        folds = walk_forward_folds(data)
        assert isinstance(folds, list)
