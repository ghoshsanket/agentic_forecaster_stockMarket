"""Unit tests for walk-forward fold generation."""

from __future__ import annotations

from agentic_forecaster.evaluation.walk_forward import walk_forward_folds


def test_fold_count():
    folds = walk_forward_folds(n_samples=1000, seq_len=30, step=100, min_train=300)
    assert len(folds) == 7  # 300..1000 step 100 -> 7 folds


def test_anchored_growth():
    folds = walk_forward_folds(n_samples=1000, seq_len=30, step=100, min_train=300)
    assert folds[0]["train_end"] == 300
    assert folds[1]["train_end"] == 400
    assert folds[0]["train_start"] == 0  # anchored
