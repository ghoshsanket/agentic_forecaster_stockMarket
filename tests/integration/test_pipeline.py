"""Integration test: full pipeline on synthetic data."""

from __future__ import annotations

from agentic_forecaster.orchestration import Pipeline


def test_pipeline_end_to_end(tiny_config):
    result = Pipeline(tiny_config).run()
    assert result.dataset is not None
    assert len(result.fitted_models) == 5
    assert "attention_lstm" in result.fitted_models
    assert result.metrics["attention_lstm"]["accuracy"] >= 0.0
    assert len(result.reports) > 0
    assert result.run_dir


def test_dataset_split_sizes(tiny_config):
    result = Pipeline(tiny_config).run()
    n = len(result.dataset.train.y) + len(result.dataset.val.y) + len(result.dataset.test.y)
    assert n > 0
    assert len(result.dataset.train.y) > len(result.dataset.test.y)
