"""Integration tests: Pipeline.run, demo config, baselines, ablations (items 2, 13, 14)."""

from __future__ import annotations

import pandas as pd
import pytest

from agentic_forecaster.config import load_config
from agentic_forecaster.orchestration import Pipeline

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def demo_result():
    config = load_config(REPO_ROOT / "configs" / "demo.yaml")
    return Pipeline(config, fold="fold_0").run("SYN00", device="cpu", n_reports=1)


def test_demo_config_loads():
    config = load_config(REPO_ROOT / "configs" / "demo.yaml")
    indicators = config["features"]["indicators"]
    assert "realized_volatility_20" in indicators
    assert "volatility_20" not in indicators
    for name in ("rsi_14", "macd", "macd_signal", "macd_histogram", "atr_14",
                 "log_return", "realized_volatility_20"):
        assert name in indicators


def test_pipeline_run_end_to_end(demo_result):
    r = demo_result
    assert r.ticker == "SYN00"
    assert r.dataset is not None
    assert r.primary is not None
    assert r.report and r.report["html"]


def test_pipeline_trains_all_five_model_families(demo_result):
    assert "attention_lstm" in demo_result.metrics
    for baseline in ("lstm", "random_forest", "logistic_regression", "majority"):
        assert baseline in demo_result.metrics, f"baseline {baseline} not executed"
        assert demo_result.metrics[baseline]


def test_pipeline_runs_ablations(demo_result):
    variants = demo_result.ablations
    for name in ("attention_lstm", "plain_lstm", "raw_vs_calibrated",
                 "ohlcv_only", "ohlcv_plus_technical",
                 "attention_vs_plain_lstm"):
        assert name in variants, f"ablation {name} missing"


def test_predictions_frame_shape(demo_result):
    df = demo_result.predictions
    assert isinstance(df, pd.DataFrame)
    for col in ("date", "ticker", "y", "raw_p_up", "calibrated_p_up",
                "direction", "confidence", "origin_date", "target_date"):
        assert col in df.columns
    assert (df["ticker"] == "SYN00").all()
    assert set(df["direction"]) <= {"UP", "DOWN"}


def test_baseline_metrics_have_real_numbers(demo_result):
    for name, metrics in demo_result.metrics.items():
        if name == "attention_lstm":
            continue
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0


def test_ablation_values_are_not_placeholders(demo_result):
    full = demo_result.ablations["attention_lstm"]
    assert "accuracy" in full and full["accuracy"] is not None
    ohclv = demo_result.ablations["ohlcv_only"]
    assert "error" not in ohclv, f"OHLCV-only ablation failed: {ohclv.get('error')}"
    assert "accuracy" in ohclv


def test_calibration_reported_both_ways(demo_result):
    m = demo_result.metrics["attention_lstm"]
    for key in ("brier_raw", "brier_calibrated", "ece_raw", "ece_calibrated"):
        assert key in m
