"""Reproduction output set, P@3 integration and final-results export (items 15-17)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from agentic_forecaster.config import load_config
from agentic_forecaster.evaluation.metrics import precision_at_3_cross_sectional
from agentic_forecaster.orchestration.walk_forward import (
    AGGREGATE_KEYS,
    PAPER_FOLDS,
    _fold_config,
    run_walk_forward,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_paper_folds_are_exact():
    assert [f["fold"] for f in PAPER_FOLDS] == ["fold_0", "fold_1"]
    f0, f1 = PAPER_FOLDS
    assert (f0["train_start"], f0["train_end"]) == ("2016-01-01", "2020-12-31")
    assert (f0["val_start"], f0["val_end"]) == ("2021-01-01", "2021-12-31")
    assert (f0["test_start"], f0["test_end"]) == ("2022-01-01", "2022-12-31")
    assert (f1["train_end"],) == ("2021-12-31",)
    assert (f1["val_start"], f1["val_end"]) == ("2022-01-01", "2022-12-31")
    assert (f1["test_start"], f1["test_end"]) == ("2023-01-01", "2023-12-31")


def test_fold_config_overrides_split():
    base = {"data": {"sequence_length": 30}}
    out = _fold_config(base, PAPER_FOLDS[0])
    assert out["data"]["train_end"] == "2020-12-31"
    assert out["data"]["test_end"] == "2022-12-31"
    # original must be untouched
    assert base["data"] == {"sequence_length": 30}


def test_p3_is_computed_by_date_not_averaged_per_ticker():
    frame = pd.DataFrame({
        "date": ["2022-01-03"] * 4 + ["2022-01-04"] * 4,
        "ticker": list("ABCD") * 2,
        "y": [1, 1, 0, 0, 0, 0, 1, 1],
        "p_up": [0.9, 0.8, 0.2, 0.1, 0.15, 0.05, 0.95, 0.85],
    })
    out = precision_at_3_cross_sectional(frame, k=3)
    assert out["n_dates"] == 2
    assert len(out["selections"]) == 2
    assert out["selections"][0]["up_tickers"][0] == "A"
    assert out["selections"][0]["down_tickers"][0] == "D"


def test_p3_selections_are_auditable():
    frame = pd.DataFrame({
        "date": ["2022-01-03"] * 5,
        "ticker": list("ABCDE"),
        "y": [1, 1, 0, 0, 0],
        "p_up": [0.9, 0.8, 0.7, 0.6, 0.5],
    })
    out = precision_at_3_cross_sectional(frame, k=3)
    sel = out["selections"][0]
    assert set(sel) == {"date", "up_tickers", "up_precision",
                        "down_tickers", "down_precision"}
    assert len(sel["up_tickers"]) == 3


@pytest.mark.slow
def test_reproduction_writes_all_artefacts(tmp_path, monkeypatch):
    """A 1-ticker, 2-fold real-data run must produce every required file."""
    from agentic_forecaster.config import get_env_roots

    config = load_config(REPO_ROOT / "configs" / "paper.yaml")
    monkeypatch.setenv("AGENTIC_OUTPUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("AGENTIC_MODEL_ROOT", str(tmp_path / "models"))
    roots = get_env_roots()

    result = run_walk_forward(config, device="cpu", run_id="testrun", tickers=["RELIANCE"])
    run_dir = Path(result["run_dir"])

    required = [
        "manifest.json", "ticker_metrics.csv", "aggregate_metrics.json",
        "predictions.csv.gz", "calibration_metrics.csv", "precision_at_3.csv",
        "p3_daily_selections.csv", "baseline_metrics.csv", "ablation_metrics.csv",
        "paper_comparison.csv", "training_summary.json",
    ]
    for name in required:
        assert (run_dir / name).is_file(), f"missing reproduction output: {name}"
    assert (run_dir / "figures").is_dir()
    assert (run_dir / "reports").is_dir()

    agg = json.loads((run_dir / "aggregate_metrics.json").read_text())
    assert agg["n_ticker_fold_runs"] == 2      # one ticker x two folds
    for key in AGGREGATE_KEYS:
        if key in agg:
            assert isinstance(agg[key], float)

    baselines = pd.read_csv(run_dir / "baseline_metrics.csv")
    assert not baselines.empty
    assert set(baselines["model"]) >= {"lstm", "random_forest",
                                      "logistic_regression", "majority"}

    # export-final-results must flip reconstructed_run.json to completed
    from agentic_forecaster.packaging import export_final_artifacts

    export_final_artifacts(roots, run_dir=run_dir)
    repo = REPO_ROOT
    recon = json.loads((repo / "results/paper_reproduction/reconstructed_run.json").read_text())
    assert recon["status"] == "completed"
    assert (repo / "results/baselines/baseline_metrics.csv").is_file()
    assert (repo / "results/ablations/ablation_metrics.csv").is_file()
    assert (repo / "results/paper_reproduction/precision_at_3.csv").is_file()
