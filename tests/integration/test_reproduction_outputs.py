"""Reproduction outputs, cross-sectional P@3 and final-results export.

Two layers:

* ``test_reproduction_writes_all_artefacts`` — SYNTHETIC data. Runs in
  ordinary CI, needs no Kaggle download, no network and no CUDA, and writes
  every artefact into a pytest-owned temporary repository.
* ``test_real_reproduction_writes_all_artefacts`` — the real Kaggle dataset.
  Opt-in only: requires ``AGENTIC_RUN_REAL_DATA_TESTS=1``. Like the synthetic
  test it writes exclusively into ``tmp_path``; it must never touch the real
  repository's ``results/``, ``figures/``, ``reports/`` or ``artifacts/``.
"""

from __future__ import annotations

import json
import os
import shutil
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
from agentic_forecaster.packaging import export_final_artifacts

REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_RUN_FILES = (
    "manifest.json", "ticker_metrics.csv", "aggregate_metrics.json",
    "predictions.csv.gz", "calibration_metrics.csv", "precision_at_3.csv",
    "p3_daily_selections.csv", "baseline_metrics.csv", "ablation_metrics.csv",
    "paper_comparison.csv", "training_summary.json",
)


# --------------------------------------------------------------- fold config

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
    assert base["data"] == {"sequence_length": 30}


# ------------------------------------------------------------------- P@3

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


# ------------------------------------------------------- synthetic end-to-end

@pytest.fixture
def synthetic_config(tmp_path):
    """A small synthetic 3-ticker, multi-year config (no Kaggle data)."""
    return {
        "experiment": {"name": "test_synthetic", "seed": 42,
                       "output_dir": str(tmp_path / "run")},
        "data": {
            "synthetic": True, "n_tickers": 3, "n_days": 1200,
            "sequence_length": 10,
        },
        "features": {
            "use_ohlcv": True,
            "indicators": ["log_return", "realized_volatility_20", "rsi_14",
                           "macd", "macd_signal", "macd_histogram", "atr_14"],
            "drop_na": True,
        },
        "models": {
            "attention_lstm": {"hidden_size": 8, "num_layers": 1, "dropout": 0.0,
                               "learning_rate": 1e-3, "batch_size": 64,
                               "epochs": 1, "patience": 10,
                               "beta1": 0.9, "beta2": 0.999,
                               "gradient_clip_norm": 1.0},
            "lstm": {"hidden_size": 8, "num_layers": 1, "dropout": 0.0,
                     "learning_rate": 1e-3, "batch_size": 64,
                     "epochs": 1, "patience": 10,
                     "beta1": 0.9, "beta2": 0.999,
                     "gradient_clip_norm": 1.0},
            "random_forest": {"n_estimators": 10, "max_depth": 3},
            "logistic_regression": {"C": 1.0, "max_iter": 100},
        },
        "baselines": {"enabled": True, "lstm": True, "random_forest": True,
                      "logistic_regression": True, "majority": True},
        "ablations": {"enabled": True},
        "calibration": {"method": "temperature", "fit_on": "val"},
        "evaluation": {"ece_bins": 10, "p_at_k": 3},
        "explainability": {"background_samples": 8, "top_k_features": 3},
        "risk": {"enabled": True, "formulation": "atr_confidence_multiplier"},
        "reporting": {"formats": ["html", "pdf"],
                      "llm": {"enabled": False}},
    }


def test_reproduction_writes_all_artefacts(synthetic_config, tmp_path, monkeypatch):
    """Synthetic end-to-end run: every artefact, written only under tmp_path."""
    monkeypatch.setenv("AGENTIC_OUTPUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("AGENTIC_MODEL_ROOT", str(tmp_path / "models"))

    result = run_walk_forward(
        synthetic_config, device="cpu", run_id="synthetic_test",
        tickers=["SYN00", "SYN01", "SYN02"], n_reports=1,
    )
    run_dir = Path(result["run_dir"])
    assert run_dir.is_relative_to(tmp_path), "run must stay inside tmp_path"

    for name in REQUIRED_RUN_FILES:
        assert (run_dir / name).is_file(), f"missing reproduction output: {name}"
    assert (run_dir / "figures").is_dir()
    assert (run_dir / "reports").is_dir()
    assert list((run_dir / "figures").glob("*.png")), "no figures produced"
    assert list((run_dir / "reports").glob("*.html")), "no HTML reports produced"
    assert list((run_dir / "reports").glob("*.pdf")), "no PDF reports produced"

    agg = json.loads((run_dir / "aggregate_metrics.json").read_text())
    assert agg["n_ticker_fold_runs"] == 6          # 3 tickers x 2 folds
    for key in AGGREGATE_KEYS:
        if key in agg:
            assert isinstance(agg[key], (int, float))

    baselines = pd.read_csv(run_dir / "baseline_metrics.csv")
    assert set(baselines["model"]) >= {"lstm", "random_forest",
                                       "logistic_regression", "majority"}

    p3 = pd.read_csv(run_dir / "precision_at_3.csv")
    assert set(p3["fold"]) == {"fold_0", "fold_1"}

    # ---- export into a pytest-owned temporary repository ------------------
    fake_repo = tmp_path / "repo"
    roots = {"AGENTIC_OUTPUT_ROOT": str(tmp_path / "out"),
             "AGENTIC_MODEL_ROOT": str(tmp_path / "models")}
    copied = export_final_artifacts(roots, run_dir=run_dir, repo=fake_repo)
    assert copied
    for rel in ("results/paper_reproduction/aggregate_metrics.json",
                "results/paper_reproduction/ticker_metrics.csv",
                "results/paper_reproduction/precision_at_3.csv",
                "results/baselines/baseline_metrics.csv",
                "results/ablations/ablation_metrics.csv",
                "figures/reliability_diagram.png"):
        assert (fake_repo / rel).is_file(), f"missing exported {rel}"

    # HTML AND PDF representative reports must be exported.
    examples = fake_repo / "reports" / "examples"
    assert list(examples.glob("*.html")), "no HTML reports exported"
    assert list(examples.glob("*.pdf")), "no PDF reports exported"


def test_synthetic_run_writes_only_inside_tmp_path(synthetic_config, tmp_path,
                                                   monkeypatch):
    """A reproduction run must not write into the checked-out repository."""
    from agentic_forecaster import packaging

    real_repo = packaging.repo_root()
    assert (real_repo / "pyproject.toml").is_file()

    sentinel = {
        rel: (real_repo / rel)
        for rel in ("results/paper_reproduction/reconstructed_run.json",)
    }
    before = {rel: p.read_bytes() if p.is_file() else None
              for rel, p in sentinel.items()}

    monkeypatch.setenv("AGENTIC_OUTPUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("AGENTIC_MODEL_ROOT", str(tmp_path / "models"))

    result = run_walk_forward(synthetic_config, device="cpu",
                              run_id="isolation_test", tickers=["SYN00"],
                              n_reports=0)
    assert Path(result["run_dir"]).is_relative_to(tmp_path)

    # The run itself must not have touched the repository.
    for rel, path in sentinel.items():
        after = path.read_bytes() if path.is_file() else None
        assert after == before[rel], f"{rel} was modified by a test run"

    # And an export aimed at a temporary repo must leave the real one alone.
    fake_repo = tmp_path / "repo3"
    export_final_artifacts(
        {"AGENTIC_OUTPUT_ROOT": str(tmp_path / "out"),
         "AGENTIC_MODEL_ROOT": str(tmp_path / "models")},
        run_dir=Path(result["run_dir"]),
        repo=fake_repo,
    )
    assert fake_repo.is_dir() and fake_repo != real_repo
    for rel, path in sentinel.items():
        after = path.read_bytes() if path.is_file() else None
        assert after == before[rel], f"{rel} was modified by an export"


# ------------------------------------------------------------- real dataset

@pytest.mark.real_data
@pytest.mark.slow
def test_real_reproduction_writes_all_artefacts(tmp_path, monkeypatch):
    """OPT-IN: runs the real Kaggle dataset. Never writes to the real repo."""
    if os.environ.get("AGENTIC_RUN_REAL_DATA_TESTS") != "1":
        pytest.skip("real Kaggle dataset test is opt-in "
                    "(set AGENTIC_RUN_REAL_DATA_TESTS=1)")

    from agentic_forecaster.config import get_env_roots

    config = load_config(REPO_ROOT / "configs" / "paper.yaml")
    monkeypatch.setenv("AGENTIC_OUTPUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("AGENTIC_MODEL_ROOT", str(tmp_path / "models"))
    roots = get_env_roots()
    assert Path(roots["AGENTIC_RAW_DATA_ROOT"]).is_dir(), "raw dataset not present"

    result = run_walk_forward(config, device="cpu", run_id="real_test",
                              tickers=["RELIANCE"], n_reports=1)
    run_dir = Path(result["run_dir"])
    assert run_dir.is_relative_to(tmp_path)
    for name in REQUIRED_RUN_FILES:
        assert (run_dir / name).is_file(), f"missing: {name}"

    # Copy the REAL universe record so coverage is judged against the actual
    # available-ticker expectation: a 1-ticker run must be "partial", never
    # "completed".
    fake_repo = tmp_path / "repo"
    (fake_repo / "results").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "results/ticker_availability.csv",
                fake_repo / "results/ticker_availability.csv")
    export_final_artifacts(
        {"AGENTIC_OUTPUT_ROOT": str(tmp_path / "out"),
         "AGENTIC_MODEL_ROOT": str(tmp_path / "models")},
        run_dir=run_dir,
        repo=fake_repo,           # <- temporary repository, never the real one
    )
    recon = json.loads(
        (fake_repo / "results/paper_reproduction/reconstructed_run.json").read_text()
    )
    assert recon["n_ticker_fold_runs"] == 2
    # A single ticker is NOT the full NIFTY-50 reproduction.
    assert recon["status"] == "partial"
    assert recon["coverage"]["expected_runs"] == 98      # available x 2 folds
    assert len(recon["coverage"]["missing_ticker_folds"]) == 96

    # The real repository must be unchanged by this test.
    real = REPO_ROOT / "results/paper_reproduction/reconstructed_run.json"
    assert json.loads(real.read_text())["status"] == "not_yet_run_on_full_universe"
