"""Packaging must export the canonical final result set and nothing else."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_forecaster.packaging import (
    BLOCKED_DIR_NAMES,
    _safe,
    export_final_artifacts,
    find_latest_run,
)

REQUIRED_RUN_FILES = (
    "aggregate_metrics.json", "ticker_metrics.csv", "paper_comparison.csv",
    "calibration_metrics.csv", "precision_at_3.csv", "p3_daily_selections.csv",
    "baseline_metrics.csv", "ablation_metrics.csv",
)


def _make_run(tmp_path: Path, n_runs: int = 2) -> Path:
    run = tmp_path / "out" / "reproduction" / "run1"
    run.mkdir(parents=True)
    for name in REQUIRED_RUN_FILES:
        (run / name).write_text("date,ticker\n2022-01-03,RELIANCE\n")
    (run / "aggregate_metrics.json").write_text(
        json.dumps({"n_ticker_fold_runs": n_runs, "accuracy": 0.51})
    )
    (run / "figures").mkdir()
    (run / "figures" / "reliability.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (run / "reports").mkdir()
    (run / "reports" / "RELIANCE_2022-12-29.html").write_text("<html></html>")
    return run


def test_no_export_when_no_run_exists(tmp_path):
    roots = {
        "AGENTIC_OUTPUT_ROOT": str(tmp_path / "out"),
        "AGENTIC_MODEL_ROOT": str(tmp_path / "models"),
    }
    assert export_final_artifacts(roots) == []


def test_find_latest_run(tmp_path):
    _make_run(tmp_path)
    found = find_latest_run(tmp_path / "out")
    assert found is not None
    assert found.name == "run1"


def test_export_populates_expected_repo_paths(tmp_path):
    run = _make_run(tmp_path)
    fake_repo = tmp_path / "repo"
    roots = {"AGENTIC_OUTPUT_ROOT": str(tmp_path / "out"),
             "AGENTIC_MODEL_ROOT": str(tmp_path / "models")}
    copied = export_final_artifacts(roots, run_dir=run, repo=fake_repo)
    assert copied
    for rel in (
        "results/paper_reproduction/aggregate_metrics.json",
        "results/paper_reproduction/ticker_metrics.csv",
        "results/paper_reproduction/paper_comparison.csv",
        "results/baselines/baseline_metrics.csv",
        "results/ablations/ablation_metrics.csv",
        "figures/reliability.png",
        "reports/examples/RELIANCE_2022-12-29.html",
    ):
        assert (fake_repo / rel).is_file(), f"missing exported {rel}"


def test_export_sets_reconstructed_run_status(tmp_path):
    run = _make_run(tmp_path, n_runs=2)
    fake_repo = tmp_path / "repo"
    export_final_artifacts(
        {"AGENTIC_OUTPUT_ROOT": str(tmp_path / "out"),
         "AGENTIC_MODEL_ROOT": str(tmp_path / "models")},
        run_dir=run, repo=fake_repo,
    )
    recon = json.loads(
        (fake_repo / "results/paper_reproduction/reconstructed_run.json").read_text()
    )
    assert recon["status"] == "completed"
    assert recon["n_ticker_fold_runs"] == 2


def test_export_status_is_pending_without_real_runs(tmp_path):
    run = _make_run(tmp_path, n_runs=0)
    fake_repo = tmp_path / "repo"
    export_final_artifacts(
        {"AGENTIC_OUTPUT_ROOT": str(tmp_path / "out"),
         "AGENTIC_MODEL_ROOT": str(tmp_path / "models")},
        run_dir=run, repo=fake_repo,
    )
    recon = json.loads(
        (fake_repo / "results/paper_reproduction/reconstructed_run.json").read_text()
    )
    assert recon["status"] == "not_yet_run_on_real_dataset"


def test_blocked_paths_are_rejected():
    root = Path("/export/root")
    for name in BLOCKED_DIR_NAMES:
        assert not _safe(root / name / "file.csv", root=root), name
    assert not _safe(Path("kaggle.json"))
    assert not _safe(Path("secret.key"))
    assert not _safe(Path("run.log"))
    assert _safe(Path("results/metrics.csv"))
