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


def _make_run(tmp_path: Path, tickers: list[str] | None = None,
              folds: tuple[str, ...] = ("fold_0", "fold_1")) -> Path:
    """Build a run directory whose ticker/fold coverage the exporter can read."""
    tickers = tickers if tickers is not None else ["RELIANCE"]
    run = tmp_path / "out" / "reproduction" / "run1"
    run.mkdir(parents=True)

    rows = "ticker,fold,n_test,accuracy\n"
    status = {}
    for t in tickers:
        for f in folds:
            rows += f"{t},{f},10,0.51\n"
            status[f"{t}/{f}"] = {"status": "trained"}
    (run / "ticker_metrics.csv").write_text(rows)
    (run / "ticker_status.json").write_text(json.dumps(status))
    (run / "aggregate_metrics.json").write_text(
        json.dumps({"n_ticker_fold_runs": len(tickers) * len(folds),
                    "accuracy": 0.51})
    )
    for name in REQUIRED_RUN_FILES:
        if name in ("ticker_metrics.csv", "aggregate_metrics.json"):
            continue
        (run / name).write_text("date,ticker\n2022-01-03,RELIANCE\n")
    (run / "figures").mkdir()
    (run / "figures" / "reliability.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (run / "reports").mkdir()
    (run / "reports" / "RELIANCE_2022-12-29.html").write_text("<html></html>")
    (run / "reports" / "RELIANCE_2022-12-29.pdf").write_bytes(b"%PDF-1.4")
    return run


def _availability_csv(repo: Path, available: list[str], requested: list[str]) -> None:
    """Write a ticker_availability.csv so coverage can be derived from it."""
    path = repo / "results" / "ticker_availability.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("requested_ticker,discovered_symbol,available\n")
        f.writelines(f"{r},{r},{'True' if r in available else 'False'}\n" for r in requested)


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


def test_status_partial_when_universe_not_covered(tmp_path):
    """One ticker must NOT be reported as a completed paper reproduction."""
    run = _make_run(tmp_path, tickers=["RELIANCE"])
    fake_repo = tmp_path / "repo"
    _availability_csv(fake_repo, available=["RELIANCE", "TCS"],
                      requested=["RELIANCE", "TCS", "ZOMATO"])
    export_final_artifacts(
        {"AGENTIC_OUTPUT_ROOT": str(tmp_path / "out"),
         "AGENTIC_MODEL_ROOT": str(tmp_path / "models")},
        run_dir=run, repo=fake_repo,
    )
    recon = json.loads(
        (fake_repo / "results/paper_reproduction/reconstructed_run.json").read_text()
    )
    assert recon["status"] == "partial"
    cov = recon["coverage"]
    assert cov["expected_available_tickers"] == 2
    assert cov["expected_folds"] == 2
    assert cov["expected_runs"] == 4
    assert cov["completed_runs"] == 2
    assert cov["missing_ticker_folds"] == ["TCS/fold_0", "TCS/fold_1"]


def test_status_completed_only_on_full_coverage(tmp_path):
    run = _make_run(tmp_path, tickers=["RELIANCE", "TCS"])
    fake_repo = tmp_path / "repo"
    # ZOMATO is requested but unavailable, so the expectation is 2 x 2 == 4.
    _availability_csv(fake_repo, available=["RELIANCE", "TCS"],
                      requested=["RELIANCE", "TCS", "ZOMATO"])
    export_final_artifacts(
        {"AGENTIC_OUTPUT_ROOT": str(tmp_path / "out"),
         "AGENTIC_MODEL_ROOT": str(tmp_path / "models")},
        run_dir=run, repo=fake_repo,
    )
    recon = json.loads(
        (fake_repo / "results/paper_reproduction/reconstructed_run.json").read_text()
    )
    # ZOMATO is unavailable, so 2 available x 2 folds == 4 expected == 4 done.
    assert recon["status"] == "completed"
    assert recon["coverage"]["expected_available_tickers"] == 2
    assert recon["coverage"]["expected_runs"] == 4
    assert recon["coverage"]["completed_runs"] == 4
    assert recon["coverage"]["missing_ticker_folds"] == []


def test_status_pending_before_any_run(tmp_path):
    fake_repo = tmp_path / "repo"
    _availability_csv(fake_repo, available=["RELIANCE", "TCS"],
                      requested=["RELIANCE", "TCS"])
    run = _make_run(tmp_path, tickers=[])
    export_final_artifacts(
        {"AGENTIC_OUTPUT_ROOT": str(tmp_path / "out"),
         "AGENTIC_MODEL_ROOT": str(tmp_path / "models")},
        run_dir=run, repo=fake_repo,
    )
    recon = json.loads(
        (fake_repo / "results/paper_reproduction/reconstructed_run.json").read_text()
    )
    assert recon["status"] == "not_yet_run_on_full_universe"


def test_pdf_reports_are_exported_alongside_html(tmp_path):
    run = _make_run(tmp_path, tickers=["RELIANCE"])
    fake_repo = tmp_path / "repo"
    _availability_csv(fake_repo, available=["RELIANCE"], requested=["RELIANCE"])
    export_final_artifacts(
        {"AGENTIC_OUTPUT_ROOT": str(tmp_path / "out"),
         "AGENTIC_MODEL_ROOT": str(tmp_path / "models")},
        run_dir=run, repo=fake_repo,
    )
    examples = fake_repo / "reports" / "examples"
    assert (examples / "RELIANCE_2022-12-29.html").is_file()
    assert (examples / "RELIANCE_2022-12-29.pdf").is_file()


def test_blocked_paths_are_rejected():
    root = Path("/export/root")
    for name in BLOCKED_DIR_NAMES:
        assert not _safe(root / name / "file.csv", root=root), name
    assert not _safe(Path("kaggle.json"))
    assert not _safe(Path("secret.key"))
    assert not _safe(Path("run.log"))
    assert _safe(Path("results/metrics.csv"))
