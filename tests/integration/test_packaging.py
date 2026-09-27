"""Integration test: packaging exports only safe files."""

from __future__ import annotations

from pathlib import Path

from agentic_forecaster.packaging import export_final_artifacts


def test_exports_nothing_when_runtime_empty(tmp_path):
    roots = {
        "AGENTIC_OUTPUT_ROOT": str(tmp_path / "empty_output"),
        "AGENTIC_MODEL_ROOT": str(tmp_path / "empty_models"),
        "AGENTIC_REPO_RESULTS_ROOT": str(tmp_path / "repo" / "results"),
        "AGENTIC_REPO_REPORTS_ROOT": str(tmp_path / "repo" / "reports"),
        "AGENTIC_REPO_FIGURES_ROOT": str(tmp_path / "repo" / "figures"),
        "AGENTIC_REPO_ARTIFACTS_ROOT": str(tmp_path / "repo" / "artifacts"),
    }
    Path(roots["AGENTIC_OUTPUT_ROOT"]).mkdir(parents=True)
    copied = export_final_artifacts(roots)
    assert copied == []


def test_exports_metrics(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    (out / "metrics.json").write_text('{"accuracy": 0.5}')
    repo = tmp_path / "repo"
    roots = {
        "AGENTIC_OUTPUT_ROOT": str(out),
        "AGENTIC_MODEL_ROOT": str(tmp_path / "models"),
        "AGENTIC_REPO_RESULTS_ROOT": str(repo / "results"),
        "AGENTIC_REPO_REPORTS_ROOT": str(repo / "reports"),
        "AGENTIC_REPO_FIGURES_ROOT": str(repo / "figures"),
        "AGENTIC_REPO_ARTIFACTS_ROOT": str(repo / "artifacts"),
    }
    copied = export_final_artifacts(roots)
    assert any("metrics" in c for c in copied)
