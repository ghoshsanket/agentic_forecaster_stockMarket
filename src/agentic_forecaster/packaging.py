"""Export final reproduction artefacts from runtime into the repository.

This backs both::

    python -m agentic_forecaster reproduce-paper --export-final-results
    python scripts/package_submission.py

It copies the CANONICAL final result set from the newest completed
reproduction run under ``$AGENTIC_OUTPUT_ROOT/reproduction/<run_id>/`` into
the repository.  It never copies raw data, secrets, caches, environments or
temporary files, and it prints exactly what was copied.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger("agentic_forecaster.packaging")

# Directory / file names that must NEVER be exported.
BLOCKED_DIR_NAMES = {
    ".venv", "venv", "env", "cache", "caches", "wandb", "tmp", "temp",
    "opencode", "xdg", "secrets", ".git", "__pycache__", ".pytest_cache",
    ".ruff_cache", "node_modules", ".ipynb_checkpoints", "raw", "dataset",
}
BLOCKED_EXTENSIONS = {".pyc", ".pyo", ".log", ".tmp", ".bak", ".swp", ".env", ".key", ".pem"}
BLOCKED_FILENAMES = {"kaggle.json", ".env", ".git"}
MAX_FILE_BYTES = 200 * 1024 * 1024

# run artefact -> repository destination
REPO_MAP = {
    "aggregate_metrics.json": "results/paper_reproduction/aggregate_metrics.json",
    "ticker_metrics.csv": "results/paper_reproduction/ticker_metrics.csv",
    "paper_comparison.csv": "results/paper_reproduction/paper_comparison.csv",
    "calibration_metrics.csv": "results/paper_reproduction/calibration_metrics.csv",
    "precision_at_3.csv": "results/paper_reproduction/precision_at_3.csv",
    "p3_daily_selections.csv": "results/paper_reproduction/p3_daily_selections.csv",
    "baseline_metrics.csv": "results/baselines/baseline_metrics.csv",
    "ablation_metrics.csv": "results/ablations/ablation_metrics.csv",
    "predictions.csv.gz": "results/predictions/predictions.csv.gz",
    "training_summary.json": "results/paper_reproduction/training_summary.json",
    "ticker_status.json": "results/paper_reproduction/ticker_status.json",
    "manifest.json": "results/paper_reproduction/run_manifest.json",
}


def _safe(path: Path, root: Path | None = None) -> bool:
    """Return True if ``path`` may be exported.

    Blocked directory names are enforced only for the components *below*
    ``root`` (or below the source directory), so an unrelated ancestor that
    happens to be named e.g. ``tmp`` does not cause a false positive.
    """
    if path.name in BLOCKED_FILENAMES:
        return False
    if path.suffix.lower() in BLOCKED_EXTENSIONS:
        return False
    if root is not None:
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        if any(part in BLOCKED_DIR_NAMES for part in rel.parts):
            return False
    return True


def _copy(src: Path, dst: Path, root: Path | None = None) -> str | None:
    if not src.is_file() or not _safe(src, root) or src.stat().st_size > MAX_FILE_BYTES:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def find_latest_run(output_root: Path) -> Path | None:
    """Newest reproduction run directory that contains aggregate_metrics.json."""
    repro = Path(output_root) / "reproduction"
    if not repro.is_dir():
        return None
    candidates = [
        d for d in repro.iterdir()
        if d.is_dir() and (d / "aggregate_metrics.json").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def repo_root() -> Path:
    """Repository root (the directory containing pyproject.toml / src)."""
    return Path(__file__).resolve().parents[2]


def export_final_artifacts(
    roots: dict,
    run_dir: Path | None = None,
    repo: Path | None = None,
) -> list[str]:
    """Copy the canonical final result set into the repository."""
    output_root = Path(roots["AGENTIC_OUTPUT_ROOT"])
    repo = Path(repo) if repo is not None else repo_root()

    run_dir = run_dir or find_latest_run(output_root)
    if run_dir is None:
        logger.warning("No completed reproduction run found under %s", output_root / "reproduction")
        return []

    copied: list[str] = []

    # 1. Flat run artefacts -> repository (Category A).
    for name, rel in REPO_MAP.items():
        got = _copy(run_dir / name, repo / rel, root=run_dir)
        if got:
            copied.append(got)

    # 2. Figures.
    for fig in sorted((run_dir / "figures").glob("*")) if (run_dir / "figures").is_dir() else []:
        if fig.suffix.lower() in (".png", ".svg", ".pdf"):
            got = _copy(fig, repo / "figures" / fig.name, root=run_dir / "figures")
            if got:
                copied.append(got)

    # 3. Representative reports (a bounded sample, not every daily report).
    reports = run_dir / "reports"
    if reports.is_dir():
        for rep in sorted(reports.glob("*.html"))[:10]:
            got = _copy(rep, repo / "reports" / "examples" / rep.name, root=reports)
            if got:
                copied.append(got)

    # 4. Model manifest (hashes only; no checkpoint binaries).
    model_root = Path(roots["AGENTIC_MODEL_ROOT"])
    if model_root.is_dir():
        from agentic_forecaster.utils import file_size_bytes, sha256_file

        index = {
            "provenance": "reconstructed_run",
            "run_id": run_dir.name,
            "runtime_model_root_env": "AGENTIC_MODEL_ROOT",
            "path_base": "relative to $AGENTIC_MODEL_ROOT",
            "models": [],
        }
        for ckpt in sorted(model_root.rglob("model.pt")):
            if not _safe(ckpt, root=model_root):
                continue
            index["models"].append({
                "ticker": ckpt.parent.parent.name,
                "fold": ckpt.parent.name,
                "path": str(ckpt.relative_to(model_root)),
                "sha256": sha256_file(ckpt),
                "bytes": file_size_bytes(ckpt),
            })
        if index["models"]:
            dst = repo / "artifacts" / "manifests" / "models" / "runtime_checkpoint_index.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(index, indent=2))
            copied.append(str(dst))

    # 5. reconstructed_run.json must only claim completion for a REAL run.
    aggregate = run_dir / "aggregate_metrics.json"
    if aggregate.is_file():
        agg = json.loads(aggregate.read_text())
        n_runs = int(agg.get("n_ticker_fold_runs", 0))
        status = "completed" if n_runs > 0 else "not_yet_run_on_real_dataset"
        payload = {
            "provenance": "reconstructed_run",
            "status": status,
            "description": (
                "Metrics produced by this implementation on the real dataset."
                if n_runs > 0 else
                "No real reproduction run has been recorded yet."
            ),
            "run_id": run_dir.name,
            "run_dir_env": "$AGENTIC_OUTPUT_ROOT/reproduction/" + run_dir.name,
            "n_ticker_fold_runs": n_runs,
            "aggregate_metrics": agg,
        }
        dst = repo / "results" / "paper_reproduction" / "reconstructed_run.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(payload, indent=2, default=str))
        copied.append(str(dst))

    logger.info("Exported %d artefact(s) from %s", len(copied), run_dir)
    return copied
