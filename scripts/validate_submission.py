#!/usr/bin/env python3
"""Validate that the repository is ready for GitHub submission.

Strict mode (default) FAILS if:
    - real reproduction is required AND reconstructed_run is pending
    - aggregate metrics absent
    - canonical model count is zero
    - all paper-comparison reconstructed values are null

Use ``--allow-pretraining`` for structure-only validation.

Usage:
    python scripts/validate_submission.py [--repo-root ROOT] [--allow-pretraining]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"KAGGLE_KEY\s*=\s*(?!<your-key>)\S+"),
    re.compile(r"BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY"),
]

ENV_DIR_NAMES = {".venv", "venv", "env", "node_modules", "__pycache__", ".git"}
CACHE_DIR_NAMES = {"cache", "caches", "wandb", ".pytest_cache", ".ruff_cache", "tmp"}


def _check(name: str, ok: bool, failures: list[str], detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not ok:
        failures.append(name)


def _git_tracked_files(repo_root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=repo_root, capture_output=True, text=True, timeout=30, check=False,
        )
        if result.returncode != 0:
            return []
        return [repo_root / line for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def validate(repo_root: Path, allow_pretraining: bool = False) -> list[str]:
    failures: list[str] = []

    src = repo_root / "src" / "agentic_forecaster"
    _check("source code present", src.is_dir() and any(src.rglob("*.py")), failures)
    _check("tests present", (repo_root / "tests").is_dir() and any((repo_root / "tests").rglob("test_*.py")), failures)
    _check("README present", (repo_root / "README.md").is_file(), failures)

    readme = (repo_root / "README.md").read_text() if (repo_root / "README.md").exists() else ""
    _check("DOI present", "10.1109/IEMENTECH202669403.2026.11434302" in readme, failures)
    _check("Kaggle URL present", "kaggle.com/datasets/debashis74017" in readme, failures)
    _check("dataset-not-committed statement", "not stored in Git" in readme or "not committed" in readme, failures)

    _check("paper traceability present", (repo_root / "docs" / "PAPER_TRACEABILITY.md").is_file(), failures)
    _check("implementation assumptions present", (repo_root / "docs" / "IMPLEMENTATION_ASSUMPTIONS.md").is_file(), failures)
    _check("dataset provenance present", (repo_root / "docs" / "DATASET_PROVENANCE.md").is_file(), failures)

    results_dir = repo_root / "results"
    _check("final result files present", results_dir.is_dir() and any(results_dir.rglob("*.json")) or any(results_dir.rglob("*.csv")), failures)
    _check("representative reports present", (repo_root / "reports" / "examples").is_dir() and any((repo_root / "reports" / "examples").iterdir()), failures)
    _check("figures present", (repo_root / "figures").is_dir() and any((repo_root / "figures").iterdir()), failures)
    _check("model manifest present", (repo_root / "artifacts" / "manifests").is_dir() and any((repo_root / "artifacts" / "manifests").rglob("*.json")), failures)

    raw_data = repo_root / "data" / "raw"
    _check("dataset not committed", not raw_data.exists() or not any(raw_data.rglob("*.csv")), failures)

    tracked = _git_tracked_files(repo_root)
    if not tracked:
        tracked = [p for p in repo_root.rglob("*") if p.is_file()]

    env_found = [p for p in tracked if any(part in ENV_DIR_NAMES for part in p.relative_to(repo_root).parts)]
    _check("no environment folder committed", not env_found, failures,
           f"found: {[str(p.relative_to(repo_root)) for p in env_found[:3]]}")

    cache_found = [p for p in tracked if any(part in CACHE_DIR_NAMES for part in p.relative_to(repo_root).parts)]
    _check("no raw cache committed", not cache_found, failures,
           f"found: {[str(p.relative_to(repo_root)) for p in cache_found[:3]]}")

    secret_hits: list[str] = []
    for p in tracked:
        if p.suffix in {".png", ".pdf", ".pt", ".joblib", ".bin"}:
            continue
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            continue
        for pat in SECRET_PATTERNS:
            if pat.search(text):
                secret_hits.append(str(p.relative_to(repo_root)))
                break
    _check("no secrets detected (basic filename/content check)", not secret_hits, failures,
           f"hits: {secret_hits[:5]}")

    abs_path_hits: list[str] = []
    for p in tracked:
        if p.name == "validate_submission.py":
            continue
        if p.suffix not in {".py", ".md", ".yaml", ".yml", ".toml", ".json", ".sh"}:
            continue
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            continue
        if "/home/iemiedc2026" in text:
            abs_path_hits.append(str(p.relative_to(repo_root)))
    _check("repository paths are portable", not abs_path_hits, failures,
           f"hits: {abs_path_hits[:5]}")

    _check("reproduction commands documented",
           ("reproduce_paper.py" in readme or "reproduce-paper" in readme)
           and ("train_all.py" in readme or "train-all" in readme), failures)

    # --- Strict real-results validation ---
    if not allow_pretraining:
        recon_path = results_dir / "paper_reproduction" / "reconstructed_run.json"
        if recon_path.exists():
            recon = json.loads(recon_path.read_text())
            status = recon.get("status", "")
            _check("reconstructed_run not pending",
                   status == "completed",
                   failures, f"status={status}")

            # Real run recorded AND it actually produced ticker/fold models.
            n_runs = int(recon.get("n_ticker_fold_runs", 0) or 0)
            _check("real reproduction recorded", n_runs > 0, failures,
                   f"n_ticker_fold_runs={n_runs}")

            aggregate = recon.get("aggregate_metrics", {}) or {}
            has_metrics = any(
                isinstance(v, (int, float)) for v in aggregate.values()
            )
            _check("aggregate metrics present", has_metrics, failures)

        agg_path = results_dir / "paper_reproduction" / "aggregate_metrics.json"
        if agg_path.exists():
            agg = json.loads(agg_path.read_text())
            _check("aggregate metrics file non-empty",
                   int(agg.get("n_ticker_fold_runs", 0)) > 0
                   and "accuracy" in agg,
                   failures)

        manifest_path = repo_root / "artifacts" / "manifests" / "models" / "runtime_checkpoint_index.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            _check("canonical model count > 0",
                   len(manifest.get("models", [])) > 0,
                   failures, f"count={len(manifest.get('models', []))}")

        comparison_path = results_dir / "paper_reproduction" / "paper_comparison.csv"
        if comparison_path.exists():
            import csv
            with open(comparison_path) as f:
                rows = list(csv.DictReader(f))
            has_recon = any(r.get("accuracy_reconstructed") for r in rows)
            _check("paper-comparison reconstructed values present", has_recon, failures)

        for rel in ("results/baselines/baseline_metrics.csv",
                    "results/ablations/ablation_metrics.csv",
                    "results/paper_reproduction/precision_at_3.csv"):
            path = repo_root / rel
            populated = path.is_file() and path.stat().st_size > 0
            if rel.endswith("baseline_metrics.csv") and populated:
                with open(path) as f:
                    populated = len(f.readlines()) > 1
            if rel.endswith("ablation_metrics.csv") and populated:
                with open(path) as f:
                    populated = len(f.readlines()) > 1
            _check(f"final artefact present: {rel}", populated, failures)

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--allow-pretraining", action="store_true",
                        help="Skip strict real-results checks (structure-only validation)")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    mode = "PRE-TRAINING (structure-only)" if args.allow_pretraining else "FINAL (strict)"
    print(f"Validating submission at: {repo_root} [{mode}]\n")
    failures = validate(repo_root, allow_pretraining=args.allow_pretraining)
    print()
    if failures:
        print(f"VALIDATION FAILED — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("VALIDATION PASSED — repository is ready for GitHub submission.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
