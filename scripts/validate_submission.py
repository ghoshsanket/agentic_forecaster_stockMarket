#!/usr/bin/env python3
"""Validate that the repository is ready for GitHub submission.

Checks (non-cryptographic):
  [x] source code present
  [x] tests present
  [x] README present
  [x] DOI present
  [x] paper traceability present
  [x] implementation assumptions present
  [x] dataset provenance present
  [x] Kaggle URL present
  [x] dataset itself not accidentally committed
  [x] final result files present
  [x] representative reports present
  [x] figures present
  [x] model manifest present
  [x] no secrets detected by basic filename/path checks
  [x] no environment folder committed
  [x] no raw cache committed
  [x] repository paths are portable
  [x] reproduction commands documented

Usage:
    python scripts/validate_submission.py [--repo-root ROOT]
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

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
    """Return files that git would track (respecting .gitignore).

    Falls back to an empty list if git is unavailable or the directory is
    not a repository.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            return []
        return [repo_root / line for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def validate(repo_root: Path) -> list[str]:
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

    # Dataset must not be committed
    raw_data = repo_root / "data" / "raw"
    _check("dataset not committed", not raw_data.exists() or not any(raw_data.rglob("*.csv")), failures)

    # Check git-tracked files for env folders, caches, secrets and portability
    tracked = _git_tracked_files(repo_root)
    if not tracked:
        # Fall back to scanning everything if git is unavailable
        tracked = [p for p in repo_root.rglob("*") if p.is_file()]

    # No environment folders committed
    env_found = [
        p for p in tracked
        if any(part in ENV_DIR_NAMES for part in p.relative_to(repo_root).parts)
    ]
    _check("no environment folder committed", not env_found, failures,
           f"found: {[str(p.relative_to(repo_root)) for p in env_found[:3]]}")

    # No cache dirs committed
    cache_found = [
        p for p in tracked
        if any(part in CACHE_DIR_NAMES for part in p.relative_to(repo_root).parts)
    ]
    _check("no raw cache committed", not cache_found, failures,
           f"found: {[str(p.relative_to(repo_root)) for p in cache_found[:3]]}")

    # Secret scan (basic, non-cryptographic)
    secret_hits: list[str] = []
    for p in tracked:
        if p.suffix in {".png", ".pdf", ".pt", ".joblib", ".bin"}:
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for pat in SECRET_PATTERNS:
            if pat.search(text):
                secret_hits.append(str(p.relative_to(repo_root)))
                break
    _check("no secrets detected (basic filename/content check)", not secret_hits, failures,
           f"hits: {secret_hits[:5]}")

    # Portability: no absolute server paths in tracked text files
    # (exclude this validator, which necessarily contains the search string)
    abs_path_hits: list[str] = []
    for p in tracked:
        if p.name == "validate_submission.py":
            continue
        if p.suffix not in {".py", ".md", ".yaml", ".yml", ".toml", ".json", ".sh"}:
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        if "/home/iemiedc2026" in text:
            abs_path_hits.append(str(p.relative_to(repo_root)))
    _check("repository paths are portable", not abs_path_hits, failures,
           f"hits: {abs_path_hits[:5]}")

    # Reproduction commands documented
    _check("reproduction commands documented",
           "reproduce_paper.py" in readme and "train_all.py" in readme, failures)

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    print(f"Validating submission at: {repo_root}\n")
    failures = validate(repo_root)
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
