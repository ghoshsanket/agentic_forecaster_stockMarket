"""Export final artefacts from runtime (Category B) into the repository (Category A).

This module backs both ``python -m agentic_forecaster package-submission``
and ``python scripts/package_submission.py``.

Safety guarantees:
  * never copies the raw dataset;
  * never copies secrets, OpenCode state, caches or virtual environments;
  * never copies temporary files;
  * prints exactly what was copied.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger("agentic_forecaster.packaging")

# Directory names that must NEVER be exported when they appear as a direct
# child of an export source root.
BLOCKED_DIR_NAMES = {
    ".venv", "venv", "env", "cache", "caches", "wandb", "tmp", "temp",
    "opencode", "xdg", "secrets", ".git", "__pycache__", ".pytest_cache",
    ".ruff_cache", "node_modules", ".ipynb_checkpoints",
}

# File extensions that must NEVER be exported.
BLOCKED_EXTENSIONS = {
    ".pyc", ".pyo", ".log", ".tmp", ".bak", ".swp", ".env", ".key", ".pem",
}

# Maximum size for a single exported file (50 MB).
MAX_FILE_BYTES = 50 * 1024 * 1024


def _is_safe(path: Path, root: Path | None = None) -> bool:
    """Return True if ``path`` is safe to export.

    When ``root`` is given, blocked directory names are only enforced for the
    path components *relative to* ``root`` (so a temp directory elsewhere on
    the filesystem does not trigger a false positive).
    """
    if root is not None:
        try:
            rel = path.relative_to(root)
            blocked_parts = set(rel.parts) & BLOCKED_DIR_NAMES
        except ValueError:
            blocked_parts = set(path.parts) & BLOCKED_DIR_NAMES
        if blocked_parts:
            return False
    if path.name in {"kaggle.json", ".env", ".git"}:
        return False
    return path.suffix.lower() not in BLOCKED_EXTENSIONS


def _copy_safe(src: Path, dst: Path, root: Path | None = None) -> str | None:
    if not src.exists():
        return None
    if src.is_dir():
        if not _is_safe(src, root):
            return None
        dst.mkdir(parents=True, exist_ok=True)
        count = 0
        for child in src.rglob("*"):
            if child.is_file() and _is_safe(child, root):
                rel = child.relative_to(src)
                target = dst / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if child.stat().st_size <= MAX_FILE_BYTES:
                    shutil.copy2(child, target)
                    count += 1
        return str(dst) if count else None
    if src.is_file() and _is_safe(src, root) and src.stat().st_size <= MAX_FILE_BYTES:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return str(dst)
    return None


def export_final_artifacts(roots: dict) -> list[str]:
    """Export final artefacts from runtime roots into the repository.

    Parameters
    ----------
    roots : dict with keys
        AGENTIC_OUTPUT_ROOT, AGENTIC_MODEL_ROOT,
        AGENTIC_REPO_RESULTS_ROOT, AGENTIC_REPO_REPORTS_ROOT,
        AGENTIC_REPO_FIGURES_ROOT, AGENTIC_REPO_ARTIFACTS_ROOT.
    """
    copied: list[str] = []
    output_root = Path(roots["AGENTIC_OUTPUT_ROOT"])
    model_root = Path(roots["AGENTIC_MODEL_ROOT"])
    repo_results = Path(roots["AGENTIC_REPO_RESULTS_ROOT"])
    repo_reports = Path(roots["AGENTIC_REPO_REPORTS_ROOT"])
    repo_figures = Path(roots["AGENTIC_REPO_FIGURES_ROOT"])
    repo_artifacts = Path(roots["AGENTIC_REPO_ARTIFACTS_ROOT"])

    # 1. Runtime metrics -> results/ (search recursively for run subdirs)
    if output_root.exists():
        for metrics_file in sorted(output_root.rglob("metrics.json")):
            if not _is_safe(metrics_file, root=output_root):
                continue
            dst = repo_results / "metrics" / "run_metrics.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(metrics_file, dst)
            copied.append(str(dst))
        for expl_file in sorted(output_root.rglob("explanation.json")):
            if not _is_safe(expl_file, root=output_root):
                continue
            dst = repo_results / "metrics" / "run_explanation.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(expl_file, dst)
            copied.append(str(dst))

    # 2. Runtime reports -> reports/
    if output_root.exists():
        for html in sorted(output_root.rglob("*.html")):
            if not _is_safe(html, root=output_root):
                continue
            dst = repo_reports / "html" / html.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(html, dst)
            copied.append(str(dst))

    # 3. Runtime figures -> figures/
    if output_root.exists():
        for fig in sorted(output_root.rglob("*.png")) + sorted(output_root.rglob("*.svg")):
            if not _is_safe(fig, root=output_root):
                continue
            dst = repo_figures / fig.name
            shutil.copy2(fig, dst)
            copied.append(str(dst))

    # 4. Runtime checkpoints -> artifacts/manifests/models/ (metadata only)
    if model_root.exists():
        manifest_dir = repo_artifacts / "manifests" / "models"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        index = {"models": []}
        for ckpt in sorted(model_root.rglob("*.pt")):
            if not _is_safe(ckpt, root=model_root):
                continue
            from agentic_forecaster.utils import file_size_bytes, sha256_file

            index["models"].append(
                {
                    "ticker": ckpt.parent.name if ckpt.parent != model_root else ckpt.stem,
                    "path": str(ckpt.relative_to(model_root)),
                    "sha256": sha256_file(ckpt),
                    "bytes": file_size_bytes(ckpt),
                }
            )
        if index["models"]:
            dst = manifest_dir / "runtime_checkpoint_index.json"
            dst.write_text(json.dumps(index, indent=2))
            copied.append(str(dst))

    return copied
