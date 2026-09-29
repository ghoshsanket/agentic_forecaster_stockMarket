"""Freeze the chosen configuration, and refuse to unfreeze it.

The recovery process ends by freezing exactly ONE configuration.  After
``configs/recovered_paper.yaml`` and its manifest exist, no parameter may
change: the final 2022/2023 evaluation is only meaningful if it is the
evaluation of a configuration chosen without seeing those years.

The freeze is enforced by a manifest that records the SHA-256 of the frozen
config.  The final test run recomputes that hash and refuses to proceed on a
mismatch, so an edited config cannot be evaluated by accident.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from agentic_forecaster.recovery.ledger import git_commit, hash_file

FROZEN_CONFIG_RELATIVE = Path("configs/recovered_paper.yaml")
FROZEN_MANIFEST_RELATIVE = Path("results/reproduction_recovery/frozen_config_manifest.json")

#: The final evaluation may only run when this environment variable is exactly
#: "1".  It exists so scoring 2022/2023 always requires a deliberate act.
FINAL_TEST_ENV_VAR = "FINAL_TEST"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def frozen_config_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / FROZEN_CONFIG_RELATIVE


def frozen_manifest_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / FROZEN_MANIFEST_RELATIVE


class FrozenConfigError(RuntimeError):
    """Raised when the frozen configuration is missing, altered or unapproved."""


def freeze_config(config_path: str | Path, *, dataset_variant: str,
                  dataset_manifest: str | Path | None, universe_id: str,
                  validation_metrics: dict, supporting_experiment_ids: list[str],
                  notes: str = "", root: Path | None = None) -> dict:
    """Write the frozen-config manifest for an already-chosen config.

    ``config_path`` is the config that has been selected, normally
    ``configs/recovered_paper.yaml``.  The manifest pins its SHA-256.
    """
    root = root or repo_root()
    cfg = Path(config_path)
    if not cfg.is_file():
        raise FrozenConfigError(f"Config to freeze does not exist: {cfg}")
    ds = Path(dataset_manifest) if dataset_manifest else None
    if ds is not None and not ds.is_file():
        raise FrozenConfigError(f"Dataset manifest does not exist: {ds}")

    manifest = {
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(cfg.relative_to(root)) if cfg.is_relative_to(root) else str(cfg),
        "config_sha256": hash_file(cfg),
        "git_commit": git_commit(root),
        "dataset_variant": dataset_variant,
        "dataset_manifest": str(ds) if ds else None,
        "dataset_manifest_sha256": hash_file(ds) if ds else None,
        "universe_id": universe_id,
        "selection_basis": (
            "Chosen using pre-2022 validation folds SEARCH_FOLD_A/B/C only. "
            "The 2022/2023 paper test years were NOT observed during search."
        ),
        "validation_metrics": validation_metrics,
        "supporting_experiment_ids": list(supporting_experiment_ids),
        "final_command": (
            "FINAL_TEST=1 uv run python scripts/run_recovered_paper.py "
            "--config configs/recovered_paper.yaml --device auto "
            "--run-id recovered_paper_final"
        ),
        "policy": (
            "NO FURTHER PARAMETER CHANGES after this file exists. Editing the "
            "config invalidates the SHA-256 and the final run will refuse to start."
        ),
        "notes": notes,
    }
    out = frozen_manifest_path(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_frozen_manifest(root: Path | None = None) -> dict:
    p = frozen_manifest_path(root)
    if not p.is_file():
        raise FrozenConfigError(
            f"No frozen configuration found at {p}. Complete the staged search, "
            "create configs/recovered_paper.yaml, then run "
            "scripts/freeze_recovered_config.py before the final test."
        )
    return json.loads(p.read_text())


def verify_frozen_config(config_path: str | Path | None = None,
                         root: Path | None = None) -> dict:
    """Verify the config still matches its frozen SHA-256.

    Raises :class:`FrozenConfigError` on a mismatch.
    """
    manifest = load_frozen_manifest(root)
    cfg = Path(config_path) if config_path else (root or repo_root()) / FROZEN_CONFIG_RELATIVE
    if not cfg.is_file():
        raise FrozenConfigError(f"Frozen config is missing: {cfg}")
    actual = hash_file(cfg)
    expected = manifest["config_sha256"]
    if actual != expected:
        raise FrozenConfigError(
            f"Frozen config has CHANGED and may not be evaluated.\n"
            f"  expected sha256: {expected}\n"
            f"  actual   sha256: {actual}\n"
            f"  file: {cfg}\n"
            "Parameter changes after freezing invalidate the exercise. Re-run the "
            "search and freeze a new configuration if the change is intended."
        )
    return manifest


def final_test_authorised(env: dict | None = None) -> bool:
    """True only when ``FINAL_TEST=1`` is explicitly set."""
    env = env if env is not None else os.environ
    return str(env.get(FINAL_TEST_ENV_VAR, "")).strip() == "1"


def assert_final_test_allowed(config_path: str | Path, *,
                              env: dict | None = None,
                              root: Path | None = None) -> dict:
    """Gate for the final paper test.  Requires BOTH a valid freeze and FINAL_TEST=1."""
    if not final_test_authorised(env):
        raise FrozenConfigError(
            "Refusing to run the final paper test. The 2022/2023 test years may "
            f"only be scored with an explicit {FINAL_TEST_ENV_VAR}=1, e.g.\n"
            f"  {FINAL_TEST_ENV_VAR}=1 uv run python scripts/run_recovered_paper.py "
            "--config configs/recovered_paper.yaml --device auto "
            "--run-id recovered_paper_final\n"
            "Use scripts/run_reproduction_search.py for anything else."
        )
    return verify_frozen_config(config_path, root=root)


def describe_freeze_policy() -> dict:
    return {
        "frozen_config": str(FROZEN_CONFIG_RELATIVE),
        "frozen_manifest": str(FROZEN_MANIFEST_RELATIVE),
        "required_env_var": f"{FINAL_TEST_ENV_VAR}=1",
        "checks": [
            "frozen_config_manifest.json exists",
            "config SHA-256 matches the manifest",
            "dataset manifest hash recorded",
            f"{FINAL_TEST_ENV_VAR}=1 set explicitly",
        ],
        "after_freeze": "no parameter changes; the final run refuses on hash mismatch",
    }
