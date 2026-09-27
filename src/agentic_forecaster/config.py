"""Configuration loading with environment-variable expansion.

Supports ``${VAR}`` and ``${VAR:-default}`` expansion so that runtime roots
(``AGENTIC_RAW_DATA_ROOT`` etc.) can be injected without editing YAML files.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def _sub(match: re.Match) -> str:
            var, default = match.group(1), match.group(2)
            if var in os.environ:
                return os.environ[var]
            if default is not None:
                return default
            raise KeyError(
                f"Environment variable {var!r} is not set and no default was "
                f"provided in the config file."
            )
        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and expand ``${ENV_VAR}`` references."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Config {path} must contain a YAML mapping.")
    return _expand(raw)


def get_env_roots() -> dict:
    """Return the canonical runtime root paths.

    Category B (outside Git) roots are read from the environment with
    workspace defaults. Category A (repository export) roots are derived
    from the repository location.
    """
    project_root = Path(__file__).resolve().parents[2]
    research_root = Path(
        os.environ.get("RESEARCH_ROOT", str(project_root.parent))
    )
    return {
        "AGENTIC_RAW_DATA_ROOT": os.environ.get(
            "AGENTIC_RAW_DATA_ROOT",
            str(research_root / "dataset" / "agentic-forecaster" / "raw"),
        ),
        "AGENTIC_PROCESSED_DATA_ROOT": os.environ.get(
            "AGENTIC_PROCESSED_DATA_ROOT",
            str(research_root / "dataset" / "agentic-forecaster" / "processed"),
        ),
        "AGENTIC_MODEL_ROOT": os.environ.get(
            "AGENTIC_MODEL_ROOT",
            str(research_root / "models" / "agentic-forecaster"),
        ),
        "AGENTIC_OUTPUT_ROOT": os.environ.get(
            "AGENTIC_OUTPUT_ROOT",
            str(research_root / "outputs" / "agentic-forecaster"),
        ),
        "AGENTIC_REPO_RESULTS_ROOT": str(project_root / "results"),
        "AGENTIC_REPO_REPORTS_ROOT": str(project_root / "reports"),
        "AGENTIC_REPO_FIGURES_ROOT": str(project_root / "figures"),
        "AGENTIC_REPO_ARTIFACTS_ROOT": str(project_root / "artifacts"),
    }
