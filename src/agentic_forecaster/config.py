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


def _research_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return Path(os.environ.get("RESEARCH_ROOT", str(project_root.parent.parent)))


def _env_defaults() -> dict:
    """Canonical defaults for the AGENTIC_* runtime roots (no recursion).

    Every default is derived underneath ``RESEARCH_ROOT`` so that a config can
    be loaded from a bare shell, with no environment variable exported at all.
    An explicitly exported environment variable always wins.

    The Yahoo dataset roots are derived from ``AGENTIC_DATA_ROOT`` so that the
    modern universe (``yfinance_daily_2000_2005``/``..._2000_2025``) and the
    legacy universe (``yfinance_legacy_nifty50_2000_2025``) stay siblings
    under one parent and can never collide.
    """
    project_root = Path(__file__).resolve().parents[2]
    research_root = _research_root()
    data_root = research_root / "dataset"
    return {
        "AGENTIC_DATA_ROOT": str(data_root),
        "AGENTIC_RAW_DATA_ROOT": str(data_root / "agentic-forecaster" / "raw"),
        "AGENTIC_PROCESSED_DATA_ROOT": str(data_root / "agentic-forecaster" / "processed"),
        "AGENTIC_YFINANCE_DAILY_ROOT": str(data_root / "yfinance_daily_2000_2025"),
        "AGENTIC_YFINANCE_LEGACY_ROOT": str(data_root / "yfinance_legacy_nifty50_2000_2025"),
        "AGENTIC_MODEL_ROOT": str(research_root / "models" / "agentic-forecaster"),
        "AGENTIC_OUTPUT_ROOT": str(research_root / "output" / "agentic-forecaster"),
        "AGENTIC_PROJECT_ROOT": str(project_root),
        "AGENTIC_REPO_RESULTS_ROOT": str(project_root / "results"),
        "AGENTIC_REPO_REPORTS_ROOT": str(project_root / "reports"),
        "AGENTIC_REPO_FIGURES_ROOT": str(project_root / "figures"),
        "AGENTIC_REPO_ARTIFACTS_ROOT": str(project_root / "artifacts"),
    }


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        defaults = _env_defaults()

        def _sub(match: re.Match) -> str:
            var, default = match.group(1), match.group(2)
            if var in os.environ:
                return os.environ[var]
            if default is not None:
                return default
            if var in defaults:
                return defaults[var]
            raise KeyError(
                f"Environment variable {var!r} is not set, has no default in the "
                f"config file, and is not a known AGENTIC_* root."
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

    Every value defaults to a path underneath ``RESEARCH_ROOT``; an exported
    environment variable always overrides the default.
    """
    defaults = _env_defaults()
    return {
        name: os.environ.get(name, default)
        for name, default in defaults.items()
        if name != "AGENTIC_PROJECT_ROOT" and not name.startswith("AGENTIC_REPO_")
    } | {
        "AGENTIC_PROJECT_ROOT": defaults["AGENTIC_PROJECT_ROOT"],
        **{name: defaults[name] for name in defaults if name.startswith("AGENTIC_REPO_")},
    }
