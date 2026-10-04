"""Append-only experiment ledger for the V2 programme.

``results/v2/experiment_ledger.csv`` is append-only.  Poor V2 experiments are
never deleted or edited: an architecture-contribution analysis is only worth
reading if its failures are visible next to its successes.

Two columns are first-class:

``test_2022_2023_evaluated``
    must be ``false`` for every run in this programme.  The append helper refuses
    a row that claims otherwise while the V2 firewalls are active, so a leaked
    run cannot be recorded as if it were legitimate.
``variant``
    the V2-A .. V2-F identifier, because the whole point of the ledger is the
    comparison between architecture variants.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

LEDGER_RELATIVE = Path("results/v2/experiment_ledger.csv")

#: PRE-COVID tracks keep their own ledger so the historical V2 record is never
#: appended to or reinterpreted.
PRECOVID_LEDGER_RELATIVE = Path("results/v2/pre_covid/experiment_ledger.csv")

LEDGER_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "timestamp",
    "git_commit",
    "experiment_dir",
    "experiment_regime",
    "variant",
    "fold",
    "seed",
    "tickers",
    "data_variant",
    "universe_id",
    "config_sha256",
    "feature_store_sha256",
    "sequence_length",
    "n_parameters",
    "max_epochs",
    "best_epoch",
    "validation_accuracy_micro",
    "validation_accuracy_macro",
    "validation_f1",
    "validation_auc",
    "validation_brier",
    "validation_ece",
    "train_majority_baseline",
    "precision_at_3_up",
    "precision_at_3_down",
    "return_mae",
    "return_spearman",
    "rank_ic",
    "test_2022_2023_evaluated",
)

#: Variants, in the order the staged programme runs them.
V2_VARIANTS: tuple[str, ...] = (
    "V2-A", "V2-B", "V2-C", "V2-D", "V2-E", "V2-F",
)

V2_VARIANT_LABELS: dict[str, str] = {
    "V2-A": "SHARED_LSTM",
    "V2-B": "LSTM_TRANSFORMER",
    "V2-C": "CONTEXTUAL_LSTM_TRANSFORMER",
    "V2-D": "MULTITASK",
    "V2-E": "FILM_CONDITIONAL_ADAPTATION",
    "V2-F": "REPTILE_STYLE_HEAD_ADAPTER",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ledger_path(root: Path | None = None, *, results_root: Path | str | None = None
                ) -> Path:
    """Resolve the ledger for a track.

    ``results_root`` selects a track-specific ledger (the PRE-COVID track uses
    ``results/v2/pre_covid/experiment_ledger.csv``); without it the historical
    V2 ledger is used.
    """
    if results_root is not None:
        return Path(results_root) / "experiment_ledger.csv"
    return (root or repo_root()) / LEDGER_RELATIVE


def precovid_ledger_path() -> Path:
    return repo_root() / PRECOVID_LEDGER_RELATIVE


def runtime_v2_root(sub: str = "") -> Path:
    """``$AGENTIC_OUTPUT_ROOT/v2`` -- runtime experiments, outside Git."""
    from agentic_forecaster.config import get_env_roots
    root = Path(get_env_roots()["AGENTIC_OUTPUT_ROOT"]) / "v2"
    return root / sub if sub else root


def experiment_dir(experiment_id: str, *, runtime_root: Path | str | None = None) -> Path:
    base = Path(runtime_root) if runtime_root is not None else runtime_v2_root()
    path = base / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def git_commit(root: Path | None = None) -> str:
    root = root or repo_root()
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                             capture_output=True, text=True, check=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def hash_payload(payload) -> str:
    """Stable SHA-256 over any JSON-serialisable object."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def new_experiment_id(prefix: str = "V2") -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


def read_ledger(path: Path | None = None, *, results_root: Path | str | None = None
                ) -> list[dict]:
    path = path or ledger_path(results_root=results_root)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def append_experiment(record: dict, path: Path | None = None, *,
                      root: Path | None = None,
                      results_root: Path | str | None = None) -> dict:
    """Append one immutable V2 experiment row and return the completed record.

    ``results_root`` selects a track-specific ledger, so the PRE-COVID track never
    appends to the historical V2 ledger (and vice versa).
    """
    path = path or ledger_path(root, results_root=results_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {c: record.get(c, "") for c in LEDGER_COLUMNS}
    row["experiment_id"] = row["experiment_id"] or new_experiment_id()
    row["timestamp"] = row["timestamp"] or datetime.now(UTC).isoformat()
    row["git_commit"] = row["git_commit"] or git_commit(root)
    row["variant"] = str(row["variant"]).upper()

    if str(row["test_2022_2023_evaluated"]).strip().lower() in ("true", "1", "yes"):
        raise ValueError(
            "Refusing to append a V2 row claiming test_2022_2023_evaluated=true. "
            "2022 and 2023 are never trained on or scored by any V2 script."
        )
    row["test_2022_2023_evaluated"] = "false"

    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(LEDGER_COLUMNS))
        if new_file:
            writer.writeheader()
        writer.writerow({c: row[c] for c in LEDGER_COLUMNS})
    return row


def ledger_summary(path: Path | None = None) -> dict:
    rows = read_ledger(path)
    return {
        "ledger_path": str(path or ledger_path()),
        "total_experiments": len(rows),
        "variants": sorted({r.get("variant", "") for r in rows if r.get("variant")}),
        "folds": sorted({r.get("fold", "") for r in rows if r.get("fold")}),
        "seeds": sorted({r.get("seed", "") for r in rows if r.get("seed")}),
        "rows_claiming_2022_2023": sum(
            1 for r in rows
            if str(r.get("test_2022_2023_evaluated", "")).lower() in ("true", "1", "yes")),
        "max_best_epoch": max((int(float(r["best_epoch"]))
                               for r in rows if str(r.get("best_epoch", "")).strip()),
                              default=None),
    }