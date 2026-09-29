"""Append-only experiment ledger for performance recovery.

Every experiment appends exactly one immutable row to
``results/reproduction_recovery/experiment_ledger.csv``.

Poor experiments are NEVER deleted or edited.  A recovery exercise whose ledger
hides its failures cannot be audited, and the whole point of searching on
pre-2022 validation is that the search history is visible and honest.

``test_evaluated`` is a first-class column.  For every search experiment it must
be ``false``; the ledger helper refuses to append a row that claims otherwise
while the firewall is active.
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

from agentic_forecaster.recovery.firewall import is_search_mode

LEDGER_RELATIVE = Path("results/reproduction_recovery/experiment_ledger.csv")

#: Exact column order required by the recovery specification.
LEDGER_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "timestamp",
    "git_commit",
    "dataset_variant",
    "dataset_hash",
    "universe_id",
    "config_hash",
    "search_fold",
    "ticker_subset",
    "feature_set",
    "rsi_method",
    "lookback",
    "scaler",
    "hidden_size",
    "layers",
    "dropout",
    "max_epochs",
    "best_epoch",
    "patience",
    "weight_decay",
    "class_weighting",
    "calibration_method",
    "seed",
    "train_loss",
    "validation_loss",
    "validation_accuracy",
    "validation_f1",
    "validation_brier",
    "validation_ece",
    "test_evaluated",
    "epochs_run",
    "notes",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ledger_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / LEDGER_RELATIVE


def runtime_dir() -> Path:
    """Runtime (non-repository) output root for recovery artefacts."""
    from agentic_forecaster.config import get_env_roots
    return Path(get_env_roots()["AGENTIC_OUTPUT_ROOT"]) / "reproduction_recovery"


def git_commit(root: Path | None = None) -> str:
    """Current commit, or ``unknown`` outside a repository."""
    root = root or repo_root()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, check=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dict(payload) -> str:
    """Stable hash of a JSON-serialisable mapping, for config_hash/dataset_hash."""
    import json
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def new_experiment_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"EXP-{stamp}-{uuid.uuid4().hex[:6]}"


def read_ledger(path: Path | None = None) -> list[dict]:
    p = path or ledger_path()
    if not p.is_file():
        return []
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def append_experiment(record: dict, path: Path | None = None, *,
                      search: bool | None = None, root: Path | None = None) -> dict:
    """Append one immutable experiment row and return the completed record.

    Refuses to record a search experiment that claims ``test_evaluated``; that
    combination would mean the firewall leaked and the row is not trustworthy.
    """
    p = path or ledger_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)

    row = {c: record.get(c, "") for c in LEDGER_COLUMNS}
    # The comprehension above always populates every column (with ""), so
    # setdefault() would never fire. Fill explicitly instead.
    if not row["experiment_id"]:
        row["experiment_id"] = new_experiment_id()
    row["timestamp"] = row["timestamp"] or datetime.now(UTC).isoformat()
    row["git_commit"] = row["git_commit"] or git_commit(root)

    claimed = str(row["test_evaluated"]).strip().lower()
    if is_search_mode(search) and claimed in ("true", "1", "yes"):
        raise ValueError(
            "Refusing to append a SEARCH experiment with test_evaluated=true. "
            "2022/2023 must remain unseen during search; a row claiming otherwise "
            "invalidates the whole ledger."
        )
    if is_search_mode(search):
        row["test_evaluated"] = "false"

    new_file = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(LEDGER_COLUMNS))
        if new_file:
            writer.writeheader()
        writer.writerow({c: row[c] for c in LEDGER_COLUMNS})
    return row


def ledger_summary(path: Path | None = None) -> dict:
    rows = read_ledger(path)
    search_rows = [r for r in rows if str(r.get("search_fold", "")).startswith("SEARCH_FOLD")]
    return {
        "ledger_path": str(path or ledger_path()),
        "total_experiments": len(rows),
        "search_experiments": len(search_rows),
        "rows_claiming_test_evaluated": sum(
            1 for r in search_rows if str(r.get("test_evaluated", "")).lower() in ("true", "1", "yes")),
        "distinct_search_folds": sorted({r.get("search_fold", "") for r in search_rows}),
        "distinct_config_hashes": len({r.get("config_hash", "") for r in search_rows}),
    }
