#!/usr/bin/env python3
"""The ONE 2019 lockbox run of the V3 exogenous programme.

REFUSES TO RUN unless BOTH are present:

    V3_PRECOVID_LOCKBOX=1
    results/v3/pre_covid_exogenous/frozen_exogenous_model.json

and unless the frozen manifest still matches the requested feature family, horizon,
architecture, exogenous store, source manifest and config.  Changing any of them
invalidates the manifest, because section 42 freezes them before 2019 is read.

Trains on 2005-01-01 .. 2018-12-31 and scores 2019 only, with seed 42, EXACTLY
ONCE.  For H = 3/5/10 the ``target_end_date`` must still terminate no later than
2019-12-31: a late-December 2019 origin whose target would fall in 2020 is dropped
and no 2020 observation ever finishes a 2019 label.

Usage::

    V3_PRECOVID_LOCKBOX=1 uv run python scripts/run_v3_precovid_lockbox.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.utils import atomic_json_dump, ensure_dir, setup_logging
from agentic_forecaster.v2.dataset import resolve_fold_window
from agentic_forecaster.v2.horizon_metrics import max_meaningful_coverage, selective_table
from agentic_forecaster.v2.horizons import horizon_phrase, load_track_config, objective_id
from agentic_forecaster.v2.ledger import experiment_dir, hash_payload
from agentic_forecaster.v3 import V3Track
from agentic_forecaster.v3.experiment import (
    append_row,
    assemble_inputs,
    new_experiment_id,
    read_ledger,
)
from agentic_forecaster.v3.store import load_exogenous_store

CONFIG = REPO_ROOT / "configs" / "v3" / "precovid_exogenous_base.yaml"

LOCKBOX_ENV = "V3_PRECOVID_LOCKBOX"
LOCKBOX_FOLD = "MH_LOCKBOX_2019"


class LockboxRefused(RuntimeError):
    """The 2019 lockbox may not run in the current state."""


def authorise(manifest_path: Path, family: str | None, horizon: int | None,
              architecture: str | None) -> dict:
    """Require the switch AND a valid frozen manifest; return it."""
    import os

    if os.environ.get(LOCKBOX_ENV, "") != "1":
        raise LockboxRefused(
            f"refusing the 2019 lockbox: {LOCKBOX_ENV}=1 is not set. The lockbox is "
            "read exactly once, after the information family, horizon and model are "
            "frozen.")
    if not manifest_path.is_file():
        raise LockboxRefused(
            f"refusing the 2019 lockbox: no frozen manifest at {manifest_path}. The "
            "family, horizon, architecture, thresholds, sources and seed must be "
            "frozen first.")
    manifest = json.loads(manifest_path.read_text())
    return {
        "manifest": manifest,
        "family": family or manifest["feature_family"],
        "horizon": int(horizon if horizon is not None else manifest["horizon"]),
        "architecture": architecture or manifest["architecture"],
    }


def already_run(track: V3Track | None = None) -> bool:
    """True when a 2019 lockbox row exists: the one-shot guard."""
    track = track or V3Track()
    return any(row.get("fold") == LOCKBOX_FOLD for row in read_ledger(track.ledger))


def verify_hashes(authorisation: dict, payload: dict) -> dict:
    """Every fingerprint the frozen manifest recorded must still hold."""

    manifest = authorisation["manifest"]
    store = load_exogenous_store(Path(payload["data"]["exogenous_store"]),
                                 final_allowed_date=str(payload["final_allowed_date"]))
    current = {
        "feature_family": manifest.get("feature_family"),
        "horizon": int(manifest.get("horizon", -1)),
        "architecture": manifest.get("architecture"),
        "exogenous_feature_schema_hash": store.schema_sha256,
        "source_manifest_sha256": (
            json.loads(Path(payload["data"]["source_manifest"]).read_text())
            ["manifest_sha256"]
            if Path(payload["data"]["source_manifest"]).is_file() else None),
        "config_sha256": hash_payload(payload),
    }
    requested = {
        "feature_family": authorisation["family"],
        "horizon": authorisation["horizon"],
        "architecture": authorisation["architecture"],
        "exogenous_feature_schema_hash": store.schema_sha256,
        "source_manifest_sha256": current["source_manifest_sha256"],
        "config_sha256": current["config_sha256"],
    }
    failed = [name for name in requested if requested[name] != current[name]]
    return {"verified": not failed, "failed_checks": failed,
            "message": ("frozen manifest verified" if not failed else
                        "refusing the 2019 lockbox: " + ", ".join(failed))}


def run_lockbox(*, config_path: Path = CONFIG, family: str | None = None,
                horizon: int | None = None, architecture: str | None = None,
                track: V3Track | None = None,
                allow_rerun: bool = False) -> dict:
    payload = load_track_config(config_path)
    track = track or V3Track()
    ensure_dir(track.results_root)
    ensure_dir(track.runtime_root)

    # the ONE-SHOT guard is checked FIRST
    if already_run(track) and not allow_rerun:
        raise LockboxRefused(
            "refusing to run the 2019 lockbox a second time: a lockbox row already "
            "exists. It is a ONE-SHOT evaluation and no other family, horizon or model "
            "may be tested after seeing 2019.")
    authorisation = authorise(track.path("freeze"), family, horizon, architecture)
    verification = verify_hashes(authorisation, payload)
    if not verification["verified"]:
        raise LockboxRefused(verification["message"])

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "v3_neural_runner", REPO_ROOT / "scripts" / "run_v3_exogenous_neural.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    seed = int(payload["seed_stability"]["lockbox_seed"])
    inputs = assemble_inputs(config_path, locked=True)
    window = resolve_fold_window(LOCKBOX_FOLD, payload["folds"])
    outcome = runner.run_neural_fold(
        inputs, family=authorisation["family"], horizon=authorisation["horizon"],
        window=window, fold=LOCKBOX_FOLD, architecture=authorisation["architecture"],
        seed=seed, device=str(payload["experiment"]["device"]),
        bootstrap=dict(payload["evaluation"]["bootstrap"]), locked=True)

    experiment_id = new_experiment_id("V3LOCKBOX")
    out_dir = experiment_dir(experiment_id, runtime_root=track.runtime_root)
    predictions = outcome.pop("predictions")
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    from agentic_forecaster.v3.checkpoint import save_neural_checkpoint

    save_neural_checkpoint(outcome.pop("model"), out_dir,
                           extra=outcome["parameters"] | {
                               "architecture": authorisation["architecture"]})

    curve = selective_table(
        predictions,
        levels=tuple(payload["evaluation"]["coverage_levels"]))
    selective = {
        "confidence": "abs(p_up - 0.5)",
        "curve": curve.to_dict("records"),
        "max_meaningful_coverage": {
            f"{target:.2f}": max_meaningful_coverage(
                predictions, target,
                min_coverage=float(payload["evaluation"]["min_meaningful_coverage"]),
                min_observations=int(
                    payload["evaluation"]["min_meaningful_observations"]))
            for target in payload["evaluation"]["coverage_accuracy_targets"]},
        "never_called": "selective accuracy is never called overall accuracy",
    }
    max_target = str(pd_max(predictions["target_date"]))
    report = {
        "track": "V3_EXOGENOUS_PRECOVID",
        "experiment_id": experiment_id,
        "feature_family": authorisation["family"],
        "horizon": authorisation["horizon"],
        "objective_id": objective_id(authorisation["horizon"]),
        "horizon_phrase": horizon_phrase(authorisation["horizon"]),
        "exact_meaning": (f"the model predicts whether Close[t+{authorisation['horizon']}] "
                          f"is above or below Close[t], where t+{authorisation['horizon']} "
                          "is that many subsequent TRADING observations"),
        "architecture": authorisation["architecture"],
        "seed": seed,
        "runs": 1,
        "fold": LOCKBOX_FOLD,
        "window": {"train_start": window.train_start, "train_end": window.train_end,
                   "val_start": window.val_start, "val_end": window.val_end},
        "metrics": outcome["metrics"],
        "metrics_non_overlapping": outcome["non_overlapping"],
        "selective_accuracy": selective,
        "frozen_manifest_verification": verification,
        "boundary": {
            "final_allowed_date": str(payload["final_allowed_date"]),
            "max_target_end_date_scored": max_target,
            "target_ends_in_2020": bool(pd_max(predictions["target_date"])
                                        > pd.Timestamp(
                                            str(payload["final_allowed_date"]))),
            "2020_close_used_to_finish_a_2019_label": False,
        },
        "data_access": {
            "max_origin_date_consumed": str(pd_max(predictions["origin_date"])),
            "max_target_end_date_consumed": max_target,
            "lockbox_year_consumed": True,
            "post_2019_consumed": 0,
        },
    }
    atomic_json_dump(report, out_dir / "lockbox_report.json")

    metrics = outcome["metrics"]
    append_row({
        "experiment_id": experiment_id,
        "experiment_dir": str(out_dir),
        "model": authorisation["architecture"],
        "feature_family": authorisation["family"],
        "horizon": authorisation["horizon"],
        "objective_id": objective_id(authorisation["horizon"]),
        "fold": LOCKBOX_FOLD,
        "seed": seed,
        "n_train": outcome["n_train"],
        "n_validation": outcome["n_validation"],
        "n_tickers": metrics.get("n_tickers"),
        "accuracy": metrics.get("accuracy"),
        "macro_accuracy": metrics.get("macro_ticker_accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "brier": metrics.get("brier"),
        "ece": metrics.get("ece"),
        "train_majority_baseline": metrics.get("train_majority_baseline"),
        "baseline_delta": metrics.get("baseline_delta"),
        "stock_only_auc": "",
        "incremental_auc": "",
        "common_sample_auc": "",
        "non_overlap_auc": outcome["non_overlapping"].get("roc_auc"),
        "config_sha256": hash_payload(payload),
        "source_manifest_sha256": authorisation["manifest"].get(
            "source_manifest_sha256"),
        "feature_schema_sha256": authorisation["manifest"].get(
            "exogenous_feature_schema_hash"),
        "2019_lockbox_evaluated": True,
        "post_2019_evaluated": False,
    }, path=track.ledger, lockbox_evaluated=True)

    print(json.dumps({
        "experiment_id": experiment_id,
        "feature_family": authorisation["family"],
        "objective_id": objective_id(authorisation["horizon"]),
        "architecture": authorisation["architecture"],
        "accuracy": metrics.get("accuracy"),
        "macro_accuracy": metrics.get("macro_ticker_accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "roc_auc": metrics.get("roc_auc"),
        "train_majority_baseline": metrics.get("train_majority_baseline"),
        "non_overlap_auc": outcome["non_overlapping"].get("roc_auc"),
        "max_target_end_date_scored": max_target,
        "selective_max_meaningful_coverage": {
            key: value.get("coverage")
            for key, value in selective["max_meaningful_coverage"].items()},
    }, indent=2))
    return report


def pd_max(values):
    """Latest date in a column, as a Timestamp."""
    return pd.Timestamp(pd.to_datetime(values).max())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--family", default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--architecture", default=None)
    args = parser.parse_args(argv)
    setup_logging()
    run_lockbox(config_path=args.config, family=args.family, horizon=args.horizon,
                architecture=args.architecture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())