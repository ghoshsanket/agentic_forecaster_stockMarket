#!/usr/bin/env python3
"""The ONE 2019 lockbox run of the multi-horizon programme.

REFUSES TO RUN unless BOTH are present:

    MULTI_HORIZON_LOCKBOX=1                  (the explicit authorisation)
    results/v2/multi_horizon/frozen_horizon_model.json

and unless the frozen manifest still matches the requested horizon, architecture,
target schema, store, supervised universe and config.  A changed horizon or a
changed architecture invalidates the manifest, exactly as section 30 requires.

The run trains on 2005-01-01 .. 2018-12-31 and scores 2019-01-01 .. 2019-12-31 with
seed 42, EXACTLY ONCE.  The selected horizon's target must terminate no later than
2019-12-31: a late-December 2019 origin whose 3/5/10-trading-day target would fall
in 2020 is dropped, and no 2020 close is used to finish a 2019 label.

Usage::

    MULTI_HORIZON_LOCKBOX=1 uv run python scripts/run_multi_horizon_lockbox.py
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
from agentic_forecaster.v2 import horizon_metrics as HM
from agentic_forecaster.v2 import horizon_screen as HS
from agentic_forecaster.v2 import horizons as HZ
from agentic_forecaster.v2.dataset import resolve_fold_window
from agentic_forecaster.v2.horizon_dataset import build_horizon_arrays
from agentic_forecaster.v2.ledger import experiment_dir, hash_payload
from agentic_forecaster.v2.sectors import load_sector_map
from agentic_forecaster.v2.store import load_store, store_fingerprints

CONFIG = REPO_ROOT / "configs" / "v2" / "multi_horizon" / "base.yaml"


class LockboxRefused(RuntimeError):
    """The 2019 lockbox may not run with the current state."""


def authorise(manifest_path: Path, horizon: int | None, architecture: str | None, *,
              track: HZ.MultiHorizonTrack | None = None) -> dict:
    """Require the switch AND a valid frozen manifest; return it."""
    track = track or HZ.MultiHorizonTrack()
    if not HZ.lockbox_unlocked():
        raise LockboxRefused(
            f"refusing the 2019 lockbox: {HZ.LOCKBOX_ENV}=1 is not set. The lockbox may "
            "be opened exactly once, after horizon and model are frozen.")
    if not manifest_path.is_file():
        raise LockboxRefused(
            f"refusing the 2019 lockbox: no frozen manifest at {manifest_path}. "
            "horizon, architecture, thresholds, features and seed must be frozen "
            "first.")
    manifest = HS.load_frozen_manifest(manifest_path)
    requested_horizon = int(manifest["horizon"] if horizon is None else horizon)
    requested_architecture = str(manifest["selected_architecture"] if architecture is None
                                 else architecture)
    return {
        "manifest": manifest,
        "horizon": requested_horizon,
        "architecture": requested_architecture,
        "objective_id": HZ.objective_id(requested_horizon),
        "horizon_phrase": HZ.horizon_phrase(requested_horizon),
    }


def already_run(track: HZ.MultiHorizonTrack | None = None) -> bool:
    """True when a 2019 lockbox row already exists: the one-shot guard."""
    track = track or HZ.MultiHorizonTrack()
    return any(row.get("fold") == HZ.LOCKBOX_FOLD for row in HZ.read_ledger(track.ledger))


def verify_hashes(authorisation: dict, payload: dict) -> dict:
    """Every fingerprint the frozen manifest recorded must still hold."""
    manifest = authorisation["manifest"]
    track = HZ.MultiHorizonTrack()
    final_allowed = str(payload["final_allowed_date"])
    _, target_meta = HZ.load_target_cache(Path(payload["data"]["horizon_target_cache"]),
                                         final_allowed_date=final_allowed)
    universe = json.loads((track.results_root / "supervised_universe_frozen.json")
                          .read_text())
    result = HS.verify_frozen_manifest(
        manifest,
        horizon=authorisation["horizon"], architecture=authorisation["architecture"],
        target_schema_sha256=str(target_meta["target_schema_sha256"]),
        store_sha256=store_fingerprints(
            Path(payload["data"]["source_store_root"])).get("store_sha256"),
        universe_sha256=universe["universe_sha256"],
        config_sha256=hash_payload(payload))
    if not result["verified"]:
        raise LockboxRefused(result["message"])
    return result


def run_lockbox(*, config_path: Path = CONFIG, horizon: int | None = None,
                architecture: str | None = None, allow_rerun: bool = False,
                track: HZ.MultiHorizonTrack | None = None) -> dict:
    payload = HZ.load_track_config(config_path)
    track = track or HZ.MultiHorizonTrack()
    ensure_dir(track.results_root)
    ensure_dir(track.runtime_root)

    # The ONE-SHOT guard is checked FIRST: a second attempt must be refused before
    # anything else is even considered.
    if already_run(track) and not allow_rerun:
        raise LockboxRefused(
            "refusing to run the 2019 lockbox a second time: a lockbox row already "
            "exists in the ledger. The lockbox is a ONE-SHOT evaluation and another "
            "horizon may not be tested after seeing 2019.")
    authorisation = authorise(track.path("freeze"), horizon, architecture, track=track)
    verification = verify_hashes(authorisation, payload)

    horizon_selected = authorisation["horizon"]
    architecture_selected = authorisation["architecture"]
    seed = int(payload["seed_stability"]["lockbox_seed"])
    final_allowed = str(payload["final_allowed_date"])
    window = resolve_fold_window(HZ.LOCKBOX_FOLD, payload["folds"])

    tickers = sorted(pd.read_csv(track.path("universe_csv"))
                     .loc[lambda f: f["eligible"], "ticker"].tolist())
    store = load_store(Path(payload["data"]["source_store_root"]),
                       final_allowed_date=final_allowed)
    arrays = build_horizon_arrays(store,
                                  load_sector_map(Path(payload["data"]["sector_map_csv"])),
                                  tickers=tickers, use_context=False)
    # locked=True: this single run is the ONLY place the 2019 labels may be read.
    targets, _ = HZ.load_target_cache(Path(payload["data"]["horizon_target_cache"]),
                                     final_allowed_date=final_allowed)
    from agentic_forecaster.v2.horizon_dataset import build_horizon_sample_table

    samples = build_horizon_sample_table(
        arrays, targets, horizon=horizon_selected,
        sequence_length=int(payload["data"]["sequence_length"]),
        final_allowed_date=final_allowed, locked=True)

    model_config = {key: value for key, value in payload["model"].items()
                    if key not in {"use_transformer", "use_context",
                                   "use_sector_embedding", "use_regime",
                                   "use_multitask", "use_film", "use_adapter",
                                   "screen_model"}}
    result = HS.run_neural_fold(
        samples, horizon=horizon_selected, window=window,
        model_name=architecture_selected, seed=seed,
        device=str(payload["experiment"]["device"]), model_config=model_config,
        train_settings=dict(payload["training"]), common_keys=None, locked=True,
        n_bootstrap=int(payload["evaluation"]["bootstrap"]["n_bootstrap"]),
        block_length=int(payload["evaluation"]["bootstrap"]["block_length_dates"]),
        fold=HZ.LOCKBOX_FOLD)

    experiment_id = HZ.new_experiment_id("MHLOCKBOX")
    out_dir = experiment_dir(experiment_id, runtime_root=track.runtime_root)
    result.predictions.to_csv(out_dir / "predictions.csv", index=False)

    curve = HM.selective_table(result.predictions,
                               levels=tuple(payload["evaluation"]["coverage_levels"]))
    selective = {
        "confidence": "abs(p_up - 0.5)",
        "curve": curve.to_dict("records"),
        "max_meaningful_coverage": {
            f"{target:.2f}": HM.max_meaningful_coverage(
                result.predictions, target,
                min_coverage=float(payload["evaluation"]["min_meaningful_coverage"]),
                min_observations=int(
                    payload["evaluation"]["min_meaningful_observations"]))
            for target in payload["evaluation"]["coverage_accuracy_targets"]
        },
        "never_called": "selective accuracy is never called overall accuracy",
    }

    dropped_lockbox = int(samples.diagnostics["lockbox_year_excluded_rows"])
    max_target = str(pd.Timestamp(result.predictions["target_date"].max()).date())
    report = {
        "track": HZ.TRACK_ID,
        "experiment_id": experiment_id,
        "objective_id": authorisation["objective_id"],
        "horizon": horizon_selected,
        "horizon_trading_observations": horizon_selected,
        "horizon_phrase": authorisation["horizon_phrase"],
        "exact_meaning": (
            f"the model predicts whether Close[t+{horizon_selected}] is above or below "
            f"Close[t], where t+{horizon_selected} is the {horizon_selected}-th "
            "subsequent TRADING observation"),
        "target_equation": HZ.TARGET_EQUATION,
        "trading_day_rule": HZ.TRADING_DAY_RULE,
        "selected_architecture": architecture_selected,
        "seed": seed,
        "runs": 1,
        "fold": HZ.LOCKBOX_FOLD,
        "window": {"train_start": window.train_start, "train_end": window.train_end,
                   "val_start": window.val_start, "val_end": window.val_end},
        "metrics_all_valid_origins": result.metrics,
        "metrics_non_overlapping": result.non_overlap_metrics,
        "selective_accuracy": selective,
        "frozen_manifest_verification": verification,
        "frozen_manifest_sha256": authorisation["manifest"].get("manifest_sha256"),
        "boundary": {
            "final_allowed_date": final_allowed,
            "max_target_end_date_scored": max_target,
            "target_ends_in_2020": bool(pd.Timestamp(max_target) > pd.Timestamp(
                final_allowed)),
            "december_2019_origins_whose_target_would_reach_2020": dropped_lockbox,
            "2020_close_used_to_finish_a_2019_label": False,
        },
        "data_access": {
            "max_origin_date_consumed": str(
                pd.Timestamp(result.predictions["origin_date"].max()).date()),
            "max_target_end_date_consumed": max_target,
            "lockbox_year_consumed": True,
            "post_2019_consumed": 0,
        },
        "firewall": HZ.describe_firewall(),
    }
    atomic_json_dump(report, out_dir / "lockbox_report.json")

    metrics = result.metrics
    row = HZ.append_row({
        "experiment_id": experiment_id,
        "experiment_dir": str(out_dir),
        "horizon": horizon_selected,
        "objective_id": authorisation["objective_id"],
        "model": architecture_selected,
        "fold": HZ.LOCKBOX_FOLD,
        "seed": seed,
        "n_train": result.n_train,
        "n_validation": result.n_val,
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
        "non_overlap_accuracy": result.non_overlap_metrics.get("accuracy"),
        "non_overlap_auc": result.non_overlap_metrics.get("roc_auc"),
        "common_origin_accuracy": "",
        "common_origin_auc": "",
        "config_sha256": hash_payload(payload),
        "target_schema_sha256": authorisation["manifest"]["target_schema_sha256"],
        "2019_lockbox_evaluated": True,
        "post_2019_evaluated": False,
    }, path=track.ledger, lockbox_evaluated=True)

    print(json.dumps({
        "experiment_id": experiment_id,
        "objective_id": authorisation["objective_id"],
        "horizon_phrase": authorisation["horizon_phrase"],
        "architecture": architecture_selected,
        "accuracy": metrics.get("accuracy"),
        "macro_accuracy": metrics.get("macro_ticker_accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "roc_auc": metrics.get("roc_auc"),
        "brier": metrics.get("brier"),
        "train_majority_baseline": metrics.get("train_majority_baseline"),
        "non_overlap_accuracy": result.non_overlap_metrics.get("accuracy"),
        "non_overlap_auc": result.non_overlap_metrics.get("roc_auc"),
        "max_target_end_date_scored": max_target,
        "selective_max_meaningful_coverage": {
            key: value.get("coverage")
            for key, value in selective["max_meaningful_coverage"].items()},
        "ledger_row": row["experiment_id"],
    }, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--architecture", choices=sorted(HZ.NEURAL_MODELS), default=None)
    args = parser.parse_args(argv)
    setup_logging()
    run_lockbox(config_path=args.config, horizon=args.horizon,
                architecture=args.architecture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())