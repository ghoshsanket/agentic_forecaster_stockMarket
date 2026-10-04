#!/usr/bin/env python3
"""STAGE 1 -- the cheap multi-horizon screen.

Runs two FIXED models at every horizon and every development year:

    M1 LogisticRegression               (StandardScaler fitted on TRAIN only)
    M2 HistGradientBoostingClassifier   (fixed, recorded settings)

for horizons 1D (CONTROL) / 3D / 5D / 10D over 2014-2018.

No hyper-parameter search is performed.  Nothing from 2020 is touched, and 2019 is
never scored: the development sample tables physically exclude the lockbox year and
every development split is audited against the ``PostCovidDataAccessError`` /
``MultiHorizonLockboxError`` firewalls.

Three sample views are reported for every run, because different horizons naturally
lose different year-end samples:

    all_valid_origins   every valid sample for that horizon
    common_origin       origins valid at 1D, 3D, 5D AND 10D inside the same split
    non_overlapping     origins spaced at least H trading observations apart

Outputs::

    results/v2/multi_horizon/experiment_ledger.csv      (append-only)
    $AGENTIC_OUTPUT_ROOT/v2/multi_horizon/<experiment>/  (predictions + metrics)

Usage::

    uv run python scripts/run_multi_horizon_screen.py --model LOGISTIC
    uv run python scripts/run_multi_horizon_screen.py --model HIST_GRADIENT_BOOSTING
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
from agentic_forecaster.v2 import horizon_screen as HS
from agentic_forecaster.v2 import horizons as HZ
from agentic_forecaster.v2.dataset import resolve_fold_window
from agentic_forecaster.v2.horizon_dataset import (
    build_horizon_arrays,
    build_horizon_sample_table,
    common_origin_masks,
)
from agentic_forecaster.v2.ledger import experiment_dir, git_commit, hash_payload
from agentic_forecaster.v2.sectors import load_sector_map
from agentic_forecaster.v2.store import load_store, store_fingerprints

CONFIG = REPO_ROOT / "configs" / "v2" / "multi_horizon" / "base.yaml"
CONFIG_BY_MODEL = {
    "LOGISTIC": REPO_ROOT / "configs/v2/multi_horizon/logistic.yaml",
    "HIST_GRADIENT_BOOSTING": REPO_ROOT
    / "configs/v2/multi_horizon/hist_gradient_boosting.yaml",
}


def eligible_tickers(track: HZ.MultiHorizonTrack) -> list[str]:
    frame = pd.read_csv(track.path("universe_csv"))
    return sorted(frame.loc[frame["eligible"], "ticker"].tolist())


def config_sha256(path: Path) -> str:
    return hash_payload(HZ.load_track_config(path))


def run_model(model_name: str, *, config_path: Path = CONFIG,
              horizons: tuple[int, ...] | None = None,
              folds: tuple[str, ...] | None = None,
              reset_ledger: bool = False) -> dict:
    """Run the whole screen for one model and append every row to the ledger."""
    payload = HZ.load_track_config(config_path)
    data = payload["data"]
    final_allowed = str(payload["final_allowed_date"])
    bootstrap = payload["evaluation"]["bootstrap"]
    sequence_length = int(data["sequence_length"])
    horizons = tuple(horizons or [int(h) for h in payload["horizons"]])
    folds = tuple(folds or payload["development_folds"])

    track = HZ.MultiHorizonTrack()
    ensure_dir(track.results_root)
    ensure_dir(track.runtime_root)
    if reset_ledger and track.ledger.is_file():
        track.ledger.unlink()

    tickers = eligible_tickers(track)
    store = load_store(Path(data["source_store_root"]),
                       final_allowed_date=final_allowed)
    sector_map = load_sector_map(Path(data["sector_map_csv"]))
    arrays = build_horizon_arrays(store, sector_map, tickers=tickers,
                                  use_context=bool(data.get("use_context", False)))
    targets, target_metadata = HZ.load_target_cache(
        Path(data["horizon_target_cache"]), final_allowed_date=final_allowed)

    tables = {
        horizon: build_horizon_sample_table(
            arrays, targets, horizon=horizon, sequence_length=sequence_length,
            final_allowed_date=final_allowed, locked=False)
        for horizon in horizons
    }
    settings = HS.SCREEN_SETTINGS[model_name]
    fingerprints = store_fingerprints(Path(data["source_store_root"]))
    schema_hash = str(target_metadata["target_schema_sha256"])
    sha = config_sha256(config_path)

    runs: list[dict] = []
    for horizon in horizons:
        samples = tables[horizon]
        for fold in folds:
            window = resolve_fold_window(fold, payload["folds"])
            common = common_origin_masks(tables, window, horizons=horizons, fold=fold)
            result = HS.run_screen_fold(
                samples, horizon=horizon, window=window, model_name=model_name,
                common_keys=common[horizon]["keys"], locked=False, settings=settings,
                fold=fold, n_bootstrap=int(bootstrap["n_bootstrap"]),
                block_length=int(bootstrap["block_length_dates"]),
                seed=int(bootstrap["seed"]))

            experiment_id = HZ.new_experiment_id("MH")
            out_dir = experiment_dir(experiment_id, runtime_root=track.runtime_root)
            result.predictions.to_csv(out_dir / "predictions.csv", index=False)
            atomic_json_dump({
                "experiment_id": experiment_id,
                "track": HZ.TRACK_ID,
                "objective_id": result.objective_id,
                "horizon_phrase": HZ.horizon_phrase(horizon),
                "model": model_name,
                "model_label": HZ.SCREEN_LABELS[model_name],
                "fold": fold,
                "stage": "HORIZON_SCREEN",
                "metrics_all_valid_origins": result.metrics,
                "metrics_common_origin": result.common_metrics,
                "metrics_non_overlapping": result.non_overlap_metrics,
                "diagnostics": result.diagnostics,
                "data_access": {
                    "final_allowed_date": final_allowed,
                    "max_origin_date_consumed": str(
                        pd.Timestamp(result.predictions["origin_date"].max()).date()),
                    "max_target_end_date_consumed": str(
                        pd.Timestamp(result.predictions["target_date"].max()).date()),
                    "lockbox_year_consumed": False,
                    "post_2019_consumed": 0,
                },
            }, out_dir / "summary.json")

            row = HZ.append_row({
                "experiment_id": experiment_id,
                "experiment_dir": str(out_dir),
                "horizon": horizon,
                "objective_id": result.objective_id,
                "model": model_name,
                "fold": fold,
                "seed": int(bootstrap["seed"]),
                "n_train": result.n_train,
                "n_validation": result.n_val,
                "n_tickers": result.metrics.get("n_tickers"),
                "accuracy": result.metrics.get("accuracy"),
                "macro_accuracy": result.metrics.get("macro_ticker_accuracy"),
                "balanced_accuracy": result.metrics.get("balanced_accuracy"),
                "f1": result.metrics.get("f1"),
                "roc_auc": result.metrics.get("roc_auc"),
                "brier": result.metrics.get("brier"),
                "ece": result.metrics.get("ece"),
                "train_majority_baseline": result.metrics.get("train_majority_baseline"),
                "baseline_delta": result.metrics.get("baseline_delta"),
                "non_overlap_accuracy": result.non_overlap_metrics.get("accuracy"),
                "non_overlap_auc": result.non_overlap_metrics.get("roc_auc"),
                "common_origin_accuracy": result.common_metrics.get("accuracy"),
                "common_origin_auc": result.common_metrics.get("roc_auc"),
                "config_sha256": sha,
                "target_schema_sha256": schema_hash,
                "2019_lockbox_evaluated": False,
                "post_2019_evaluated": False,
            }, path=track.ledger)
            runs.append({"row": row, "out_dir": str(out_dir),
                         "common_n": common[horizon]["n_val"]})
            print(f"[screen] {model_name:>22} {HZ.objective_id(horizon):>18} {fold}: "
                  f"acc={result.metrics.get('accuracy'):.4f} "
                  f"auc={result.metrics.get('roc_auc'):.4f} "
                  f"bal={result.metrics.get('balanced_accuracy'):.4f} "
                  f"base={result.metrics.get('train_majority_baseline'):.4f} "
                  f"(common n={common[horizon]['n_val']}, "
                  f"nonoverlap n={result.non_overlap_metrics.get('n')})", flush=True)

    manifest = {
        "track": HZ.TRACK_ID,
        "stage": "HORIZON_SCREEN",
        "model": model_name,
        "model_label": HZ.SCREEN_LABELS[model_name],
        "config": str(config_path),
        "config_sha256": sha,
        "settings": settings,
        "hyperparameter_search": False,
        "horizons": list(horizons),
        "folds": list(folds),
        "sequence_length": sequence_length,
        "feature_schema": arrays.stock_features,
        "feature_schema_sha256": hash_payload(arrays.stock_features),
        "n_supervised_tickers": len(tickers),
        "supervised_tickers": tickers,
        "universe_sha256": json.loads(
            (track.results_root / "supervised_universe_frozen.json").read_text()
        )["universe_sha256"],
        "target_schema_sha256": schema_hash,
        "store_sha256": fingerprints.get("store_sha256"),
        "git_commit": git_commit(),
        "final_allowed_date": final_allowed,
        "lockbox_env_var": HZ.LOCKBOX_ENV,
        "2019_lockbox_evaluated": False,
        "post_2019_evaluated": False,
        "n_runs": len(runs),
        "firewall": HZ.describe_firewall(),
    }
    name = ("logistic" if model_name == "LOGISTIC" else "hist_gradient_boosting")
    atomic_json_dump(manifest, track.results_root / f"screen_{name}_manifest.json")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(HZ.SCREEN_MODELS), required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--horizons", type=int, nargs="*", default=None)
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--reset-ledger", action="store_true")
    args = parser.parse_args(argv)
    setup_logging()
    config_path = args.config or CONFIG_BY_MODEL[args.model]
    manifest = run_model(args.model, config_path=config_path,
                         horizons=tuple(args.horizons) if args.horizons else None,
                         folds=tuple(args.folds) if args.folds else None,
                         reset_ledger=args.reset_ledger)
    print(json.dumps({k: manifest[k] for k in (
        "model", "n_runs", "horizons", "folds", "n_supervised_tickers",
        "config_sha256", "target_schema_sha256", "2019_lockbox_evaluated")},
        indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())