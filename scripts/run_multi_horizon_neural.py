#!/usr/bin/env python3
"""STAGE 2 -- neural fits, only for horizons that SCREEN-PASSED.

For each selected horizon:

    N1  shared LSTM            (unchanged V2-A architecture)
    N2  shared LSTM + Transformer   (unchanged V2-B architecture)

across the five development years 2014-2018.

Deliberately NOT run in this programme: context, multi-task, FiLM and
meta-learning.  The previous PRE-COVID experiment found the plain shared LSTM
strongest while Transformer/context added little or hurt, and this programme exists
to establish whether the changed HORIZON itself improves predictability before any
further architecture work.

2019 is never read.  Nothing from 2020 is touched.

Usage::

    uv run python scripts/run_multi_horizon_neural.py --horizons 3 5
    uv run python scripts/run_multi_horizon_neural.py --horizons 5 --seeds 11 42 73
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
    "SHARED_LSTM": REPO_ROOT / "configs/v2/multi_horizon/shared_lstm.yaml",
    "LSTM_TRANSFORMER": REPO_ROOT / "configs/v2/multi_horizon/lstm_transformer.yaml",
}


def run_neural(model_name: str, horizons: list[int], *, seeds: list[int] | None = None,
               config_path: Path | None = None,
               folds: list[str] | None = None,
               append_ledger: bool = True) -> dict:
    payload = HZ.load_track_config(config_path or CONFIG)
    data = payload["data"]
    final_allowed = str(payload["final_allowed_date"])
    bootstrap = payload["evaluation"]["bootstrap"]
    sequence_length = int(data["sequence_length"])
    folds = list(folds or payload["development_folds"])
    seeds = list(seeds or [int(payload["training"]["seed"])])

    track = HZ.MultiHorizonTrack()
    ensure_dir(track.results_root)
    ensure_dir(track.runtime_root)

    tickers = sorted(pd.read_csv(track.path("universe_csv"))
                     .loc[lambda f: f["eligible"], "ticker"].tolist())
    store = load_store(Path(data["source_store_root"]),
                       final_allowed_date=final_allowed)
    arrays = build_horizon_arrays(
        store, load_sector_map(Path(data["sector_map_csv"])), tickers=tickers,
        use_context=False)
    targets, target_metadata = HZ.load_target_cache(
        Path(data["horizon_target_cache"]), final_allowed_date=final_allowed)
    tables = {int(h): build_horizon_sample_table(
        arrays, targets, horizon=int(h), sequence_length=sequence_length,
        final_allowed_date=final_allowed, locked=False) for h in horizons}

    model_config = {key: value for key, value in payload["model"].items()
                    if key not in {"use_transformer", "use_context",
                                   "use_sector_embedding", "use_regime",
                                   "use_multitask", "use_film", "use_adapter",
                                   "screen_model"}}
    train_settings = dict(payload["training"])
    fingerprints = store_fingerprints(Path(data["source_store_root"]))
    schema_hash = str(target_metadata["target_schema_sha256"])
    config_path = config_path or CONFIG_BY_MODEL[model_name]
    sha = hash_payload(HZ.load_track_config(config_path))

    runs: list[dict] = []
    for horizon in horizons:
        samples = tables[int(horizon)]
        for seed in seeds:
            for fold in folds:
                window = resolve_fold_window(fold, payload["folds"])
                common = common_origin_masks(tables, window,
                                             horizons=tuple(horizons), fold=fold)
                result = HS.run_neural_fold(
                    samples, horizon=int(horizon), window=window,
                    model_name=model_name, seed=int(seed),
                    device=str(payload["experiment"]["device"]),
                    model_config=model_config, train_settings=train_settings,
                    common_keys=common[int(horizon)]["keys"], locked=False,
                    n_bootstrap=int(bootstrap["n_bootstrap"]),
                    block_length=int(bootstrap["block_length_dates"]), fold=fold)

                experiment_id = HZ.new_experiment_id("MH")
                out_dir = experiment_dir(experiment_id, runtime_root=track.runtime_root)
                result.predictions.to_csv(out_dir / "predictions.csv", index=False)
                atomic_json_dump({
                    "experiment_id": experiment_id,
                    "track": HZ.TRACK_ID,
                    "objective_id": result.objective_id,
                    "horizon_phrase": HZ.horizon_phrase(horizon),
                    "model": model_name,
                    "model_label": HZ.NEURAL_LABELS[model_name],
                    "fold": fold,
                    "seed": int(seed),
                    "stage": "NEURAL",
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

                if append_ledger:
                    HZ.append_row({
                        "experiment_id": experiment_id,
                        "experiment_dir": str(out_dir),
                        "horizon": int(horizon),
                        "objective_id": result.objective_id,
                        "model": model_name,
                        "fold": fold,
                        "seed": int(seed),
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
                        "train_majority_baseline": result.metrics.get(
                            "train_majority_baseline"),
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
                runs.append({"experiment_id": experiment_id, "horizon": int(horizon),
                             "fold": fold, "seed": int(seed)})
                print(f"[neural] {model_name:>18} {HZ.objective_id(horizon):>18} "
                      f"{fold} seed={seed}: acc={result.metrics.get('accuracy'):.4f} "
                      f"auc={result.metrics.get('roc_auc'):.4f} "
                      f"bal={result.metrics.get('balanced_accuracy'):.4f} "
                      f"base={result.metrics.get('train_majority_baseline'):.4f} "
                      f"best_epoch={result.diagnostics['complexity']['best_epoch']} "
                      f"({result.diagnostics['complexity']['training_seconds']:.0f}s)",
                      flush=True)

    manifest = {
        "track": HZ.TRACK_ID,
        "stage": "NEURAL",
        "model": model_name,
        "model_label": HZ.NEURAL_LABELS[model_name],
        "horizons": [int(h) for h in horizons],
        "folds": folds,
        "seeds": seeds,
        "config": str(config_path),
        "config_sha256": sha,
        "architecture_unchanged": True,
        "architecture_source": "configs/v2/multi_horizon/base.yaml :: model",
        "components": HS.neural_components(model_name),
        "train_settings": train_settings,
        "hyperparameter_search": False,
        "n_supervised_tickers": len(tickers),
        "feature_schema_sha256": hash_payload(arrays.stock_features),
        "target_schema_sha256": schema_hash,
        "store_sha256": fingerprints.get("store_sha256"),
        "git_commit": git_commit(),
        "final_allowed_date": final_allowed,
        "2019_lockbox_evaluated": False,
        "post_2019_evaluated": False,
        "not_run_in_this_programme": ["CONTEXT", "MULTITASK", "FILM", "META_LEARNING"],
        "n_runs": len(runs),
    }
    atomic_json_dump(manifest, track.results_root / f"neural_{model_name.lower()}_manifest.json")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(HZ.NEURAL_MODELS), required=True)
    parser.add_argument("--horizons", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    setup_logging()
    manifest = run_neural(args.model, args.horizons, seeds=args.seeds,
                          config_path=args.config, folds=args.folds)
    print(json.dumps({k: manifest[k] for k in (
        "model", "horizons", "folds", "seeds", "n_runs", "config_sha256",
        "2019_lockbox_evaluated")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())