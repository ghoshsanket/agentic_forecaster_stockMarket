#!/usr/bin/env python3
"""Build and cache the 1D/3D/5D/10D horizon targets, then FREEZE the common
supervised universe.

Both steps happen BEFORE any model is trained.

1. HORIZON TARGET CACHE
     y_H = 1 if log(Close[t+H] / Close[t]) > 0 else 0,  H in {1, 3, 5, 10}

   ``H`` counts FUTURE TRADING OBSERVATIONS of the security: the target row is the
   H-th subsequent row of that security's ordered trading sequence, so weekends and
   exchange holidays are skipped by construction.  ``date + timedelta(days=H)`` is
   never used.

   A target that would end after 2019-12-31 is DROPPED, so no 2020 close ever
   finishes a 2019 label.

   The 1-day control is asserted to reproduce the existing PRE-COVID
   ``y_direction`` EXACTLY (label, target date and return), which is what makes it a
   control rather than a second experiment.

2. SUPERVISED UNIVERSE (frozen)

   Stock-only eligibility: a security must have >= 1000 valid H=10 training samples
   in MH_DEV_2014 and >= 180 valid validation samples in EVERY development year
   after horizon boundary trimming.  H=10 is the most restrictive horizon, so it
   establishes ONE fixed common universe that makes 1D/3D/5D/10D comparable.

   No context requirement may eliminate a stock-only sample.

Outputs::

    $AGENTIC_PROCESSED_DATA_ROOT/v2/multi_horizon/horizon_targets/
        targets.parquet
        metadata.json
    results/v2/multi_horizon/supervised_universe.csv
    results/v2/multi_horizon/supervised_universe_frozen.json

Usage::

    uv run python scripts/build_multi_horizon_targets.py
    uv run python scripts/build_multi_horizon_targets.py --verify
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
from agentic_forecaster.v2 import horizons as HZ
from agentic_forecaster.v2.dataset import resolve_fold_window
from agentic_forecaster.v2.horizon_dataset import build_horizon_sample_table
from agentic_forecaster.v2.sectors import load_sector_map
from agentic_forecaster.v2.store import (
    assert_store_within_pre_covid,
    load_store,
    store_fingerprints,
)

CONFIG = REPO_ROOT / "configs" / "v2" / "multi_horizon" / "base.yaml"

#: Required column order of the frozen universe CSV (section 11).
UNIVERSE_COLUMNS: tuple[str, ...] = (
    "ticker",
    "raw_first_date",
    "first_usable_date",
    "samples_1d",
    "samples_3d",
    "samples_5d",
    "samples_10d",
    "eligible",
    "exclusion_reason",
)

#: Additional, explicitly-labelled evidence columns appended after the required set.
EVIDENCE_COLUMNS: tuple[str, ...] = (
    "train_10d_mh_dev_2014",
    "val_min_1d",
    "val_min_3d",
    "val_min_5d",
    "val_min_10d",
    "sector",
)


def build_targets(config_path: Path) -> dict:
    payload = HZ.load_track_config(config_path)
    data = payload["data"]
    final_allowed = str(payload["final_allowed_date"])
    horizons = tuple(int(h) for h in payload["horizons"])

    source_store_root = Path(data["source_store_root"])
    store = load_store(source_store_root, final_allowed_date=final_allowed)
    regime = assert_store_within_pre_covid(store.metadata,
                                           final_allowed_date=final_allowed)
    fingerprints = store_fingerprints(source_store_root)

    targets = HZ.build_horizon_target_frame(
        data["dataset_root"], variant=str(data["dataset_variant"]),
        tickers=sorted(store.stock["ticker"].unique().tolist()),
        horizons=horizons, final_allowed_date=final_allowed,
        where="multi-horizon target build")

    control = HZ.assert_control_matches_store(targets, store.targets,
                                             horizon=int(payload["control_horizon"]))

    schema = HZ.target_schema(
        targets, store_sha256=fingerprints.get("store_sha256"),
        source_manifest_sha256=fingerprints.get("source_manifest_sha256"),
        feature_schema_sha256=HZ.hash_payload(list(payload["data"].get(
            "feature_schema", [])) or _stock_feature_schema()),
        final_allowed_date=final_allowed, horizons=horizons)
    schema["control_identity_check"] = control
    schema["source_store"] = {
        "root": str(source_store_root), "regime_check": regime,
        "read_only": True,
        "note": "the PRE-COVID store is reused as-is and is never rebuilt by this track",
    }
    written = HZ.write_target_cache(targets, schema=schema)

    universe = freeze_universe(config_path, targets)
    summary = {
        "track": HZ.TRACK_ID,
        "final_allowed_date": final_allowed,
        "horizons": list(horizons),
        "objectives": {str(h): HZ.objective_id(h) for h in horizons},
        "target_schema_sha256": written["target_schema_sha256"],
        "target_cache": str(HZ.target_cache_root()),
        "control_identity_check": control,
        "per_horizon": written["per_horizon"],
        "universe": universe,
        "source_store_sha256": fingerprints.get("store_sha256"),
    }
    track = HZ.MultiHorizonTrack()
    atomic_json_dump(summary, track.results_root / "target_and_universe_summary.json")
    return summary


def _stock_feature_schema() -> list[str]:
    from agentic_forecaster.v2 import features as feat

    return list(feat.STOCK_FEATURE_NAMES)


def freeze_universe(config_path: Path, targets: pd.DataFrame) -> dict:
    """Derive and FREEZE the common supervised universe (section 11)."""
    payload = HZ.load_track_config(config_path)
    data = payload["data"]
    rules = payload["universe"]
    final_allowed = str(payload["final_allowed_date"])
    horizons = tuple(int(h) for h in payload["horizons"])
    reference_horizon = int(rules["reference_horizon"])
    reference_fold = str(rules["reference_fold"])
    min_train = int(rules["min_train_samples_10d_mh_dev_2014"])
    min_val = int(rules["min_validation_samples_per_year"])
    sequence_length = int(data["sequence_length"])

    store = load_store(Path(data["source_store_root"]), final_allowed_date=final_allowed)
    sector_of = load_sector_map(Path(data["sector_map_csv"])).sector_series()

    arrays = _stock_only_arrays(store, sorted(store.stock["ticker"].unique().tolist()))
    tables = {
        horizon: build_horizon_sample_table(
            arrays, targets, horizon=horizon, sequence_length=sequence_length,
            final_allowed_date=final_allowed, locked=False)
        for horizon in horizons
    }

    folds = payload["folds"]
    window = resolve_fold_window(reference_fold, folds)
    counts: dict[tuple[str, int], dict] = {}
    for horizon, samples in tables.items():
        frame = samples.frame
        origin = pd.to_datetime(frame["origin_date"])
        target = pd.to_datetime(frame["target_date"])
        train = ((origin >= pd.Timestamp(window.train_start))
                 & (origin <= pd.Timestamp(window.train_end))
                 & (target >= pd.Timestamp(window.train_start))
                 & (target <= pd.Timestamp(window.train_end)))
        per_year = {}
        for year in range(2014, 2019):
            in_year = ((origin >= pd.Timestamp(f"{year}-01-01"))
                       & (origin <= pd.Timestamp(f"{year}-12-31"))
                       & (target >= pd.Timestamp(f"{year}-01-01"))
                       & (target <= pd.Timestamp(f"{year}-12-31")))
            per_year[year] = int(in_year.sum())
        for ticker, group in frame.groupby("ticker", sort=True):
            index = group.index
            key = (str(ticker), int(horizon))
            entry = counts.setdefault(key, {"total": 0, "train_ref": 0,
                                            "per_year": {y: 0 for y in per_year},
                                            "first_usable": None})
            entry["total"] += len(index)
            entry["train_ref"] += int(train.loc[index].sum())
            for year in per_year:
                entry["per_year"][year] += int(in_year.loc[index].sum())
            first = str(pd.Timestamp(origin.loc[index].min()).date())
            entry["first_usable"] = (min(entry["first_usable"], first)
                                     if entry["first_usable"] else first)

    raw_first = {str(ticker): str(pd.Timestamp(
        store.stock.loc[store.stock["ticker"] == ticker, "date"].min()).date())
        for ticker in store.stock["ticker"].unique()}

    rows: list[dict] = []
    for ticker in sorted(raw_first):
        row = {"ticker": ticker, "raw_first_date": raw_first[ticker],
               "first_usable_date": None,
               "eligible": False, "exclusion_reason": "",
               "sector": sector_of.get(ticker, "UNKNOWN")}
        for horizon in horizons:
            entry = counts.get((ticker, horizon))
            row[f"samples_{horizon}d"] = int(entry["total"]) if entry else 0
            if entry and row["first_usable_date"] is None:
                row["first_usable_date"] = entry["first_usable"]
        reference = counts.get((ticker, reference_horizon))
        row["train_10d_mh_dev_2014"] = int(reference["train_ref"]) if reference else 0
        for horizon in horizons:
            entry = counts.get((ticker, horizon))
            row[f"val_min_{horizon}d"] = (min(entry["per_year"].values())
                                          if entry else 0)

        reasons: list[str] = []
        if all(row[f"samples_{h}d"] == 0 for h in horizons):
            reasons.append("no usable causal sample at any horizon (insufficient history)")
        if row["train_10d_mh_dev_2014"] < min_train:
            reasons.append(f"only {row['train_10d_mh_dev_2014']} valid H=10 training "
                           f"samples in {reference_fold} (need {min_train})")
        for horizon in horizons:
            if row[f"val_min_{horizon}d"] < min_val:
                reasons.append(f"only {row[f'val_min_{horizon}d']} valid validation "
                               f"samples in the weakest development year at H={horizon} "
                               f"(need {min_val})")
        row["eligible"] = not reasons
        row["exclusion_reason"] = "; ".join(reasons)
        rows.append(row)

    frame = pd.DataFrame(rows, columns=list(UNIVERSE_COLUMNS) + list(EVIDENCE_COLUMNS))
    frame = frame.sort_values("ticker").reset_index(drop=True)

    track = HZ.MultiHorizonTrack()
    ensure_dir(track.results_root)
    csv_path = track.path("universe_csv")
    frame.to_csv(csv_path, index=False)
    eligible = sorted(frame.loc[frame["eligible"], "ticker"].tolist())
    frozen = {
        "track": HZ.TRACK_ID,
        "frozen": True,
        "frozen_before_training": True,
        "n_candidates": len(frame),
        "n_eligible": len(eligible),
        "eligible_tickers": eligible,
        "excluded": {str(r["ticker"]): r["exclusion_reason"]
                     for r in frame.loc[~frame["eligible"]].to_dict("records")},
        "rules": {
            "min_train_samples_10d_mh_dev_2014": min_train,
            "min_validation_samples_per_year": min_val,
            "reference_horizon": reference_horizon,
            "reference_fold": reference_fold,
            "basis": ("stock-only stationary features; no context requirement may "
                      "eliminate a stock-only sample"),
            "note": ("H=10 is the most restrictive horizon, so it establishes ONE fixed "
                     "common universe for the 1D/3D/5D/10D comparison"),
        },
        "columns": list(UNIVERSE_COLUMNS),
        "evidence_columns": list(EVIDENCE_COLUMNS),
        "universe_sha256": HZ.hash_payload(frame.to_csv(index=False)),
        "csv": str(csv_path),
    }
    atomic_json_dump(frozen, track.results_root / "supervised_universe_frozen.json")
    return frozen


def _stock_only_arrays(store, tickers: list[str]):
    from agentic_forecaster.v2.horizon_dataset import build_horizon_arrays

    return build_horizon_arrays(store, _sector_map(store), tickers=tickers,
                                use_context=False)


def _sector_map(store):
    from agentic_forecaster.v2.sectors import load_sector_map

    return load_sector_map(Path(HZ.load_track_config(CONFIG)["data"]["sector_map_csv"]))


def verify(config_path: Path) -> int:
    payload = HZ.load_track_config(config_path)
    data = payload["data"]
    final_allowed = str(payload["final_allowed_date"])
    horizons = tuple(int(h) for h in payload["horizons"])

    targets, metadata = HZ.load_target_cache(
        Path(data["horizon_target_cache"]), final_allowed_date=final_allowed)
    store = load_store(Path(data["source_store_root"]), final_allowed_date=final_allowed)
    control = HZ.assert_control_matches_store(targets, store.targets,
                                              horizon=int(payload["control_horizon"]))

    frozen = json.loads((HZ.MultiHorizonTrack().results_root
                         / "supervised_universe_frozen.json").read_text())
    frame = pd.read_csv(HZ.MultiHorizonTrack().path("universe_csv"))
    recomputed = HZ.hash_payload(frame.to_csv(index=False))
    eligible_now = sorted(frame.loc[frame["eligible"], "ticker"].tolist())
    universe_ok = (recomputed == frozen["universe_sha256"]
                   and eligible_now == frozen["eligible_tickers"])

    report = {
        "target_cache": str(HZ.target_cache_root()),
        "target_schema_sha256": metadata.get("target_schema_sha256"),
        "n_target_rows": len(targets),
        "horizons_present": sorted(int(h) for h in targets["horizon"].unique()),
        "control_identity_check": control,
        "max_target_end_date": str(pd.Timestamp(targets["target_end_date"].max()).date()),
        "universe_csv": str(HZ.MultiHorizonTrack().path("universe_csv")),
        "n_eligible": len(eligible_now),
        "universe_sha256": recomputed,
        "universe_verified": bool(universe_ok),
        "max_horizon": max(horizons),
    }
    print(json.dumps(report, indent=2))
    return 0 if (universe_ok and control["identical"]) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    setup_logging()
    if args.verify:
        return verify(args.config)
    print(json.dumps(build_targets(args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())