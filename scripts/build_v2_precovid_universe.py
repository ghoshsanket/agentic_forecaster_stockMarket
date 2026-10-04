#!/usr/bin/env python3
"""Derive and FREEZE the PRE-COVID eligible supervised universe.

The count is DERIVED, never guessed.  A security is supervised only if it passes
EVERY eligibility rule (see ``v2/precovid.py``):

    A. sufficient TRAIN history by 2016-12-31
    B. >= 1000 usable supervised TRAIN samples in PRECOVID_DEV_A
    C. >= 180 usable samples in calendar 2017
    D. >= 180 usable samples in calendar 2018
    E. >= 180 usable samples in calendar 2019
    F. all required stationary features generated causally

Context population stays separate: market / sector / cross-sectional context is
built from EVERY security of the reconstructed fixed universe that has a valid
bar on date ``t``, so a later listing contributes from the day it genuinely
exists and is never backfilled.

Outputs (frozen BEFORE any PRE-COVID variant is trained):

    results/v2/pre_covid/pre_covid_supervised_universe.csv
    configs/v2/pre_covid/supervised_universe.yaml

Usage::

    uv run python scripts/build_v2_precovid_universe.py
    uv run python scripts/build_v2_precovid_universe.py --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.config import load_config
from agentic_forecaster.utils import atomic_json_dump, ensure_dir, setup_logging
from agentic_forecaster.v2 import precovid as pc
from agentic_forecaster.v2.dataset import (
    build_feature_arrays,
    build_sample_table,
)
from agentic_forecaster.v2.experiment import assemble_context_frame
from agentic_forecaster.v2.firewall import assert_pre_covid_dates
from agentic_forecaster.v2.sectors import load_sector_map
from agentic_forecaster.v2.store import (
    assert_store_within_pre_covid,
    load_store,
    store_fingerprints,
)

CONFIG = REPO_ROOT / "configs" / "v2" / "pre_covid" / "base.yaml"


def build(config_path: Path) -> dict:
    """Derive eligibility for the whole reconstructed fixed universe."""
    payload = load_config(config_path)
    data = payload["data"]
    final_allowed = str(payload["final_allowed_date"])
    processed_root = Path(data["store_root"])

    store = load_store(processed_root, final_allowed_date=final_allowed)
    regime_check = assert_store_within_pre_covid(store.metadata,
                                                 final_allowed_date=final_allowed)
    assert_pre_covid_dates(feature_dates=[regime_check["store_last_feature_date"]],
                           target_dates=[regime_check["store_last_target_date"]],
                           final_allowed_date=final_allowed,
                           where="precovid universe derivation")

    sector_map = load_sector_map(Path(data["sector_map_csv"]))
    sector_of = sector_map.sector_series()

    # Every security of the reconstructed fixed universe that has ANY usable
    # history: this is the candidate pool for SUPERVISION.  Context uses the same
    # store tables and therefore the same, possibly larger, population.
    candidates = sorted(store.stock["ticker"].unique().tolist())
    arrays = build_feature_arrays(store.stock, assemble_context_frame(store), sector_map,
                                 tickers=candidates, use_context=True)
    samples = build_sample_table(
        arrays, store.targets,
        sequence_length=int(data["sequence_length"]),
        require_targets=["y_direction", "y_return", "y_rank"],
        max_date=final_allowed, final_allowed_date=final_allowed)

    raw_first = {
        str(ticker): str(pd_first(store.stock, ticker)) for ticker in candidates
    }
    universe = pc.derive_supervised_universe(
        samples.frame, raw_first_dates=raw_first, sector_of=sector_of,
        candidates=candidates)

    track = pc.PreCovidTrack()
    ensure_dir(track.results_root)
    ensure_dir(PC_YAML.parent)
    frozen = pc.write_universe(
        universe, csv_path=track.path("universe_csv"), yaml_path=PC_YAML,
        store_sha256=store_fingerprints(processed_root).get("store_sha256"))
    summary = {
        "experiment_regime": pc.EXPERIMENT_REGIME,
        "survivorship_bias_label": pc.BIAS_LABEL,
        "universe_phrase": pc.UNIVERSE_PHRASE,
        "final_allowed_date": final_allowed,
        "store": regime_check | store_fingerprints(processed_root),
        "cohort": pc.cohort_summary(universe),
        "rules": pc.EligibilityRules().to_dict(),
        "universe_sha256": frozen["universe_sha256"],
        "candidates_considered": len(candidates),
        "context_population_note": (
            "context uses every security of the reconstructed fixed universe with a "
            "valid bar on date t; supervision uses only the eligible list"),
        "outputs": {
            "universe_csv": str(track.path("universe_csv")),
            "universe_yaml": str(PC_YAML),
        },
    }
    atomic_json_dump(summary, track.results_root / "precovid_universe_summary.json")
    return summary


def pd_first(stock_frame, ticker: str):
    import pandas as pd

    return pd.Timestamp(stock_frame.loc[stock_frame["ticker"] == ticker, "date"].min())


PC_YAML = REPO_ROOT / "configs" / "v2" / "pre_covid" / "supervised_universe.yaml"


def verify() -> int:
    frozen = pc.load_universe(PC_YAML)
    import pandas as pd

    frame = pd.read_csv(pc.PreCovidTrack().path("universe_csv"))
    recomputed = pc.universe_hash(frame)
    ok = (recomputed == frozen["universe_sha256"]
          and pc.eligible_tickers(frame) == frozen["eligible_tickers"])
    print(json.dumps({
        "frozen_universe_yaml": str(PC_YAML),
        "universe_csv": str(pc.PreCovidTrack().path("universe_csv")),
        "n_eligible": len(frozen["eligible_tickers"]),
        "universe_sha256": recomputed,
        "frozen_sha256": frozen["universe_sha256"],
        "verified": bool(ok),
    }, indent=2))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    setup_logging()
    if args.verify:
        return verify()
    summary = build(args.config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())