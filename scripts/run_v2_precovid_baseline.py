#!/usr/bin/env python3
"""PRECOVID_LOGISTIC_BASELINE: the cheap linear ceiling check.

Fits a regularised logistic regression on the SAME V2 origin-row features the
sequence models see (current-origin stationary stock features plus, for the
contextual configuration, current-origin context features), TRAIN only, and
scores the PRE-COVID validation fold only.

It answers one question: do these engineered features expose ANY simple linear
directional signal in 2017 / 2018?  It is a diagnostic ceiling, not a competitor
and not a replacement for the sequence model.

Preprocessing is fitted on TRAIN rows only.  No 2019 access, nothing after
2019-12-31.

Usage::

    uv run python scripts/run_v2_precovid_baseline.py
    uv run python scripts/run_v2_precovid_baseline.py --no-context
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.utils import atomic_json_dump, ensure_dir, setup_logging
from agentic_forecaster.v2 import precovid as pc
from agentic_forecaster.v2.baseline import BASELINE_LABEL, run_logistic_baseline
from agentic_forecaster.v2.dataset import split_masks
from agentic_forecaster.v2.experiment import (
    assemble_run_data,
    load_run_config,
)
from agentic_forecaster.v2.store import store_fingerprints

CONFIG = REPO_ROOT / "configs" / "v2" / "pre_covid" / "v2_c_contextual.yaml"


def run(config_path: Path, folds: list[str], use_context: bool) -> dict:
    results: dict[str, dict] = {}
    for fold in folds:
        config = load_run_config(config_path, fold=fold)
        if config.pre_covid_mode and "LOCKBOX" in fold.upper():
            raise AssertionError(
                "the logistic ceiling check is a DEVELOPMENT diagnostic and must never "
                "score the 2019 lockbox")
        data = assemble_run_data(config)
        masks = split_masks(data.samples, config.window)
        result = run_logistic_baseline(data.samples, fold=fold,
                                       train_mask=masks["train"],
                                       val_mask=masks["val"],
                                       use_context=use_context)
        results[fold] = result.to_dict()

    def mean(key: str) -> float | None:
        values = [results[f][key] for f in folds if results[f].get(key) is not None]
        return statistics.fmean(values) if values else None

    beats = [results[f]["accuracy_macro_ticker"] > results[f]["train_majority_baseline"]
             for f in folds]
    payload = {
        "label": BASELINE_LABEL,
        "experiment_regime": pc.EXPERIMENT_REGIME,
        "survivorship_bias_label": pc.BIAS_LABEL,
        "purpose": ("diagnostic ceiling: do the V2 features expose any simple linear "
                    "directional signal in 2017/2018? Not a competitor to the sequence "
                    "model"),
        "context_features_used": bool(use_context),
        "folds": results,
        "mean": {
            "accuracy_macro_ticker": mean("accuracy_macro_ticker"),
            "accuracy_micro": mean("accuracy_micro"),
            "f1": mean("f1"),
            "balanced_accuracy": mean("balanced_accuracy"),
            "roc_auc": mean("roc_auc"),
            "brier": mean("brier"),
            "ece": mean("ece"),
            "train_majority_baseline": mean("train_majority_baseline"),
            "delta_vs_train_majority_macro": mean("delta_vs_train_majority_macro"),
            "precision_at_3_up": mean("precision_at_3_up"),
            "precision_at_3_down": mean("precision_at_3_down"),
        },
        "beats_baseline_both_folds": bool(all(beats)) if beats else None,
        "n_supervised_tickers": len(config.supervised_tickers),
        "data_access": {
            "final_allowed_date": config.final_allowed_date,
            "folds_scored": folds,
            "lockbox_scored": False,
        },
        "store": store_fingerprints(config.processed_root),
    }
    track = pc.PreCovidTrack()
    ensure_dir(track.results_root)
    atomic_json_dump(payload, track.path("logistic"))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--folds", nargs="+", default=list(pc.DEV_FOLDS))
    parser.add_argument("--no-context", action="store_true",
                        help="stock features only (no context columns)")
    args = parser.parse_args(argv)
    setup_logging()
    payload = run(args.config, [f.upper() for f in args.folds],
                  use_context=not args.no_context)
    print(json.dumps({"label": payload["label"], "mean": payload["mean"],
                      "folds": {f: {k: v for k, v in b.items()
                                    if k in ("accuracy_macro_ticker", "accuracy_micro",
                                             "roc_auc", "brier",
                                             "train_majority_baseline")}
                                for f, b in payload["folds"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())