#!/usr/bin/env python3
"""Run ONE pre-2022 search experiment and append it to the experiment ledger.

Every invocation scores a SEARCH_FOLD_A/B/C validation window.  The test-set
firewall is enabled for the whole run, so 2022 and 2023 cannot be reached even
if a config mistakenly points at them.

The ledger is append-only: poor experiments are kept, because a search whose
failures are hidden cannot be audited.

Examples
--------
STAGE 0 sanity (tiny overfit + shuffled control)::

    uv run python scripts/run_reproduction_search.py --stage 0 \\
        --config configs/reproduction_search/paper_snapshot_2025_unadjusted_perf.yaml

A single training-length point on a search fold::

    uv run python scripts/run_reproduction_search.py \\
        --config configs/reproduction_search/paper_snapshot_2025_unadjusted_perf.yaml \\
        --search-fold SEARCH_FOLD_C --tickers RELIANCE,TCS,INFY \\
        --training-length T30 --feature-family F1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--search-fold", default="SEARCH_FOLD_C",
                        help="pre-2022 validation window (never paper fold 0/1)")
    parser.add_argument("--tickers", default=None, help="comma-separated subset")
    parser.add_argument("--stage", default="A", help="stage label recorded in the ledger")
    parser.add_argument("--training-length", default=None, help="T3/T30/T50/T100")
    parser.add_argument("--feature-family", default=None, help="F0/F1/F2/F3")
    parser.add_argument("--rsi-method", default=None, help="R0_rolling/R1_wilder/R2_ema")
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--scaler", default=None)
    parser.add_argument("--volume-mode", default=None, help="raw/log1p")
    parser.add_argument("--architecture", default=None, help="A0/A1/A2/A3")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--class-weighting", default=None, help="none/pos_weight")
    parser.add_argument("--calibration", default=None,
                        help="none/temperature/platt/isotonic")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and firewall-check the config, run nothing")
    args = parser.parse_args()

    from agentic_forecaster.config import load_config
    from agentic_forecaster.recovery import folds, variants
    from agentic_forecaster.recovery.firewall import firewall_guard

    base = load_config(args.config)
    # Firewall ON for the entire run.
    with firewall_guard(True):
        cfg = folds.fold_config_for(args.search_fold, base)

        tl = args.training_length or variants.DEFAULT_TRAINING_LENGTH
        train = variants.TRAINING_LENGTHS[tl]
        models = cfg.setdefault("models", {})
        attn = models.setdefault("attention_lstm", {})
        attn["max_epochs"] = train["max_epochs"]
        attn["patience"] = train["patience"]
        attn["restore_best_checkpoint"] = train["restore_best_checkpoint"]

        fam = args.feature_family or variants.DEFAULT_FEATURE_FAMILY
        rsi = args.rsi_method or variants.DEFAULT_RSI_METHOD
        indicators = list(variants.build_feature_indicators(fam, rsi))
        cfg.setdefault("features", {})["indicators"] = indicators
        cfg["features"]["use_ohlcv"] = variants.FEATURE_FAMILIES[fam]["use_ohlcv"]

        if args.lookback is not None:
            cfg["data"]["sequence_length"] = int(args.lookback)
        if args.scaler:
            cfg["data"]["scaler"] = args.scaler
        if args.volume_mode:
            cfg["data"]["volume_mode"] = args.volume_mode
        if args.architecture:
            arch = variants.ARCHITECTURES[args.architecture]
            attn["hidden_size"] = arch["hidden_size"]
            attn["num_layers"] = arch["num_layers"]
        if args.dropout is not None:
            attn["dropout"] = float(args.dropout)
        if args.weight_decay is not None:
            attn["weight_decay"] = float(args.weight_decay)
        if args.class_weighting:
            attn["class_weighting"] = args.class_weighting
        if args.calibration:
            cfg.setdefault("calibration", {})["method"] = args.calibration
        if args.seed is not None:
            cfg.setdefault("experiment", {})["seed"] = int(args.seed)

        resolved = {
            "search_fold": args.search_fold,
            "stage": args.stage,
            "training_length": tl,
            "feature_family": fam,
            "rsi_method": rsi,
            "lookback": cfg["data"].get("sequence_length"),
            "scaler": cfg["data"].get("scaler"),
            "architecture": {"hidden_size": attn.get("hidden_size"),
                             "num_layers": attn.get("num_layers"),
                             "dropout": attn.get("dropout")},
            "max_epochs": attn["max_epochs"],
            "patience": attn["patience"],
            "weight_decay": attn.get("weight_decay"),
            "class_weighting": attn.get("class_weighting", "none"),
            "calibration": cfg.get("calibration", {}).get("method"),
            "seed": cfg.get("experiment", {}).get("seed", 42),
            "n_indicators": len(indicators),
            "tickers": (args.tickers or "<all>"),
            "data_windows": {k: cfg["data"].get(k)
                             for k in ("train_start", "train_end", "val_start", "val_end")},
            "test_window": "REMOVED (search folds have no test period)",
        }

    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        print("\nDRY RUN: firewall active, no training performed.")
        return 0

    raise SystemExit(
        "Training is intentionally not wired into this script yet.\n"
        "Use --dry-run to verify config resolution and the firewall.\n"
        "Stage 0 sanity first:\n"
        "  uv run python scripts/run_reproduction_search.py --stage 0 "
        "--config <cfg> --dry-run"
    )


if __name__ == "__main__":
    raise SystemExit(main())
