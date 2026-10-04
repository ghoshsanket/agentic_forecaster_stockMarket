#!/usr/bin/env python3
"""Freeze the ONE chosen configuration, and pin its SHA-256.

Run this exactly once, after the staged search has selected a configuration on
pre-2022 validation.  It records the config hash, git commit, dataset variant
and manifest hash, universe id, validation metrics and the supporting experiment
ids.  After it succeeds, NO parameter may change: the final test run refuses to
start on a hash mismatch.

Example::

    uv run python scripts/freeze_recovered_config.py \\
        --config configs/recovered_paper.yaml \\
        --dataset-variant unadjusted \\
        --dataset-manifest \\
        $AGENTIC_DATA_ROOT/yfinance_paper_snapshot_2025_11_04/manifests/paper_snapshot_manifest.json \\
        --universe-id PAPER_FIXED_NIFTY50_2025_11_04_CANDIDATE \\
        --validation-metrics results/reproduction_recovery/best_validation_metrics.json \\
        --experiment-ids EXP-1,EXP-2
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
    parser.add_argument("--config", required=True, help="config to freeze (normally configs/recovered_paper.yaml)")
    parser.add_argument("--dataset-variant", required=True, choices=["unadjusted", "adjusted"])
    parser.add_argument("--dataset-manifest", default=None)
    parser.add_argument("--universe-id", required=True)
    parser.add_argument("--validation-metrics", default=None,
                        help="JSON file of the selected pre-2022 validation metrics")
    parser.add_argument("--experiment-ids", default="",
                        help="comma-separated ledger experiment ids")
    parser.add_argument("--notes", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from agentic_forecaster.recovery import freeze
    from agentic_forecaster.recovery.ledger import hash_file, read_ledger

    cfg = Path(args.config)
    if not cfg.is_absolute():
        cfg = REPO_ROOT / cfg

    metrics = {}
    if args.validation_metrics:
        mp = Path(args.validation_metrics)
        if not mp.is_absolute():
            mp = REPO_ROOT / mp
        metrics = json.loads(mp.read_text())

    ids = [i for i in args.experiment_ids.split(",") if i]
    known = {r["experiment_id"] for r in read_ledger()}
    unknown = [i for i in ids if known and i not in known]
    if unknown:
        print(f"WARNING: experiment ids not present in the ledger: {unknown}")

    if args.dry_run:
        print(json.dumps({
            "config": str(cfg), "config_exists": cfg.is_file(),
            "config_sha256": hash_file(cfg) if cfg.is_file() else None,
            "dataset_variant": args.dataset_variant,
            "universe_id": args.universe_id,
            "experiment_ids": ids,
            "unknown_experiment_ids": unknown,
            "would_write": str(freeze.frozen_manifest_path()),
        }, indent=2))
        return 0

    manifest = freeze.freeze_config(
        cfg, dataset_variant=args.dataset_variant,
        dataset_manifest=args.dataset_manifest,
        universe_id=args.universe_id, validation_metrics=metrics,
        supporting_experiment_ids=ids, notes=args.notes)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
