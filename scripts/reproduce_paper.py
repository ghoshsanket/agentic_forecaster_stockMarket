#!/usr/bin/env python3
"""Reproduce the paper end-to-end.

This is a THIN WRAPPER around the same implementation used by::

    python -m agentic_forecaster reproduce-paper --config configs/paper.yaml

so both entry points execute identical logic.

Usage:
    python scripts/reproduce_paper.py --config configs/paper.yaml \
        [--device cpu] [--tickers RELIANCE,TCS] [--export-final-results]

Heavy runtime artefacts are written under $AGENTIC_OUTPUT_ROOT.  With
--export-final-results the canonical final result set is copied into the
repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentic_forecaster.config import load_config
from agentic_forecaster.orchestration.walk_forward import run_walk_forward
from agentic_forecaster.utils import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", default=None,
                        help="Device: auto, cpu, cuda, cuda:0")
    parser.add_argument("--tickers", default=None, help="Comma-separated ticker subset")
    parser.add_argument("--run-id", default=None, help="Explicit run id")
    parser.add_argument("--export-final-results", action="store_true",
                        help="Export the canonical final result set into the repo")
    args = parser.parse_args()
    setup_logging()

    config = load_config(args.config)
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    result = run_walk_forward(
        config, device=args.device, run_id=args.run_id, tickers=tickers
    )
    print("\n=== Reproduction complete ===")
    print(f"Run id  : {result['run_id']}")
    print(f"Run dir : {result['run_dir']}")
    print(f"Runs    : {result['n_ticker_fold_runs']} ticker/fold models")
    print("Aggregate:")
    for key, value in result["aggregate"].items():
        if key != "policy":
            print(f"  {key}: {value}")
    print("Cross-sectional Precision@3:")
    for row in result["precision_at_3"]:
        print(f"  {row}")

    if args.export_final_results:
        from agentic_forecaster.config import get_env_roots
        from agentic_forecaster.packaging import export_final_artifacts

        copied = export_final_artifacts(get_env_roots(), run_dir=Path(result["run_dir"]))
        print(f"\nExported {len(copied)} artefact(s) into the repository:")
        for item in copied:
            print(f"  + {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
