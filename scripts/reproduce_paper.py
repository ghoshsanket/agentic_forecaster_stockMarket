#!/usr/bin/env python3
"""Reproduce the paper end-to-end.

Usage:
    python scripts/reproduce_paper.py --config configs/paper.yaml \
        [--export-final-results]

Heavy runtime artefacts are written under ``$AGENTIC_OUTPUT_ROOT`` and
``$AGENTIC_MODEL_ROOT`` (Category B).  With ``--export-final-results`` the
canonical final result set is exported into the repository (Category A).
"""

from __future__ import annotations

import argparse

from agentic_forecaster.config import load_config
from agentic_forecaster.orchestration import Pipeline
from agentic_forecaster.utils import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--export-final-results", action="store_true")
    args = parser.parse_args()
    setup_logging()

    config = load_config(args.config)
    result = Pipeline(config).run()

    print("\n=== Reproduction complete ===")
    print(f"Run dir : {result.run_dir}")
    print("Metrics :")
    for model_name, metrics in result.metrics.items():
        print(f"  {model_name}: {metrics}")

    if args.export_final_results:
        from agentic_forecaster.commands.package_submission import main as pkg_main
        pkg_main([])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
