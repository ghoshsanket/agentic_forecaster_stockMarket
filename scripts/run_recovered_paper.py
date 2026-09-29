#!/usr/bin/env python3
"""Run the FINAL paper walk-forward test on the FROZEN configuration.

This is the ONLY script that may score 2022 and 2023, and it refuses to run
unless ALL of the following hold:

  1. results/reproduction_recovery/frozen_config_manifest.json exists
  2. configs/recovered_paper.yaml hashes to the value in that manifest
  3. FINAL_TEST=1 is set explicitly in the environment

Folds executed (the publication's exact protocol):

  fold_0: train 2016-2020, validation 2021, test 2022
  fold_1: train 2016-2021, validation 2022, test 2023

    FINAL_TEST=1 uv run python scripts/run_recovered_paper.py \\
        --config configs/recovered_paper.yaml \\
        --device auto \\
        --run-id recovered_paper_final
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
    parser.add_argument("--config", default="configs/recovered_paper.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-id", default="recovered_paper_final")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from agentic_forecaster.recovery import freeze
    from agentic_forecaster.recovery.folds import PAPER_FOLDS

    cfg = Path(args.config)
    if not cfg.is_absolute():
        cfg = REPO_ROOT / cfg

    # The gate. Raises unless frozen, hash-matched, and explicitly authorised.
    manifest = freeze.assert_final_test_allowed(cfg)

    print(json.dumps({
        "frozen_config": manifest["config_path"],
        "config_sha256": manifest["config_sha256"],
        "git_commit": manifest["git_commit"],
        "dataset_variant": manifest["dataset_variant"],
        "universe_id": manifest["universe_id"],
        "selection_basis": manifest["selection_basis"],
        "folds": [f["fold"] for f in PAPER_FOLDS],
        "test_years": [2022, 2023],
        "run_id": args.run_id,
    }, indent=2))
    print("\nFinal test authorised. Executing the two paper walk-forward folds...")
    raise SystemExit(
        "Walk-forward execution is intentionally not wired into this script yet.\n"
        "Once wired it will call run_walk_forward() from "
        "agentic_forecaster.orchestration.walk_forward, which retrains every "
        "ticker per fold. The gate above already refuses to reach this point "
        "without a valid freeze and FINAL_TEST=1."
    )


if __name__ == "__main__":
    raise SystemExit(main())
