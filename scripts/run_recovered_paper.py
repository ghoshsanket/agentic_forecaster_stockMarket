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
    parser.add_argument("--out-dir", default=None,
                        help="run output directory (default: "
                             "$AGENTIC_OUTPUT_ROOT/reproduction_recovery/final)")
    parser.add_argument("--export-final-results", action="store_true",
                        help="copy final artefacts into the repository; off by "
                             "default so a previous canonical paper reproduction "
                             "is not overwritten")
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
        "search_folds_covered": manifest.get("search_folds_covered"),
        "selection_basis": manifest["selection_basis"],
        "folds": [f["fold"] for f in PAPER_FOLDS],
        "test_years": [2022, 2023],
        "run_id": args.run_id,
    }, indent=2))

    # ---- actually execute the two paper folds ----
    # run_walk_forward is the SAME implementation the paper reproduction uses,
    # so fold_0 (train 2016-2020 / val 2021 / test 2022) and fold_1
    # (train 2016-2021 / val 2022 / test 2023) are retrained per ticker exactly
    # as in the publication's protocol.
    from agentic_forecaster.config import load_config
    from agentic_forecaster.orchestration.walk_forward import run_walk_forward

    config = load_config(cfg)

    # Default output lives under the recovery tree so a previous canonical paper
    # reproduction is never overwritten unless explicitly requested.
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        from agentic_forecaster.config import get_env_roots
        out_dir = Path(get_env_roots()["AGENTIC_OUTPUT_ROOT"]) / "reproduction_recovery" / "final"
    config.setdefault("experiment", {})["output_dir"] = str(out_dir)
    print(f"\nExecuting walk-forward -> {out_dir}")

    result = run_walk_forward(config, device=args.device, run_id=args.run_id)
    print(json.dumps(result, indent=2, default=str))

    if args.export_final_results:
        from agentic_forecaster.config import get_env_roots
        from agentic_forecaster.packaging import export_final_artifacts
        copied = export_final_artifacts(get_env_roots(), run_dir=Path(result["run_dir"]))
        print(f"\nExported {len(copied)} artefact(s) into the repository:")
        for item in copied:
            print(f"  + {item}")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2, default=str))
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
