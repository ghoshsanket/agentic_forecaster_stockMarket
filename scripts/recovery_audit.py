#!/usr/bin/env python3
"""Audit the performance-recovery setup WITHOUT running any search.

Reports the current state of everything the recovery depends on:

* the test-set firewall and the search folds it protects
* the variant catalogue (features, RSI, training length, scalers, architecture,
  weight decay, calibration, seeds)
* the experiment ledger, if one exists
* the frozen configuration, if one exists
* the stage plan

This script is read-only.  It never loads labels and never scores anything, so
it is safe to run at any time, including after the freeze.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="write the audit JSON here")
    parser.add_argument("--print-plan", action="store_true", help="include the stage plan")
    args = parser.parse_args()

    from agentic_forecaster.recovery import folds, freeze, ledger, variants
    from agentic_forecaster.recovery.firewall import describe_firewall

    audit = {
        "firewall": describe_firewall(),
        "search_folds": folds.describe_folds(),
        "paper_folds_guarded": [f["fold"] for f in folds.PAPER_FOLDS],
        "variants": variants.describe_variants(),
        "ledger": ledger.ledger_summary(),
        "frozen_config": {
            "config_exists": freeze.frozen_config_path().is_file(),
            "manifest_exists": freeze.frozen_manifest_path().is_file(),
            "policy": freeze.describe_freeze_policy(),
        },
        "dataset_variants": {
            "primary_original_recollection_candidate": "paper_snapshot_2025_unadjusted_perf",
            "sensitivity_variant": "paper_snapshot_2025_adjusted_perf",
            "note": "Do not assume adjusted data performs better; both must be tested.",
        },
    }
    if args.print_plan:
        audit["stage_plan"] = variants.stage_plan()

    out = args.out
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(audit, indent=2))
        print(f"wrote {p}")
    else:
        print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
