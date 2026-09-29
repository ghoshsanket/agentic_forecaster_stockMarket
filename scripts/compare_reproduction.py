#!/usr/bin/env python3
"""Compare a reproduction run against the published paper reference.

Reports, per fold and aggregate:

* the paper's published metrics
* the reproduced metrics
* the absolute gap

IMPORTANT: this comparison exists to be run AFTER the final frozen test. Running
it during search would turn the published numbers into a tuning objective, which
is exactly what the test-set firewall prevents. The script therefore refuses to
produce a ranked "distance to the paper" score when search mode is active.

Published reference (docs/PAPER_TRACEABILITY.md / paper_reference.json):
accuracy 0.815, F1 0.802, Brier 0.104, ROC-AUC 0.861, precision_at_3 0.560.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

PAPER_REFERENCE = {
    "accuracy": 0.815,
    "f1": 0.802,
    "brier": 0.104,
    "roc_auc": 0.861,
    "precision_at_3": 0.560,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="directory of a finished run")
    parser.add_argument("--metrics-csv", default=None,
                        help="per-ticker metrics CSV (defaults to <run-dir>/ticker_metrics.csv)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--search-mode", action="store_true",
                        help="refuse to rank against the published numbers")
    args = parser.parse_args()

    from agentic_forecaster.recovery.firewall import firewall_guard

    if args.search_mode:
        with firewall_guard(True):
            print("Search mode: ranking against the published test numbers is "
                  "BLOCKED. Comparing to the paper during search would tune on the "
                  "answer. Re-run without --search-mode only after the final test.")

    run_dir = Path(args.run_dir)
    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else run_dir / "ticker_metrics.csv"
    if not metrics_csv.is_file():
        print(f"metrics CSV not found: {metrics_csv}")
        return 1

    df = pd.read_csv(metrics_csv)
    numeric = [c for c in df.columns if c in PAPER_REFERENCE]
    if not numeric:
        print(f"no recognised metric columns in {metrics_csv}; found {list(df.columns)}")
        return 1

    summary = {m: float(df[m].mean()) for m in numeric}
    comparison = {
        "run_dir": str(run_dir),
        "metrics_csv": str(metrics_csv),
        "n_rows": len(df),
        "paper_reference": {m: PAPER_REFERENCE[m] for m in numeric},
        "reproduced": summary,
        "absolute_gap": {m: round(summary[m] - PAPER_REFERENCE[m], 6) for m in numeric},
        "note": ("Brier is mean((p_up - y)^2). A negative gap on Brier means the "
                 "reproduction is BETTER calibrated than the paper; for the other "
                 "metrics a positive gap means better."),
    }
    text = json.dumps(comparison, indent=2)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f"wrote {p}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
