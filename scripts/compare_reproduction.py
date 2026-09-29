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

Published reference values come from
``agentic_forecaster.paper_reference.PAPER_REFERENCE`` - the single canonical
source. Table II reports no ROC-AUC, so none is invented here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# The publication does NOT publish ROC-AUC in Table II, so there is deliberately
# no roc_auc reference. Metrics come from the single canonical source.
from agentic_forecaster.paper_reference import PAPER_REFERENCE

#: The model family to compare against. The paper reports four families.
DEFAULT_FAMILY = "attention_lstm_calibrated"

#: Metrics present in Table II. No ROC-AUC is included, deliberately.
COMPARABLE = ("accuracy", "f1", "brier", "precision_at_3_up", "precision_at_3_down")


def _family_metrics(family: str) -> dict:
    if family not in PAPER_REFERENCE:
        raise KeyError(
            f"Unknown paper model family {family!r}. "
            f"Available: {sorted(PAPER_REFERENCE)}"
        )
    return PAPER_REFERENCE[family]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="directory of a finished run")
    parser.add_argument("--metrics-csv", default=None,
                        help="per-ticker metrics CSV (defaults to <run-dir>/ticker_metrics.csv)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--search-mode", action="store_true",
                        help="REFUSE to compare against the published numbers")
    parser.add_argument("--family", default=DEFAULT_FAMILY,
                        help="paper model family to compare against")
    args = parser.parse_args()

    from agentic_forecaster.recovery.firewall import firewall_guard

    if args.search_mode:
        # REFUSE, and do it BEFORE loading any metrics. Comparing to the paper
        # during search would tune on the answer.
        with firewall_guard(True):
            sys.stderr.write(
                "REFUSED: --search-mode is active.\n"
                "Comparing a run against the published test numbers during model "
                "search would tune on the answer, which is exactly what the "
                "test-set firewall prevents. No metrics were loaded and no "
                "gap-to-paper was computed.\n"
                "Re-run WITHOUT --search-mode only after the final frozen test.\n")
        return 2

    family = _family_metrics(args.family)
    run_dir = Path(args.run_dir)
    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else run_dir / "ticker_metrics.csv"
    if not metrics_csv.is_file():
        print(f"metrics CSV not found: {metrics_csv}")
        return 1

    df = pd.read_csv(metrics_csv)
    # The pipeline emits brier_calibrated / brier_raw; the paper calls it
    # "brier". Map pipeline column names onto the paper's metric names.
    aliases = {
        "accuracy": "accuracy",
        "f1": "f1",
        "brier": "brier",
        "brier_calibrated": "brier",
        "precision_at_3_up": "precision_at_3_up",
        "precision_at_3_down": "precision_at_3_down",
    }
    source_col: dict[str, str] = {}
    for col in df.columns:
        metric = aliases.get(col)
        if metric and metric in COMPARABLE and metric not in source_col:
            source_col[metric] = col
    numeric = [m for m in COMPARABLE if m in source_col]
    if not numeric:
        print(f"no recognised metric columns in {metrics_csv}; found {list(df.columns)}")
        return 1

    summary = {m: float(df[source_col[m]].mean()) for m in numeric}
    comparison = {
        "run_dir": str(run_dir),
        "metrics_csv": str(metrics_csv),
        "n_rows": len(df),
        "paper_family": args.family,
        "paper_reference": {m: family[m] for m in numeric},
        "reproduced": summary,
        "absolute_gap": {m: round(summary[m] - family[m], 6) for m in numeric},
        "roc_auc_note": ("The publication reports no ROC-AUC in Table II, so no "
                         "ROC-AUC reference exists and none is compared."),
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
