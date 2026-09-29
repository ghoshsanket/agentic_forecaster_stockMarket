#!/usr/bin/env python3
"""Summarise the Stage-A PILOT from its own artifacts.

Reads ONLY the pilot experiment artifacts written under
``$AGENTIC_OUTPUT_ROOT/reproduction_recovery/EXP-*``.

Deliberately does NOT:
  * import PAPER_REFERENCE, or read any published paper metric;
  * read 2022 or 2023 metrics;
  * read ``results/paper_reproduction/``.

The whole point is that the pilot is judged on PRE-2022 validation only. If this
script ever compared against the published test numbers it would turn the pilot
into a tuning target, which is what the firewall exists to prevent.

Usage:
    python scripts/summarize_stage_a_pilot.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

#: Series are identified from each experiment's RESOLVED CONFIG, never guessed
#: from a directory name. `max_epochs` is the authoritative discriminator,
#: because the T3/T10/T100 catalog keys changed names when 10 epochs became
#: author-confirmed while the underlying runs were left untouched.
#:
#: 10 epochs is AUTHOR-CONFIRMED and is the PRIMARY reconstruction candidate.
#: T3 and T100 are diagnostics: T3 was a reconstruction shortcut, T100 was a
#: recovery diagnostic. Neither may be selected on performance alone.
SERIES = {
    "T3-F1":   {"max_epochs": 3,   "feature_family": "F1", "role": "diagnostic",
                "label": "T3_RECONSTRUCTION_SHORTCUT"},
    "T10-F1":  {"max_epochs": 10,  "feature_family": "F1", "role": "PRIMARY",
                "label": "T10_AUTHOR_CONFIRMED"},
    "T10-F2":  {"max_epochs": 10,  "feature_family": "F2", "role": "PRIMARY",
                "label": "T10_AUTHOR_CONFIRMED"},
    "T100-F1": {"max_epochs": 100, "feature_family": "F1", "role": "diagnostic",
                "label": "T100_DIAGNOSTIC"},
    "T100-F2": {"max_epochs": 100, "feature_family": "F2", "role": "diagnostic",
                "label": "T100_DIAGNOSTIC"},
}
#: Report order: the author-confirmed schedule first, then diagnostics.
SERIES_ORDER = ["T10-F1", "T10-F2", "T3-F1", "T100-F1", "T100-F2"]
AUTHOR_CONFIRMED_SERIES = ("T10-F1", "T10-F2")
FOLDS = ["SEARCH_FOLD_A", "SEARCH_FOLD_B", "SEARCH_FOLD_C"]


def pilot_experiments(root: Path) -> list[dict]:
    """Load pilot experiment directories, tagging each with its P0/P1/P2 label."""
    out: list[dict] = []
    for d in sorted(root.glob("EXP-*")):
        man = d / "manifest.json"
        agg = d / "aggregate_validation_metrics.json"
        if not (man.is_file() and agg.is_file()):
            continue
        m = json.loads(man.read_text())
        a = json.loads(agg.read_text())
        resolved = a.get("resolved", {})
        label = None
        for name, spec in SERIES.items():
            if (resolved.get("max_epochs") == spec["max_epochs"]
                    and resolved.get("feature_family") == spec["feature_family"]):
                label = name
                break
        if label is None:
            continue  # not a pilot configuration
        if m.get("test_evaluated") is not False:
            continue
        if m.get("test_rows", 0) != 0:
            continue
        out.append({"label": label, "dir": d, "manifest": m,
                    "aggregate": a, "resolved": resolved})
    return out


def per_case(experiment: dict) -> list[dict]:
    """One row per (config, fold, ticker) from ticker_metrics.csv."""
    f = experiment["dir"] / "ticker_metrics.csv"
    if not f.is_file():
        return []
    df = pd.read_csv(f)
    df["config"] = experiment["label"]
    df["fold"] = experiment["manifest"]["search_fold"]
    return df.to_dict("records")


def summarise(cases: list[dict], experiments: list[dict]) -> dict:
    if not cases:
        return {}
    df = pd.DataFrame(cases)
    out: dict = {}
    for label, grp in df.groupby("config"):
        per_fold = {}
        for fold, fg in grp.groupby("fold"):
            per_fold[fold] = {
                "n_tickers": len(fg),
                "mean_accuracy": float(fg["accuracy"].mean()),
                "mean_f1": float(fg["f1"].mean()),
                "mean_brier_raw": float(fg["brier"].mean()),
                "mean_ece": float(fg["ece"].mean()),
                "mean_majority_accuracy": float(fg["majority_accuracy"].mean()),
                "beating_majority_pct": float(
                    (fg["accuracy"] > fg["majority_accuracy"]).mean() * 100),
                "tickers": sorted(fg["ticker"].tolist()),
            }
        cross = {
            "mean_accuracy": float(grp["accuracy"].mean()),
            "std_accuracy": float(grp["accuracy"].std(ddof=0)),
            "mean_f1": float(grp["f1"].mean()),
            "std_f1": float(grp["f1"].std(ddof=0)),
            "mean_brier_raw": float(grp["brier"].mean()),
            "std_brier_raw": float(grp["brier"].std(ddof=0)),
            "mean_ece": float(grp["ece"].mean()),
            "mean_majority_accuracy": float(grp["majority_accuracy"].mean()),
            "beating_majority_pct": float(
                (grp["accuracy"] > grp["majority_accuracy"]).mean() * 100),
            "n_cases": len(grp),
        }
        # cross-sectional P@3 is computed over the 8 tickers TOGETHER, so it
        # lives in each experiment's aggregate file, not in ticker_metrics.csv.
        p3 = {}
        for e in experiments:
            if e["label"] != label:
                continue
            agg = e["aggregate"].get("aggregate", {})
            for k in ("precision_at_3_up", "precision_at_3_down"):
                v = agg.get(k)
                if v is not None and pd.notna(v):
                    p3.setdefault(k, []).append(float(v))
        cross["precision_at_3_up"] = float(np.mean(p3["precision_at_3_up"])) if "precision_at_3_up" in p3 else None
        cross["precision_at_3_down"] = float(np.mean(p3["precision_at_3_down"])) if "precision_at_3_down" in p3 else None
        out[label] = {"per_fold": per_fold, "cross_fold": cross,
                      "per_ticker": {
                          t: {"mean_accuracy": float(g["accuracy"].mean()),
                              "mean_f1": float(g["f1"].mean()),
                              "beating_majority_pct": float(
                                  (g["accuracy"] > g["majority_accuracy"]).mean() * 100)}
                          for t, g in grp.groupby("ticker")}}
        be = [v for v in grp.get("best_epoch", pd.Series(dtype=float)) if pd.notna(v)]
        if be:
            arr = np.array(be, dtype=float)
            out[label]["best_epoch"] = {
                "n": int(arr.size), "median": float(np.median(arr)),
                "mean": float(arr.mean()), "max": float(arr.max()),
                "pct_gt_3": float((arr > 3).mean() * 100),
                "pct_gt_10": float((arr > 10).mean() * 100),
                "histogram": {str(int(k)): int(v) for k, v in
                              zip(*np.unique(arr, return_counts=True))},
            }
    return out


def epoch_buckets(summary: dict, series: str) -> dict:
    """Best-epoch distribution, bucketed around the author-confirmed 10-epoch cap.

    The question this answers is whether the confirmed 10-epoch budget was
    materially different from the earlier 3-epoch shortcut. If most fits select
    a best epoch <= 3, then 3 and 10 epochs are nearly the same experiment and
    the earlier shortcut cost almost nothing.
    """
    be = summary.get(series, {}).get("best_epoch")
    if not be or not be.get("histogram"):
        return {}
    hist = {int(k): int(v) for k, v in be["histogram"].items()}
    n = sum(hist.values()) or 1
    le3 = sum(v for k, v in hist.items() if k <= 3)
    mid = sum(v for k, v in hist.items() if 4 <= k <= 6)
    hi = sum(v for k, v in hist.items() if 7 <= k <= 10)
    return {
        "counts_by_epoch": {str(k): hist[k] for k in sorted(hist)},
        "n_fits": n,
        "median_best_epoch": be.get("median"),
        "mean_best_epoch": be.get("mean"),
        "max_best_epoch": be.get("max"),
        "frac_le_3": le3 / n,
        "frac_4_to_6": mid / n,
        "frac_7_to_10": hi / n,
        "frac_gt_10": sum(v for k, v in hist.items() if k > 10) / n,
        "note": ("A best epoch above 10 is impossible under a 10-epoch cap; a "
                 "non-zero frac_gt_10 would indicate a cap that was not applied."),
    }


def signal_gate(summary: dict) -> dict:
    """PRE2022_T10_SIGNAL: is signal present at the AUTHOR-CONFIRMED 10 epochs?

    This is NOT a target-to-paper gate. It asks only whether pre-2022
    validation shows signal worth chasing at the confirmed schedule:
    improvement over the majority baseline and/or better F1, stability across
    tickers and folds, and Brier better than the trivial ~0.25 behaviour of an
    uninformative constant-0.5 forecast. The published 0.815 test accuracy is
    deliberately NOT used.

    Only the T10 series are eligible. T3 and T100 are diagnostics and must not
    drive the decision: the objective is faithful reconstruction, so a longer
    diagnostic budget winning here is a forensic clue, not a promotion.
    """
    eligible = [s for s in AUTHOR_CONFIRMED_SERIES if s in summary]
    if not eligible:
        return {"PRE2022_T10_SIGNAL": "ABSENT", "verdict": "ABSENT",
                "reason": "no author-confirmed T10 results present",
                "proceed_to_stage_b": False,
                "recommendation": "RUN_PRE2022_FORENSICS"}
    best = max(eligible, key=lambda s: summary[s]["cross_fold"]["mean_accuracy"])
    c = summary[best]["cross_fold"]
    over_majority = c["mean_accuracy"] - c["mean_majority_accuracy"]
    beats_majority = c["beating_majority_pct"] >= 55.0
    f1_useful = c["mean_f1"] >= 0.52
    brier_useful = c["mean_brier_raw"] < 0.25
    # PROMISING requires the effect to repeat, not just to average out positive.
    stable = c["beating_majority_pct"] > 0
    promising = (over_majority > 0.005 and (beats_majority or f1_useful)) and brier_useful
    if promising and stable:
        verdict = "PROMISING"
    elif over_majority > 0 or f1_useful:
        verdict = "WEAK"
    else:
        verdict = "ABSENT"
    proceed = bool(verdict == "PROMISING" and stable)
    return {
        "PRE2022_T10_SIGNAL": verdict,
        "verdict": verdict,
        "best_author_confirmed_series": best,
        "role": "AUTHOR-CONFIRMED PRIMARY (10 epochs)",
        "mean_accuracy": c["mean_accuracy"],
        "mean_majority_accuracy": c["mean_majority_accuracy"],
        "accuracy_over_majority": over_majority,
        "mean_f1": c["mean_f1"],
        "mean_brier_raw": c["mean_brier_raw"],
        "beating_majority_pct": c["beating_majority_pct"],
        "criteria": {"accuracy_over_majority>0.005": over_majority > 0.005,
                     "beats_majority_pct>=55": beats_majority,
                     "f1>=0.52": f1_useful, "brier<0.25": brier_useful,
                     "stable_across_tickers": stable},
        "proceed_to_stage_b": proceed,
        "recommendation": ("PROCEED_TO_STAGE_B" if proceed
                           else "RUN_PRE2022_FORENSICS"),
        "note": ("Judged on pre-2022 validation at the author-confirmed 10-epoch "
                 "schedule only. Deliberately NOT compared to the paper's "
                 "published test metrics. T3/T100 are diagnostics and cannot "
                 "promote the decision."),
    }


def diagnostics(summary: dict) -> dict:
    """T3 -> T10 -> T100, and the F1 -> F2 contrast at exactly 10 epochs."""
    d: dict = {}

    def cross(s):
        return summary.get(s, {}).get("cross_fold")
    pairs = {
        "T3_to_T10_F1":   ("T3-F1", "T10-F1"),
        "T10_to_T100_F1": ("T10-F1", "T100-F1"),
        "F1_to_F2_at_T10": ("T10-F1", "T10-F2"),
    }
    for name, (a_key, b_key) in pairs.items():
        a, b = cross(a_key), cross(b_key)
        if not a or not b:
            continue
        d[name] = {
            "from": a_key, "to": b_key,
            "accuracy_delta": b["mean_accuracy"] - a["mean_accuracy"],
            "f1_delta": b["mean_f1"] - a["mean_f1"],
            "brier_delta": b["mean_brier_raw"] - a["mean_brier_raw"],
            "ece_delta": b["mean_ece"] - a["mean_ece"],
            "beating_majority_delta_pct": (b["beating_majority_pct"]
                                           - a["beating_majority_pct"]),
        }
    if "T3-F1" in summary:
        d["T3_best_epoch"] = summary["T3-F1"].get("best_epoch")
    for s in AUTHOR_CONFIRMED_SERIES:
        if s in summary:
            d[f"{s}_best_epoch"] = summary[s].get("best_epoch")
    for s in ("T100-F1", "T100-F2"):
        if s in summary:
            d[f"{s}_best_epoch"] = summary[s].get("best_epoch")
    # A diagnostic win is a forensic clue, never a promotion.
    best_any = max((s for s in summary if cross(s)),
                   key=lambda s: cross(s)["mean_accuracy"], default=None)
    d["highest_accuracy_series"] = best_any
    d["selection_policy"] = (
        "The primary candidate remains T10 (AUTHOR-CONFIRMED) regardless of "
        "which series scores highest. A T100 win over T10 is evidence that some "
        "OTHER aspect of the reconstruction is unfaithful and warrants forensic "
        "diagnosis; it is not grounds to select T100.")
    return d


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None)
    parser.add_argument("--root", default=None,
                        help="pilot output root (default: "
                             "$AGENTIC_OUTPUT_ROOT/reproduction_recovery)")
    args = parser.parse_args()

    from agentic_forecaster.config import get_env_roots
    root = Path(args.root) if args.root else (
        Path(get_env_roots()["AGENTIC_OUTPUT_ROOT"]) / "reproduction_recovery")

    experiments = pilot_experiments(root)
    if not experiments:
        print(f"no pilot experiments found under {root}", file=sys.stderr)
        return 1
    cases: list[dict] = []
    for e in experiments:
        cases.extend(per_case(e))
    summary = summarise(cases, experiments)
    report = {
        "source": str(root),
        "experiment_runs": len(experiments),
        "ticker_fits": len(cases),
        "folds": FOLDS,
        "firewall": "no date >= 2022-01-01 was read or scored",
        "paper_reference_used": False,
        "summary": summary,
        "series_order": SERIES_ORDER,
        "series_roles": {k: SERIES[k]["role"] for k in SERIES},
        "diagnostics": diagnostics(summary),
        "epoch_analysis": {s: epoch_buckets(summary, s) for s in SERIES_ORDER
                           if s in summary},
        "signal_gate": signal_gate(summary),
    }
    text = json.dumps(report, indent=2, default=str)
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
