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

#: P0/P1/P2 are identified from the experiment's resolved config, never guessed.
PILOT_LABELS = {
    "P0": {"training_length": "T3", "feature_family": "F1"},
    "P1": {"training_length": "T100", "feature_family": "F1"},
    "P2": {"training_length": "T100", "feature_family": "F2"},
}
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
        for name, spec in PILOT_LABELS.items():
            if (resolved.get("max_epochs") == (3 if name == "P0" else 100)
                    and spec["feature_family"] == resolved.get("feature_family")):
                label = name
                break
        if label is None:
            continue  # not a pilot configuration
        if m.get("test_evaluated") is not False:
            continue
        if m.get("test_rows", 0) != 0:
            continue
        out.append({"label": name_label(label), "dir": d, "manifest": m,
                    "aggregate": a, "resolved": resolved})
    return out


def name_label(label: str) -> str:
    return {"P0": "P0", "P1": "P1", "P2": "P2"}.get(label, label)


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


def diagnostics(summary: dict) -> dict:
    """Training-length (P0 vs P1) and feature (P1 vs P2) diagnostics."""
    d: dict = {}
    if "P0" in summary and "P1" in summary:
        a, b = summary["P0"]["cross_fold"], summary["P1"]["cross_fold"]
        d["P0_vs_P1"] = {
            "accuracy_delta": b["mean_accuracy"] - a["mean_accuracy"],
            "f1_delta": b["mean_f1"] - a["mean_f1"],
            "brier_delta": b["mean_brier_raw"] - a["mean_brier_raw"],
            "beating_majority_delta_pct": (b["beating_majority_pct"]
                                           - a["beating_majority_pct"]),
            "P1_best_epoch": summary["P1"].get("best_epoch"),
        }
    if "P1" in summary and "P2" in summary:
        a, b = summary["P1"]["cross_fold"], summary["P2"]["cross_fold"]
        d["P1_vs_P2"] = {
            "accuracy_delta": b["mean_accuracy"] - a["mean_accuracy"],
            "f1_delta": b["mean_f1"] - a["mean_f1"],
            "brier_delta": b["mean_brier_raw"] - a["mean_brier_raw"],
            "beating_majority_delta_pct": (b["beating_majority_pct"]
                                           - a["beating_majority_pct"]),
        }
    return d


def signal_gate(summary: dict) -> dict:
    """Is a broad Stage-B/C search scientifically worthwhile?

    This is NOT a target-to-paper gate. It asks only whether pre-2022
    validation shows signal worth chasing: improvement over the majority
    baseline and/or better F1, stability across tickers and folds, and Brier
    better than the trivial ~0.25 behaviour of an uninformative constant-0.5
    forecast. The published 0.815 accuracy is deliberately NOT used.
    """
    best = None
    for label in ("P1", "P2"):
        if label in summary and (
            best is None
            or summary[label]["cross_fold"]["mean_accuracy"]
            > summary[best]["cross_fold"]["mean_accuracy"]
        ):
            best = label
    if best is None:
        return {"verdict": "ABSENT", "reason": "no pilot configuration produced results",
                "proceed_to_stage_b": False}
    c = summary[best]["cross_fold"]
    over_majority = c["mean_accuracy"] - c["mean_majority_accuracy"]
    beats_majority = c["beating_majority_pct"] >= 55.0
    f1_useful = c["mean_f1"] >= 0.52
    brier_useful = c["mean_brier_raw"] < 0.25
    stable = c["beating_majority_pct"] > 0
    promising = (over_majority > 0.005 and (beats_majority or f1_useful)) and brier_useful
    if promising and stable:
        verdict = "PROMISING"
    elif over_majority > 0 or f1_useful:
        verdict = "WEAK"
    else:
        verdict = "ABSENT"
    return {
        "best_configuration": best,
        "verdict": verdict,
        "mean_accuracy": c["mean_accuracy"],
        "mean_majority_accuracy": c["mean_majority_accuracy"],
        "accuracy_over_majority": over_majority,
        "mean_f1": c["mean_f1"],
        "mean_brier_raw": c["mean_brier_raw"],
        "beating_majority_pct": c["beating_majority_pct"],
        "criteria": {"beats_majority_pct>=55": beats_majority,
                     "f1>=0.52": f1_useful, "brier<0.25": brier_useful,
                     "stable_across_tickers": stable},
        "proceed_to_stage_b": bool(promising and stable),
        "note": ("Judged on pre-2022 validation only. Deliberately NOT compared "
                 "to the paper's published test metrics."),
    }


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
        "diagnostics": diagnostics(summary),
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
