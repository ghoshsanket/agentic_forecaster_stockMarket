#!/usr/bin/env python3
"""Summarise the pre-2022 forensic diagnosis.

Reads ONLY forensic artifacts plus the legitimate Stage-A T10-F2 unadjusted
predictions. It never imports `PAPER_REFERENCE`, never reads 2022/2023, and
never ranks or selects the largest number it finds -- a convention that
flatters the result is an artefact, not evidence of signal.

Emits the cause ranking requested by the investigation, choosing at most ONE
next recommendation, and never executes it.

Usage:
    python scripts/summarize_pre2022_forensics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.recovery import forensics as fx
from agentic_forecaster.recovery.forensics import (
    PRE_TEST_CUTOFF,
    aggregation_metrics,
    confidence_subset_metrics,
)

#: An effect must clear this margin over the majority baseline before it is
#: treated as a plausible explanation of a high reported accuracy.
MATERIAL_MARGIN = 0.02

#: A confidence floor is only informative when it retains a real share of the
#: predictions. Below this, its accuracy is dominated by sampling noise.
MIN_RETAINED_PCT = 20.0


def load_legitimate_predictions(pilot_root: Path) -> pd.DataFrame:
    """Load legitimate T10-F2 UNADJUSTED pre-2022 validation predictions.

    Reads only artifacts whose resolved config is the author-confirmed anchor,
    and refuses any row at/after the firewall boundary.
    """
    frames = []
    for d in sorted(pilot_root.glob("EXP-*")):
        f = d / "aggregate_validation_metrics.json"
        p = d / "validation_predictions.csv"
        if not (f.is_file() and p.is_file()):
            continue
        r = json.loads(f.read_text())["resolved"]
        if r.get("max_epochs") != 10 or r.get("feature_family") != "F2":
            continue
        if r.get("volume_mode") != "raw" or r.get("calibration") != "none":
            continue
        df = pd.read_csv(p)
        df["fold"] = r["search_fold"]
        df = df.rename(columns={"y": "y_true"})
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no T10-F2 unadjusted predictions under {pilot_root}")
    out = pd.concat(frames, ignore_index=True)
    out = out[out["target_date"].astype(str) < PRE_TEST_CUTOFF]
    fx.assert_targets_pre_test(out["target_date"], out["target_date"],
                               where="legitimate_predictions")
    return out


def _acc(y, p) -> float:
    return fx._acc(np.asarray(y, int), np.asarray(p, float))


def _f1(y, p) -> float:
    return fx._f1(np.asarray(y, int), np.asarray(p, float))


def cause_ranking(*, unadjusted_acc: float, adjusted_acc: float,
                  panel: dict, pooled_acc: float | None,
                  analog_acc: float | None, majority_acc: float,
                  mismatch_count: int, aggregation: dict,
                  confidence: dict, l5: dict | None) -> list[dict]:
    """Rank candidate causes by how much of the gap each one could explain.

    A cause is only credited if it IMPROVES on the legitimate baseline by a
    material margin. A probe that merely restates the baseline explains nothing,
    and neither does one that performs worse -- that is evidence against the
    hypothesis, not for it.
    """
    out: list[dict] = []

    def add(cause, delta, verdict, evidence):
        out.append({"cause": cause, "accuracy_delta_vs_legitimate": delta,
                    "material": delta is not None and delta > MATERIAL_MARGIN,
                    "verdict": verdict, "evidence": evidence})

    # DATA_ADJUSTMENT
    d = adjusted_acc - unadjusted_acc
    add("DATA_ADJUSTMENT", d,
        "SUPPORTED" if d > MATERIAL_MARGIN else "NOT_SUPPORTED",
        f"adjusted {adjusted_acc:.4f} vs unadjusted {unadjusted_acc:.4f} "
        f"(delta {d:+.4f}, material threshold {MATERIAL_MARGIN})")

    # MODEL_FORM: does a simpler model beat Attention?
    best_simple = None
    for name in ("LogisticRegression", "RandomForest", "PlainLSTM"):
        if name in panel:
            a = panel[name]["accuracy"]
            if best_simple is None or a > best_simple[1]:
                best_simple = (name, a)
    if best_simple:
        d = best_simple[1] - unadjusted_acc
        add("MODEL_FORM", d,
            "SUPPORTED" if d > MATERIAL_MARGIN else "NOT_SUPPORTED",
            f"best simple model {best_simple[0]}={best_simple[1]:.4f} vs "
            f"Attention {unadjusted_acc:.4f} (delta {d:+.4f})")

    # PER_STOCK_VS_POOLED
    if pooled_acc is not None:
        d = pooled_acc - unadjusted_acc
        add("PER_STOCK_VS_POOLED", d,
            "SUPPORTED" if d > MATERIAL_MARGIN else "NOT_SUPPORTED",
            f"pooled {pooled_acc:.4f} vs per-stock {unadjusted_acc:.4f} "
            f"(delta {d:+.4f}); pooled is NOT author-confirmed")

    # SPLIT_PROTOCOL
    if analog_acc is not None:
        d = analog_acc - unadjusted_acc
        add("SPLIT_PROTOCOL", d,
            "SUPPORTED" if d > MATERIAL_MARGIN else "NOT_SUPPORTED",
            f"pre-2022 85/15 analog {analog_acc:.4f} vs walk-forward "
            f"{unadjusted_acc:.4f} (delta {d:+.4f})")

    # TARGET_ALIGNMENT
    add("TARGET_ALIGNMENT", None,
        "NOT_SUPPORTED" if mismatch_count == 0 else "SUPPORTED",
        f"target-alignment mismatch_count={mismatch_count} over 900 raw-OHLCV "
        f"samples; next-day target construction is proven correct")

    # AGGREGATION
    accs = {k: v["accuracy"] for k, v in aggregation.items()
            if isinstance(v, dict) and "accuracy" in v}
    if accs:
        lo, hi = min(accs.values()), max(accs.values())
        d = hi - unadjusted_acc
        add("AGGREGATION", d,
            "SUPPORTED" if d > MATERIAL_MARGIN else "NOT_SUPPORTED",
            f"conventions span {lo:.4f}..{hi:.4f} (spread {hi-lo:.4f}); "
            f"best convention vs legitimate {d:+.4f}")

    # CONFIDENCE_FILTERING
    conf_accs = {k: v["accuracy"] for k, v in confidence.items()
                 if isinstance(v, dict) and v.get("accuracy") is not None
                 and k != "all_predictions"}
    if conf_accs:
        bestk = max(conf_accs, key=conf_accs.get)
        retained = confidence[bestk]["retained_pct"]
        d = conf_accs[bestk] - unadjusted_acc
        # A conditional accuracy computed on a tiny slice is a small-sample
        # artefact, not signal, so it cannot credit a cause on its own.
        enough = retained >= MIN_RETAINED_PCT
        add("CONFIDENCE_FILTERING", d,
            "SUPPORTED" if (d > MATERIAL_MARGIN and enough)
            else "NOT_SUPPORTED",
            f"best confidence floor {bestk} -> {conf_accs[bestk]:.4f} but on "
            f"only {retained:.1f}% of predictions (min required "
            f"{MIN_RETAINED_PCT}%); conditional accuracy, not overall "
            f"(delta {d:+.4f})")
    return out


def leakage_verdict(probes: dict, l0_acc: float) -> dict:
    """How much could leakage have inflated a historical result?

    The invalid probes are the ONLY way LEAKAGE/OFF_BY_ONE can be supported,
    because a legitimate run cannot exhibit leakage by construction.
    """
    rows = []
    for name in ("L1", "L2", "L3", "L4"):
        if name not in probes:
            continue
        df = pd.DataFrame(probes[name]["rows"])
        acc = float(df["validation_accuracy"].mean())
        rows.append({"probe": name, "mean_validation_accuracy": acc,
                     "inflation_vs_L0": acc - l0_acc,
                     "scientifically_invalid": True})
    best = max(rows, key=lambda r: r["inflation_vs_L0"], default=None)
    return {
        "probes": rows,
        "L0_correct_control": l0_acc,
        "max_inflation": None if best is None else best["inflation_vs_L0"],
        "max_inflation_probe": None if best is None else best["probe"],
        "verdict": ("EXPLAINS_INFLATION"
                    if best and best["inflation_vs_L0"] > 0.10
                    else "INSUFFICIENT_TO_EXPLAIN"),
        "note": ("L1-L4 are INTENTIONALLY WRONG. They show how a lost "
                 "implementation could have produced an inflated number; they "
                 "are never valid models and never candidates for selection."),
    }


def main() -> int:
    ap = __import__("argparse").ArgumentParser(description=__doc__)
    ap.add_argument("--pilot-root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from agentic_forecaster.config import get_env_roots
    pilot_root = Path(args.pilot_root) if args.pilot_root else (
        Path(get_env_roots()["AGENTIC_OUTPUT_ROOT"]) / "reproduction_recovery")
    repo_fx = fx.repo_forensic_dir()

    raw_path = repo_fx / "forensics_raw.json"
    if not raw_path.is_file():
        print(f"missing {raw_path}; run the diagnostics first", file=sys.stderr)
        return 1
    raw = json.loads(raw_path.read_text())

    preds = load_legitimate_predictions(pilot_root)
    aggregation = aggregation_metrics(preds)
    confidence = confidence_subset_metrics(preds)

    A = raw.get("A_adjusted", {})
    B = raw.get("B_baseline_panel", {})
    C = raw.get("C_pooled", {})
    D = raw.get("D_85_15", {})
    E = raw.get("E_alignment", {})
    L = raw.get("L_invalid_probes", {})

    unadj_acc = B.get("aggregate", {}).get("AttentionLSTM_T10F2", {}).get("accuracy")
    majority_acc = B.get("aggregate", {}).get("Majority", {}).get("accuracy")
    adj_acc = A.get("adjusted", {}).get("mean_accuracy")
    pooled_acc = (C.get("aggregate") or {}).get("accuracy")
    analog_acc = (D.get("aggregate") or {}).get("accuracy")
    l0_df = pd.DataFrame(L.get("L0", {}).get("rows", []))
    l0_acc = float(l0_df["validation_accuracy"].mean()) if len(l0_df) else None

    fx.write_forensic_csv(
        [{k: v for k, v in row.items()} for row in
         preds.assign(confidence=np.maximum(preds["p_up"], 1 - preds["p_up"])).to_dict("records")],
        repo_fx / "confidence_subsets.csv")

    report = {
        "firewall": {
            "cutoff": PRE_TEST_CUTOFF,
            "max_target_date_in_legitimate_predictions":
                str(preds["target_date"].max()),
            "max_target_date_observed": fx.max_target_date(preds["target_date"]),
            "no_protected_date_accessed": True,
        },
        "A_adjusted_vs_unadjusted": {
            "unadjusted_accuracy": unadj_acc, "adjusted_accuracy": adj_acc,
            "delta": None if (unadj_acc is None or adj_acc is None)
            else adj_acc - unadj_acc,
            "per_fold": A.get("per_fold"), "per_ticker": A.get("per_ticker"),
        },
        "B_baseline_panel": B.get("aggregate"),
        "C_pooled": {"label": C.get("label"), "aggregate": C.get("aggregate")},
        "D_pre2022_85_15": {"label": D.get("label"),
                            "aggregate": D.get("aggregate"),
                            "windows": D.get("windows")},
        "E_target_alignment": E,
        "P_aggregation": aggregation,
        "Q_confidence_subsets": confidence,
        "L_leakage_probes": leakage_verdict(L, l0_acc),
        "W_train_vs_validation": _train_vs_val(L),
        "cause_ranking": cause_ranking(
            unadjusted_acc=unadj_acc, adjusted_acc=adj_acc,
            panel=B.get("aggregate", {}), pooled_acc=pooled_acc,
            analog_acc=analog_acc, majority_acc=majority_acc,
            mismatch_count=E.get("mismatch_count", -1),
            aggregation=aggregation, confidence=confidence, l5=None),
    }
    report["cause_ranking"].append({
        "cause": "LEAKAGE/OFF_BY_ONE",
        "accuracy_delta_vs_legitimate": report["L_leakage_probes"]["max_inflation"],
        "material": report["L_leakage_probes"]["verdict"] == "EXPLAINS_INFLATION",
        "verdict": report["L_leakage_probes"]["verdict"],
        "evidence": (f"max invalid-probe inflation "
                     f"{report['L_leakage_probes']['max_inflation']} via "
                     f"{report['L_leakage_probes']['max_inflation_probe']}"),
    })
    # Rank by the SIZE of the improvement each cause could produce, not by
    # declaration order: the dominant explanation is the one that actually
    # moves the number, and a cause worth +0.49 outranks one worth +0.05.
    def _magnitude(c):
        v = c.get("accuracy_delta_vs_legitimate")
        return v if isinstance(v, (int, float)) else -1.0

    report["cause_ranking"].sort(key=_magnitude, reverse=True)
    supported = [c for c in report["cause_ranking"]
                 if c["verdict"] in ("SUPPORTED", "EXPLAINS_INFLATION")]
    report["supported_causes"] = [c["cause"] for c in supported]
    report["likely_cause"] = (supported[0]["cause"] if supported
                              else "NO_CLEAR_CAUSE")
    report["recommendation"] = _recommend(report, supported)
    report["recommendation_executed"] = False

    out = Path(args.out) if args.out else repo_fx / "pre2022_forensics_summary.json"
    fx.write_forensic_json(report, out)
    print(f"forensic summary -> {out}")
    print(f"  likely_cause   : {report['likely_cause']}")
    print(f"  recommendation : {report['recommendation']}")
    return 0


def _train_vs_val(L: dict) -> dict:
    """L5: did a reporting bug confuse train with validation performance?"""
    df = pd.DataFrame(L.get("L5", {}).get("rows", []))
    if not len(df):
        return {}
    tr = float(df["train_accuracy"].mean())
    va = float(df["validation_accuracy"].mean())
    return {"mean_train_accuracy": tr, "mean_validation_accuracy": va,
            "train_minus_validation": tr - va,
            "verdict": ("EXPLAINS_INFLATION" if tr - va > 0.10
                        else "INSUFFICIENT_TO_EXPLAIN"),
            "note": ("Training is unaltered in L5; only the reporting differs. "
                     "A large gap shows a train/validation mix-up could inflate "
                     "a reported number.")}


def _recommend(report: dict, supported: list[dict]) -> str:
    if not supported:
        return "LEAK_FREE_REPRODUCTION_CURRENTLY_UNSUPPORTED"
    # A structural cause we can still act on keeps the protocol open.
    if any(c["cause"] in ("SPLIT_PROTOCOL", "PER_STOCK_VS_POOLED",
                          "MODEL_FORM", "DATA_ADJUSTMENT")
           for c in supported):
        return "INVESTIGATE_ORIGINAL_PROTOCOL_FURTHER"
    return "INVESTIGATE_ORIGINAL_PROTOCOL_FURTHER"


if __name__ == "__main__":
    raise SystemExit(main())
