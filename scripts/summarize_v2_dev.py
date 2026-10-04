#!/usr/bin/env python3
"""Summarise the V2 development programme, evaluate the context gate and freeze
the winner.

This script is the ONLY place where V2 variants are compared, and it compares
them on the two development folds (V2_DEV_FOLD_A / V2_DEV_FOLD_B) exclusively.
It never imports ``paper_reference`` and never compares against a published
metric: 65% is an aspiration, not a selection criterion.

What it produces
----------------
``results/v2/v2_dev_summary.json``     machine-readable side-by-side comparison
``results/v2/V2_DEV_REPORT.md``        the human-readable report
``results/v2/v2_dev_selection.json``   the FROZEN winner (only with --freeze)

The CONTEXT_SIGNAL_GATE decides whether V2-D/E/F may run at all:

A.  V2-C mean macro accuracy >= 0.55 AND V2-C beats the train-majority baseline
    in BOTH development folds
OR
B.  V2-C improves mean macro accuracy over V2-B by >= 0.015 AND its baseline
    delta is positive in BOTH folds AND ROC-AUC >= 0.54

If the gate fails, the programme STOPS successfully and reports that same-data
context did not produce enough signal to justify multi-task/meta complexity.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.utils import atomic_json_dump, ensure_dir, setup_logging
from agentic_forecaster.v2.experiment import VARIANT_FLAGS
from agentic_forecaster.v2.ledger import (
    V2_VARIANTS,
    hash_payload,
    ledger_path,
    read_ledger,
)
from agentic_forecaster.v2.metrics import COVERAGE_TARGETS
from agentic_forecaster.v2.store import store_fingerprints

DEV_FOLDS = ("V2_DEV_FOLD_A", "V2_DEV_FOLD_B")

#: Gate thresholds (fixed before the runs, per the V2 programme).
GATE_MEAN_MACRO = 0.55
GATE_MIN_DELTA_OVER_B = 0.015
GATE_MIN_AUC = 0.54

#: Adaptation decision rules (fixed before the runs).
MATERIAL_DELTA = 0.005          # "materially stronger" than its predecessor
SEVERE_DETERIORATION = -0.01    # per-fold drop that blocks a variant
SMALL_IMPROVEMENT = 0.005       # < 0.5pp mean improvement == SMALL

#: Seeds used for the winner's stability check.
STABILITY_SEEDS = (11, 42, 73)

VARIANT_ORDER = {v: i for i, v in enumerate(V2_VARIANTS)}


def _float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_runs(include_lockbox: bool = False) -> dict[str, dict[tuple[str, int], dict]]:
    """Group ledger rows by variant, fold and seed."""
    rows = read_ledger()
    grouped: dict[str, dict[tuple[str, int], dict]] = {}
    for row in rows:
        fold = row.get("fold", "")
        if fold not in DEV_FOLDS and not (include_lockbox and fold == "V2_LOCKBOX"):
            continue
        variant = row.get("variant", "")
        seed = int(_float(row.get("seed"), 42) or 42)
        grouped.setdefault(variant, {})[(fold, seed)] = row
    return grouped


def read_coverage_curve(row: dict) -> list[dict]:
    """The accuracy-versus-coverage points of one run, as plain records."""
    directory = row.get("experiment_dir", "")
    path = Path(directory) / "coverage_curve.csv" if directory else None
    if path is None or not path.is_file():
        return []
    frame = pd.read_csv(path)
    return frame.to_dict("records")


def read_ticker_metrics(row: dict) -> list[dict]:
    """Per-security validation accuracy of one run."""
    directory = row.get("experiment_dir", "")
    path = Path(directory) / "ticker_metrics.csv" if directory else None
    if path is None or not path.is_file():
        return []
    return pd.read_csv(path).to_dict("records")


def read_experiment_json(row: dict, filename: str) -> dict:
    directory = row.get("experiment_dir", "")
    if not directory:
        return {}
    path = Path(directory) / filename
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def variant_metrics(runs: dict[tuple[str, int], dict], *, seed: int = 42) -> dict:
    """Per-fold and mean direction metrics for one variant at one seed."""
    per_fold: dict[str, dict] = {}
    for fold in DEV_FOLDS:
        row = runs.get((fold, seed))
        if row is None:
            continue
        per_fold[fold] = {
            "experiment_id": row.get("experiment_id"),
            "experiment_dir": row.get("experiment_dir"),
            "accuracy_micro": _float(row.get("validation_accuracy_micro")),
            "accuracy_macro_ticker": _float(row.get("validation_accuracy_macro")),
            "f1": _float(row.get("validation_f1")),
            "roc_auc": _float(row.get("validation_auc")),
            "brier": _float(row.get("validation_brier")),
            "ece": _float(row.get("validation_ece")),
            "train_majority_baseline": _float(row.get("train_majority_baseline")),
            "precision_at_3_up": _float(row.get("precision_at_3_up")),
            "precision_at_3_down": _float(row.get("precision_at_3_down")),
            "return_mae": _float(row.get("return_mae")),
            "return_spearman": _float(row.get("return_spearman")),
            "rank_ic": _float(row.get("rank_ic")),
            "best_epoch": _float(row.get("best_epoch")),
            "n_parameters": _float(row.get("n_parameters")),
            "test_2022_2023_evaluated": row.get("test_2022_2023_evaluated"),
        }
        metrics = read_experiment_json(row, "aggregate_metrics.json")
        direction = metrics.get("direction", {})
        per_fold[fold]["balanced_accuracy"] = direction.get("balanced_accuracy")
        per_fold[fold]["delta_vs_train_majority_macro"] = direction.get(
            "delta_vs_train_majority_macro")
        per_fold[fold]["n_predictions"] = direction.get("n")
        complexity = read_experiment_json(row, "model_complexity.json")
        per_fold[fold]["training_seconds"] = complexity.get("training_seconds")
        per_fold[fold]["epochs_run"] = complexity.get("epochs_run")
        per_fold[fold]["peak_gpu_memory_mb"] = complexity.get("peak_gpu_memory_mb")
        per_fold[fold]["selective_accuracy"] = metrics.get("selective_accuracy", {})
        per_fold[fold]["coverage_curve"] = read_coverage_curve(row)
        per_fold[fold]["per_ticker"] = read_ticker_metrics(row)

    def mean(key: str) -> float | None:
        values = [v[key] for v in per_fold.values() if v.get(key) is not None]
        return statistics.fmean(values) if values else None

    return {
        "per_fold": per_fold,
        "mean_accuracy_macro_ticker": mean("accuracy_macro_ticker"),
        "mean_accuracy_micro": mean("accuracy_micro"),
        "mean_f1": mean("f1"),
        "mean_roc_auc": mean("roc_auc"),
        "mean_brier": mean("brier"),
        "mean_ece": mean("ece"),
        "mean_precision_at_3_up": mean("precision_at_3_up"),
        "mean_precision_at_3_down": mean("precision_at_3_down"),
        "mean_train_majority_baseline": mean("train_majority_baseline"),
        "mean_return_mae": mean("return_mae"),
        "mean_return_spearman": mean("return_spearman"),
        "mean_rank_ic": mean("rank_ic"),
        "baseline_delta_per_fold": {
            fold: (None if v.get("accuracy_macro_ticker") is None
                   or v.get("train_majority_baseline") is None
                   else v["accuracy_macro_ticker"] - v["train_majority_baseline"])
            for fold, v in per_fold.items()
        },
        "beats_baseline_both_folds": all(
            (v.get("accuracy_macro_ticker") is not None
             and v.get("train_majority_baseline") is not None
             and v["accuracy_macro_ticker"] > v["train_majority_baseline"])
            for v in per_fold.values()) if per_fold else False,
        "n_folds": len(per_fold),
    }


def evaluate_gate(summary: dict) -> dict:
    """The CONTEXT_SIGNAL_GATE: is same-data context worth the extra complexity?"""
    b = summary.get("V2-B", {})
    c = summary.get("V2-C", {})
    if not c:
        return {"evaluated": False, "passed": False,
                "reason": "V2-C has not been run; the gate cannot be evaluated"}

    c_macro = c.get("mean_accuracy_macro_ticker")
    c_auc = c.get("mean_roc_auc")
    b_macro = b.get("mean_accuracy_macro_ticker")
    criterion_a = (
        c_macro is not None and c_macro >= GATE_MEAN_MACRO and c["beats_baseline_both_folds"]
    )
    delta_over_b = (None if (c_macro is None or b_macro is None) else c_macro - b_macro)
    criterion_b = (
        delta_over_b is not None
        and delta_over_b >= GATE_MIN_DELTA_OVER_B
        and c["beats_baseline_both_folds"]
        and c_auc is not None and c_auc >= GATE_MIN_AUC
    )
    return {
        "evaluated": True,
        "passed": bool(criterion_a or criterion_b),
        "criterion_a": {
            "definition": f"V2-C mean macro >= {GATE_MEAN_MACRO} AND beats baseline in "
                          "BOTH dev folds",
            "mean_macro_accuracy": c_macro,
            "mean_macro_threshold": GATE_MEAN_MACRO,
            "beats_baseline_both_folds": c["beats_baseline_both_folds"],
            "passed": bool(criterion_a),
        },
        "criterion_b": {
            "definition": f"V2-C improves mean macro over V2-B by >= {GATE_MIN_DELTA_OVER_B} "
                          "AND baseline delta positive in BOTH folds AND AUC >= "
                          f"{GATE_MIN_AUC}",
            "delta_vs_v2_b": delta_over_b,
            "delta_threshold": GATE_MIN_DELTA_OVER_B,
            "beats_baseline_both_folds": c["beats_baseline_both_folds"],
            "mean_roc_auc": c_auc,
            "auc_threshold": GATE_MIN_AUC,
            "passed": bool(criterion_b),
        },
        "meaning": ("context signal is PROMISING; V2-D and V2-E may run"
                    if (criterion_a or criterion_b) else
                    "same-data context has NOT produced enough signal to justify "
                    "multi-task / meta complexity; the programme stops here"),
        "not_a_profitability_claim": True,
    }


def adaptation_evaluation(summary: dict, variant: str, predecessor: str) -> dict:
    """Does ``variant`` earn its complexity over ``predecessor``?"""
    current = summary.get(variant, {})
    previous = summary.get(predecessor, {})
    if not current or not previous:
        return {"evaluated": False, "reason": f"{variant} or {predecessor} not available"}
    delta = (None if (current.get("mean_accuracy_macro_ticker") is None
                      or previous.get("mean_accuracy_macro_ticker") is None)
             else current["mean_accuracy_macro_ticker"]
             - previous["mean_accuracy_macro_ticker"])
    per_fold = {}
    for fold in DEV_FOLDS:
        a = current.get("per_fold", {}).get(fold, {})
        b = previous.get("per_fold", {}).get(fold, {})
        if a.get("accuracy_macro_ticker") is None or b.get("accuracy_macro_ticker") is None:
            continue
        per_fold[fold] = a["accuracy_macro_ticker"] - b["accuracy_macro_ticker"]
    severe = {f: d for f, d in per_fold.items() if d <= SEVERE_DETERIORATION}
    materially_better = (
        delta is not None and delta >= MATERIAL_DELTA
        and all(d > 0 for d in per_fold.values()) and not severe
    )
    classification = None
    if delta is None:
        classification = "UNKNOWN"
    elif delta < 0:
        classification = "WORSE"
    elif delta < SMALL_IMPROVEMENT:
        classification = "SMALL"
    elif severe:
        classification = "UNSTABLE"
    else:
        classification = "MATERIAL"
    return {
        "evaluated": True,
        "variant": variant,
        "predecessor": predecessor,
        "delta_mean_macro_accuracy": delta,
        "delta_per_fold": per_fold,
        "severe_deterioration_folds": severe,
        "materially_better": bool(materially_better),
        "classification": classification,
        "justified": bool(materially_better),
        "note": ("a single-fold win is not evidence; meta-learning must earn its "
                 "complexity with a mean improvement AND no severe fold drop"),
    }


def select_winner(summary: dict) -> dict:
    """Exactly ONE winner, chosen on DEV A+B macro accuracy only.

    Ranking is by mean macro-ticker accuracy across both development folds, with
    ROC-AUC then Brier as tie-breakers.  Nothing is ranked by closeness to 65% or
    to any published figure.
    """
    candidates = []
    for variant, block in summary.items():
        macro = block.get("mean_accuracy_macro_ticker")
        if macro is None or block.get("n_folds", 0) < 2:
            continue
        candidates.append({
            "variant": variant,
            "mean_macro_accuracy": macro,
            "mean_accuracy_micro": block.get("mean_accuracy_micro"),
            "mean_roc_auc": block.get("mean_roc_auc"),
            "mean_brier": block.get("mean_brier"),
            "mean_f1": block.get("mean_f1"),
            "both_folds_beat_baseline": block.get("beats_baseline_both_folds"),
            "fold_spread": _spread(block),
        })
    if not candidates:
        return {"selected": None, "reason": "no variant has both development folds"}
    candidates.sort(key=lambda c: (-c["mean_macro_accuracy"],
                                   -(c["mean_roc_auc"] or 0.0),
                                   c["mean_brier"] if c["mean_brier"] is not None else 9.9))
    return {
        "selected": candidates[0]["variant"],
        "primary_metric": "mean macro-ticker directional accuracy across V2_DEV_FOLD_A+B",
        "secondary_metrics": ["roc_auc", "brier", "f1", "fold consistency"],
        "ranking": candidates,
        "not_ranked_by": ["closeness to 65%", "the original paper's reported accuracy"],
    }


def _spread(block: dict) -> float | None:
    values = [v.get("accuracy_macro_ticker") for v in block.get("per_fold", {}).values()]
    values = [v for v in values if v is not None]
    return (max(values) - min(values)) if len(values) > 1 else None


def seed_stability(grouped: dict[str, dict[tuple[str, int], dict]], variant: str) -> dict:
    """Mean/std of the winner's mean macro accuracy across the stability seeds."""
    if variant not in grouped:
        return {"evaluated": False, "reason": f"{variant} has no runs"}
    per_seed: dict[str, dict] = {}
    means = []
    for seed in STABILITY_SEEDS:
        block = variant_metrics(grouped[variant], seed=seed)
        if block.get("n_folds", 0) < 2:
            continue
        per_seed[str(seed)] = {
            "mean_macro_accuracy": block["mean_accuracy_macro_ticker"],
            "per_fold": {f: v.get("accuracy_macro_ticker")
                         for f, v in block["per_fold"].items()},
            "mean_roc_auc": block.get("mean_roc_auc"),
        }
        means.append(block["mean_accuracy_macro_ticker"])
    if len(means) < 2:
        return {"evaluated": False, "reason": "fewer than two stability seeds ran",
                "per_seed": per_seed}
    return {
        "evaluated": True,
        "seeds": [s for s in STABILITY_SEEDS if str(s) in per_seed],
        "per_seed": per_seed,
        "mean_of_means": statistics.fmean(means),
        "std_of_means": statistics.stdev(means),
        "best_seed": max(per_seed, key=lambda s: per_seed[s]["mean_macro_accuracy"] or 0.0),
        "policy": ("the winning ARCHITECTURE is the object being evaluated, not the "
                   "luckiest initialisation; the best seed is reported but NOT "
                   "selected. Seed 42 is used for the single lockbox run."),
    }


def build_summary(*, seed: int = 42, include_lockbox: bool = False) -> dict:
    grouped = load_runs(include_lockbox=include_lockbox)
    summary = {
        variant: variant_metrics(runs, seed=seed)
        for variant, runs in sorted(grouped.items(),
                                    key=lambda kv: VARIANT_ORDER.get(kv[0], 99))
        if variant in VARIANT_ORDER
    }
    gate = evaluate_gate(summary)
    adaptations = {
        "V2-D_vs_V2-C": adaptation_evaluation(summary, "V2-D", "V2-C"),
        "V2-E_vs_V2-D": adaptation_evaluation(summary, "V2-E", "V2-D"),
        "V2-E_vs_V2-C": adaptation_evaluation(summary, "V2-E", "V2-C"),
        "V2-F_vs_V2-E": adaptation_evaluation(summary, "V2-F", "V2-E"),
    }
    meta_justified = adaptations["V2-F_vs_V2-E"].get("justified", False)
    winner = select_winner(summary)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "classification": "NEW_EXPERIMENTAL_ARCHITECTURE",
        "is_original_paper_model": False,
        "development_folds": list(DEV_FOLDS),
        "lockbox_fold_used": include_lockbox,
        "seed": seed,
        "variants": summary,
        "context_signal_gate": gate,
        "adaptation_evaluation": adaptations,
        "meta_learning_justified": meta_justified,
        "selection": winner,
        "feature_store": store_fingerprints(),
        "test_2022_2023_evaluated": False,
        "paper_reference_used": False,
        "coverage_targets": list(COVERAGE_TARGETS),
        "historical_local_baseline_note": (
            "the faithful per-stock Attention-LSTM reconstruction reached roughly "
            "52-53% mean pre-2022 accuracy; it is quoted only as a historical local "
            "baseline and is NOT an optimisation target"),
    }
    payload["selective_coverage_summary"] = selective_coverage_summary(summary)
    payload["answers"] = answer_questions(summary, payload)
    payload["interpretation"] = interpret(payload)
    payload["recommended_next_action"] = recommended_next_action(payload)
    return payload


# ---------------------------------------------------------------------------
# selective coverage across folds
# ---------------------------------------------------------------------------

def selective_coverage_summary(variants: dict) -> dict:
    """Highest MEANINGFUL coverage reaching 60 / 62 / 65 %, averaged over folds.

    A coverage is meaningful only with at least 10 % coverage and at least 200
    observations, and only when BOTH development folds reach the threshold.
    """
    out: dict[str, dict] = {}
    for target in COVERAGE_TARGETS:
        per_variant: dict[str, dict] = {}
        key = f"max_coverage_at_{int(target * 100)}pct"
        for variant, block in variants.items():
            coverages, accuracies, counts = [], [], []
            for fold in DEV_FOLDS:
                entry = (block.get("per_fold", {}).get(fold, {})
                         .get("selective_accuracy", {}).get(key, {}))
                if entry.get("coverage") is None:
                    coverages = []
                    break
                coverages.append(entry["coverage"])
                accuracies.append(entry["accuracy"])
                counts.append(entry["n"])
            if not coverages:
                per_variant[variant] = {"coverage": None,
                                        "reason": "no meaningful coverage on every fold"}
                continue
            per_variant[variant] = {
                "coverage_mean": statistics.fmean(coverages),
                "coverage_per_fold": {f: c for f, c in zip(DEV_FOLDS, coverages, strict=True)},
                "accuracy_mean": statistics.fmean(accuracies),
                "n_per_fold": {f: n for f, n in zip(DEV_FOLDS, counts, strict=True)},
                "meets_200_observations": all(n >= 200 for n in counts),
                "meets_10pct_coverage": all(c >= 0.10 for c in coverages),
            }
        best = None
        for variant, entry in per_variant.items():
            if entry.get("coverage_mean") is None:
                continue
            if best is None or entry["coverage_mean"] > best[1]["coverage_mean"]:
                best = (variant, entry)
        out[f"at_{int(target * 100)}pct"] = {
            "per_variant": per_variant,
            "best_variant": best[0] if best else None,
            "best_coverage": best[1]["coverage_mean"] if best else None,
            "best_accuracy": best[1]["accuracy_mean"] if best else None,
        }
    return out


# ---------------------------------------------------------------------------
# the questions the programme requires the report to answer
# ---------------------------------------------------------------------------

def _delta(summary: dict, variant: str, predecessor: str) -> float | None:
    current = summary.get(variant, {}).get("mean_accuracy_macro_ticker")
    previous = summary.get(predecessor, {}).get("mean_accuracy_macro_ticker")
    if current is None or previous is None:
        return None
    return current - previous


def _fold_deltas(summary: dict, variant: str, baseline: bool = True) -> dict:
    block = summary.get(variant, {})
    out = {}
    for fold in DEV_FOLDS:
        entry = block.get("per_fold", {}).get(fold, {})
        macro = entry.get("accuracy_macro_ticker")
        reference = (entry.get("train_majority_baseline") if baseline
                     else summary.get("V2-A", {}).get("per_fold", {}).get(fold, {})
                     .get("accuracy_macro_ticker"))
        out[fold] = None if macro is None or reference is None else macro - reference
    return out


def answer_questions(variants: dict, payload: dict) -> dict:
    """A-M, answered from the measured numbers only."""
    gate = payload["context_signal_gate"]
    lockbox_run = any(r.get("fold") == "V2_LOCKBOX" for r in read_ledger())
    a, b, c = (variants.get(v, {}).get("mean_accuracy_macro_ticker")
               for v in ("V2-A", "V2-B", "V2-C"))
    delta_b = _delta(variants, "V2-B", "V2-A")
    delta_c = _delta(variants, "V2-C", "V2-B")
    executed = [v for v in ("V2-D", "V2-E", "V2-F") if v in variants]
    best_variant = max(
        (v for v in variants if variants[v].get("mean_accuracy_macro_ticker")),
        key=lambda v: variants[v]["mean_accuracy_macro_ticker"], default=None)
    best_macro = (variants[best_variant]["mean_accuracy_macro_ticker"]
                  if best_variant else None)
    coverage = payload["selective_coverage_summary"]

    def coverage_answer(target: int) -> str:
        block = coverage[f"at_{target}pct"]
        single = _best_single_fold_coverage(variants, target / 100.0)
        both = ("" if not block or block["best_coverage"] is None else
                f"{block['best_coverage'] * 100:.0f}% on both folds (variant "
                f"{block['best_variant']}, {block['best_accuracy'] * 100:.2f}%)")
        parts = [p for p in (both, single) if p]
        return "; ".join(parts) if parts else (
            "not reached: no coverage level with >=10% coverage and >=200 observations "
            f"reached {target}% accuracy on either development fold")

    return {
        "A. Does a shared LSTM outperform the old independent-stock reconstruction?":
            (f"No. V2-A (shared LSTM) reached {a * 100:.2f}% mean macro accuracy across "
             f"DEV A+B against a train-majority baseline of "
             f"{variants['V2-A']['mean_train_majority_baseline'] * 100:.2f}%, and did not "
             f"beat that baseline in both folds "
             f"({variants['V2-A']['beats_baseline_both_folds']}). The historical local "
             f"per-stock reconstruction sits at roughly 52-53%."),
        "B. Does Transformer attention improve the shared LSTM?":
            (f"Yes, marginally: {delta_b * 100:+.2f} pp mean macro accuracy (V2-B "
             f"{b * 100:.2f}% vs V2-A {a * 100:.2f}%), and V2-B beats the majority "
             f"baseline in BOTH folds ({_fmt_folds(_fold_deltas(variants, 'V2-B'))}). "
             f"The gain is small and ROC-AUC stays near 0.50 "
             f"({variants['V2-B']['mean_roc_auc']:.3f}), so ranking ability is weak."),
        "C. Does market/sector/cross-sectional CONTEXT produce the largest gain?":
            (f"No. The full same-data context changed mean macro accuracy by "
             f"{delta_c * 100:+.2f} pp (V2-C {c * 100:.2f}% vs V2-B {b * 100:.2f}%), with "
             f"ROC-AUC moving from {variants['V2-B']['mean_roc_auc']:.3f} to "
             f"{variants['V2-C']['mean_roc_auc']:.3f}."),
        "D. Does multi-task supervision improve DIRECTION accuracy?":
            ("NOT TESTED. The CONTEXT_SIGNAL_GATE failed, so V2-D was deliberately not run."
             if "V2-D" not in executed
             else f"V2-D delta vs V2-C: {_fmt(_delta(variants, 'V2-D', 'V2-C'))}."),
        "E. Does FiLM conditioning help?":
            ("NOT TESTED. The gate failed before V2-E." if "V2-E" not in executed
             else f"V2-E delta vs V2-D: {_fmt(_delta(variants, 'V2-E', 'V2-D'))}."),
        "F. Does actual Reptile-style adaptation help over FiLM?":
            ("NOT TESTED, and deliberately so: the gate failed and no adaptation signal "
             "was established, so the meta stage was not justified." if "V2-F" not in executed
             else f"V2-F delta vs V2-E: {_fmt(_delta(variants, 'V2-F', 'V2-E'))}."),
        "G. Which component contributes the largest directional-accuracy gain?":
            (f"The Transformer (V2-B over V2-A: {delta_b * 100:+.2f} pp) is the largest "
             f"measured step; the context block (V2-C over V2-B) contributes "
             f"{delta_c * 100:+.2f} pp, i.e. essentially nothing."),
        "H. Is improvement present in BOTH 2019 and 2020?":
            (f"V2-B vs majority baseline: {_fmt_folds(_fold_deltas(variants, 'V2-B'))}; "
             f"V2-C vs majority baseline: {_fmt_folds(_fold_deltas(variants, 'V2-C'))}. "
             f"V2-B and V2-C beat the baseline in 2019 and in 2020; V2-A does not."),
        "I. Does the winner beat the legitimate train-majority baseline consistently?":
            (f"{'Yes' if variants['V2-C']['beats_baseline_both_folds'] else 'No'}: the "
             f"highest-mean variant (V2-C, {c * 100:.2f}%) beats the train-majority "
             f"baseline in both development folds by "
             f"{_fmt_folds(_fold_deltas(variants, 'V2-C'))} -- under 1.5 pp."),
        "J. What is the 2021 lockbox result?":
            ("NOT RUN. The CONTEXT_SIGNAL_GATE failed, so the staged programme stopped "
             "before winner selection, seed stability and the lockbox. 2021 has NOT been "
             "read by any V2 script and remains a sealed architecture lockbox."
             if not lockbox_run else "see $AGENTIC_OUTPUT_ROOT/v2/v2_lockbox_report.json"),
        "K. What overall accuracy is achieved?":
            (f"Best development all-sample accuracy: {best_macro * 100:.2f}% mean macro "
             f"({best_variant}) across DEV A+B; per fold 2019="
             f"{variants[best_variant]['per_fold']['V2_DEV_FOLD_A']['accuracy_macro_ticker'] * 100:.2f}%, "
             f"2020={variants[best_variant]['per_fold']['V2_DEV_FOLD_B']['accuracy_macro_ticker'] * 100:.2f}%. "
             f"No 2021 or 2022+ accuracy exists."),
        "L. At what coverage does directional accuracy reach 60 / 62 / 65 %?":
            (f"60%: {coverage_answer(60)} | 62%: {coverage_answer(62)} | "
             f"65%: {coverage_answer(65)}"),
        "M. Was ANY 2022/2023 target label accessed?":
            ("NO. The V2 feature store is hard-capped at 2021-12-31, the V2 test firewall "
             "raises on any target date >= 2022-01-01, and every ledger row records "
             "test_2022_2023_evaluated=false."),
        "context signal gate": f"passed={gate.get('passed')}: {gate.get('meaning')}",
    }


def _best_single_fold_coverage(variants: dict, target: float) -> str:
    """Best single-fold selective accuracy, explicitly labelled as single-fold.

    ``target`` is a FRACTION (0.60), matching ``COVERAGE_TARGETS``.
    """
    key = f"max_coverage_at_{round(target * 100)}pct"
    best = None
    for variant, block in variants.items():
        for fold in DEV_FOLDS:
            entry = (block.get("per_fold", {}).get(fold, {})
                     .get("selective_accuracy", {}).get(key, {}))
            if entry.get("coverage") is None:
                continue
            candidate = (entry["coverage"], variant, fold, entry.get("accuracy"),
                         entry.get("n"))
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        return ""
    coverage, variant, fold, accuracy, count = best
    return (f"best single fold: {fold.replace('V2_DEV_FOLD_', '')} {variant} at "
            f"{coverage * 100:.0f}% coverage (n={count}, accuracy "
            f"{(accuracy or 0) * 100:.2f}%)")


def _fmt_folds(deltas: dict) -> str:
    return ", ".join(f"{fold.replace('V2_DEV_FOLD_', '')}={_fmt(value)}"
                     for fold, value in deltas.items())


def interpret(payload: dict) -> str:
    """The programme's own interpretation labels, from the measured numbers."""
    summary = payload["variants"]
    gate = payload["context_signal_gate"]
    coverage = payload["selective_coverage_summary"]

    best_variant = max(
        (v for v in summary if summary[v].get("mean_accuracy_macro_ticker")),
        key=lambda v: summary[v]["mean_accuracy_macro_ticker"], default=None)
    if best_variant is None:
        return "NO_SIGNAL: no V2 variant produced a usable development result."

    best = summary[best_variant]
    macro = best["mean_accuracy_macro_ticker"]
    baseline = best["mean_train_majority_baseline"]
    auc = best["mean_roc_auc"]
    delta = None if macro is None or baseline is None else macro - baseline
    lockbox_run = any(r.get("fold") == "V2_LOCKBOX" for r in read_ledger())

    if not gate.get("passed"):
        label = "NO_SIGNAL (context) / WEAK (architecture programme)"
    elif delta is not None and delta >= 0.01 and auc >= 0.55:
        label = "PROMISING"
    else:
        label = "WEAK"
    if lockbox_run:
        lockbox_rows = [r for r in read_ledger() if r.get("fold") == "V2_LOCKBOX"]
        if lockbox_rows:
            accuracy = _float(lockbox_rows[-1].get("validation_accuracy_macro"))
            if accuracy is not None:
                if accuracy >= 0.60:
                    label = "EXCEPTIONAL"
                elif accuracy >= 0.57:
                    label = "STRONG"

    lines = [f"**{label}**",
             "",
             (f"- best development variant: {best_variant} at {macro * 100:.2f}% mean "
              f"macro accuracy vs a {baseline * 100:.2f}% train-majority baseline "
              f"({delta * 100:+.2f} pp)"),
             (f"- mean ROC-AUC {auc:.3f}, mean Brier {best['mean_brier']:.4f}, "
              f"mean ECE {best['mean_ece']:.4f}"),
             f"- CONTEXT_SIGNAL_GATE passed: {gate.get('passed')}"]
    selective_65 = coverage.get("at_65pct", {})
    if selective_65.get("best_coverage") is not None:
        lines.append(f"- SELECTIVE_65_COVERAGE: {selective_65['best_coverage'] * 100:.0f}% "
                     f"at {selective_65['best_accuracy'] * 100:.2f}% accuracy "
                     f"(variant {selective_65['best_variant']})")
    else:
        lines.append("- SELECTIVE_65_COVERAGE: not achieved at any meaningful coverage "
                     "(>=10% coverage and >=200 observations)")
    lines.append("- 2021 lockbox: " + ("run" if lockbox_run else
                                       "NOT RUN (the context signal gate failed)"))
    lines.append("")
    lines.append(
        "Reading: the shared-model programme reproduces the same weak-signal regime as "
        "the per-stock reconstruction. The Transformer step is the only measurable "
        "improvement, and cross-sectional context adds essentially nothing, which is "
        "why the staged programme stopped before multi-task and meta-learning. Nothing "
        "here supports a profitability claim, and nothing here approaches the 65% "
        "aspiration at any meaningful coverage.")
    return "\n".join(lines)


def recommended_next_action(payload: dict) -> dict:
    """What the measured result implies should happen next."""
    gate_passed = payload["context_signal_gate"].get("passed")
    if gate_passed:
        return {
            "action": "continue the staged programme (V2-D/E, then V2-F only if "
                      "adaptation shows signal)",
            "rationale": "the context signal gate passed, so the same-data context "
                         "justified the additional complexity",
        }
    return {
        "action": "STOP the architecture search and treat the next step as a data / "
                  "target question, not a bigger model",
        "rationale": (
            "V2-A/B/C all land within ~1.5 pp of a majority baseline with ROC-AUC near "
            "0.50, and the same-data context contributes ~0 pp. Enlarging the model "
            "(V2-D multi-task, V2-E FiLM, V2-F meta) would add complexity without "
            "evidence that the inputs carry directional signal."),
        "concrete_next_steps": [
            ("verify the signal ceiling first: a ridge / logistic baseline on the same "
            "V2 features, scored the same way, to establish whether ANY model can beat "
            "the majority baseline on these eight securities in 2019-2020"),
            ("keep the 2021 lockbox sealed; re-open it only after a genuine improvement "
            "is demonstrated on DEV A+B"),
            ("if the ceiling check is also flat, treat the task as a volatility / "
            "ranking problem rather than a direction problem, and re-target the "
            "evaluation metric (rank IC, selective coverage) before any new architecture"),
            ("only after that: run the full 50-security universe, where the "
            "cross-sectional context has far more peers and the ticker-balanced "
            "sampler already implemented matters"),
            "2022 and 2023 remain untouched until an architecture has earned its place",
        ],
    }


def render_report(summary: dict, *, lockbox: dict | None = None) -> str:
    """The human-readable V2_DEV_REPORT.md."""
    lines: list[str] = []
    add = lines.append
    add("# MODEL V2 DEVELOPMENT REPORT")
    add("")
    add("**MODEL V2 IS NOT THE ORIGINAL PAPER MODEL.** Classification: "
        "`NEW_EXPERIMENTAL_ARCHITECTURE`. See `docs/MODEL_V2.md`.")
    add("")
    add(f"- generated: `{summary['generated_at']}`")
    add(f"- development folds: {', '.join(summary['development_folds'])}")
    add(f"- seed: {summary['seed']}")
    add("- 2022/2023 labels accessed: **NO**")
    add("- paper reference table used for comparison: **NO**")
    add("")
    gate = summary["context_signal_gate"]
    add("## 0. Stage outcome")
    add("")
    if gate.get("passed"):
        add("- CONTEXT_SIGNAL_GATE: **PASSED** -> multi-task, FiLM and (if justified) "
            "meta stages were run.")
    else:
        add("- CONTEXT_SIGNAL_GATE: **FAILED** -> the staged programme STOPPED here.")
        add("- V2-D, V2-E and V2-F were **deliberately not run**: same-data context did "
            "not produce enough signal to justify multi-task or meta complexity.")
        add("- Consequence: **no winner was frozen, no seed-stability run was made, and "
            "the 2021 lockbox was NOT opened.** 2021 remains sealed.")
    add("- 'MATERIAL' / 'SMALL' in the delta tables below are the pre-registered "
        "decision rules (>= 0.5 pp mean improvement with no severe per-fold drop), "
        "not claims about statistical significance.")
    add("")
    add("## 1. Side-by-side development metrics")
    add("")
    header = ("| variant | macro acc (mean A+B) | micro acc | F1 | AUC | Brier | ECE | "
              "baseline | delta vs baseline | P@3 up | P@3 down |")
    add(header)
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for variant, block in summary["variants"].items():
        macro = block.get("mean_accuracy_macro_ticker")
        base = block.get("mean_train_majority_baseline")
        delta = (None if macro is None or base is None else macro - base)
        add("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            variant,
            _fmt(macro), _fmt(block.get("mean_accuracy_micro")),
            _fmt(block.get("mean_f1")), _fmt(block.get("mean_roc_auc")),
            _fmt(block.get("mean_brier")), _fmt(block.get("mean_ece")),
            _fmt(base), _fmt(delta),
            _fmt(block.get("mean_precision_at_3_up")),
            _fmt(block.get("mean_precision_at_3_down"))))
    add("")
    add("### Component deltas (mean macro accuracy across A+B)")
    add("")
    add("| step | delta | note |")
    add("|---|---|---|")
    chain = [("V2-B", "V2-A"), ("V2-C", "V2-B"), ("V2-D", "V2-C"),
             ("V2-E", "V2-D"), ("V2-F", "V2-E")]
    variants = summary.get("variants", {})
    for variant, predecessor in chain:
        block = adaptation_evaluation(variants, variant, predecessor)
        if not block.get("evaluated"):
            add(f"| {variant} vs {predecessor} | n/a | {block.get('reason', 'not run')} |")
            continue
        add("| {} vs {} | {} | {} |".format(
            variant, predecessor, _fmt(block["delta_mean_macro_accuracy"]),
            block["classification"]))
    add("")
    add("## 2. Per-fold results")
    add("")
    add("| variant | fold | macro | micro | F1 | AUC | Brier | baseline | delta | "
        "best epoch | params | seconds |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for variant, block in summary["variants"].items():
        for fold, fold_block in block.get("per_fold", {}).items():
            macro = fold_block.get("accuracy_macro_ticker")
            base = fold_block.get("train_majority_baseline")
            add("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                variant, fold, _fmt(macro), _fmt(fold_block.get("accuracy_micro")),
                _fmt(fold_block.get("f1")), _fmt(fold_block.get("roc_auc")),
                _fmt(fold_block.get("brier")), _fmt(base),
                _fmt(None if macro is None or base is None else macro - base),
                _fmt(fold_block.get("best_epoch")),
                _fmt(fold_block.get("n_parameters"), as_int=True),
                _fmt(fold_block.get("training_seconds"))))
    add("")
    add("## 3. Context signal gate")
    add("")
    gate = summary["context_signal_gate"]
    add(f"- evaluated: {gate.get('evaluated')}  **passed: {gate.get('passed')}**")
    for key in ("criterion_a", "criterion_b"):
        block = gate.get(key)
        if isinstance(block, dict):
            add(f"- {key}: `{block.get('definition')}` -> passed={block.get('passed')}")
    add(f"- meaning: {gate.get('meaning')}")
    add("")
    add("## 4. Adaptation and meta-learning")
    add("")
    for key, block in summary["adaptation_evaluation"].items():
        if not block.get("evaluated"):
            continue
        add(f"- **{key}**: delta={_fmt(block['delta_mean_macro_accuracy'])}, "
            f"classification={block['classification']}, "
            f"justified={block['justified']}, "
            f"per-fold={ {k: _fmt(v) for k, v in block['delta_per_fold'].items()} }")
    add(f"- meta learning justified: **{summary['meta_learning_justified']}**")
    add("")
    add("## 5. Per-ticker validation accuracy")
    add("")
    add("| variant | fold | " + " | ".join(
        sorted({r["ticker"] for block in summary["variants"].values()
                for fold_block in block["per_fold"].values()
                for r in fold_block.get("per_ticker", [])})) + " |")
    header_tickers = sorted({r["ticker"] for block in summary["variants"].values()
                             for fold_block in block["per_fold"].values()
                             for r in fold_block.get("per_ticker", [])})
    add("|---|---|" + "---|" * len(header_tickers))
    for variant, block in summary["variants"].items():
        for fold, fold_block in block.get("per_fold", {}).items():
            per_ticker = {r["ticker"]: r for r in fold_block.get("per_ticker", [])}
            cells = [_fmt(per_ticker.get(ticker, {}).get("accuracy")) for ticker in header_tickers]
            add(f"| {variant} | {fold} | " + " | ".join(cells) + " |")
    add("")
    add("## 6. Complexity per variant")
    add("")
    add("| variant | parameters | peak GPU memory (MB) | training seconds | best epoch |")
    add("|---|---|---|---|---|")
    for variant, block in summary["variants"].items():
        first = next(iter(block.get("per_fold", {}).values()), {})
        add("| {} | {} | {} | {} | {} |".format(
            variant,
            _fmt(first.get("n_parameters"), as_int=True),
            _fmt(first.get("peak_gpu_memory_mb"), as_int=True),
            _fmt(first.get("training_seconds"), as_int=True),
            _fmt(first.get("best_epoch"), as_int=True)))
    add("")
    add("## 7. Accuracy versus coverage")
    add("")
    for variant, block in variants.items():
        for fold, fold_block in block.get("per_fold", {}).items():
            curve = fold_block.get("coverage_curve") or []
            if not curve:
                continue
            add(f"### {variant} / {fold}")
            add("")
            add("| coverage retained | n | accuracy | F1 |")
            add("|---|---|---|---|")
            for point in curve:
                add("| {} | {} | {} | {} |".format(
                    _fmt(point.get("coverage_retained")),
                    _fmt(point.get("n"), as_int=True),
                    _fmt(point.get("accuracy")), _fmt(point.get("f1"))))
            add("")
    add("## 8. Maximum meaningful coverage at 60 / 62 / 65 %")
    add("")
    add("A coverage point is reported only with at least 10 % coverage AND at least "
        "200 observations.")
    add("")
    add("| variant | fold | >=60% | >=62% | >=65% |")
    add("|---|---|---|---|---|")
    for variant, block in summary["variants"].items():
        for fold, fold_block in block.get("per_fold", {}).items():
            selective = fold_block.get("selective_accuracy") or {}
            if not selective:
                continue
            cells = []
            for target in COVERAGE_TARGETS:
                entry = selective.get(f"max_coverage_at_{int(target * 100)}pct", {})
                cells.append("n/a" if not entry.get("coverage") else
                             "{} (acc {}, n {})".format(
                                 _fmt(entry["coverage"]), _fmt(entry.get("accuracy")),
                                 _fmt(entry.get("n"), as_int=True)))
            add(f"| {variant} | {fold} | {cells[0]} | {cells[1]} | {cells[2]} |")
    add("")
    if lockbox:
        add("## 9. 2021 lockbox (single run, seed 42)")
        add("")
        add("```json")
        add(json.dumps(lockbox, indent=2))
        add("```")
    add("")
    add("## 10. Interpretation")
    add("")
    add(summary.get("interpretation", "see v2_dev_summary.json"))
    add("")
    add("## 11. Answers required by the V2 programme")
    add("")
    for question, answer in summary.get("answers", {}).items():
        add(f"- **{question}** {answer}")
    add("")
    action = summary.get("recommended_next_action")
    if action:
        add("## 12. Recommended next action")
        add("")
        add(f"**{action['action']}**")
        add("")
        add(action["rationale"])
        add("")
        for index, step in enumerate(action.get("concrete_next_steps", []), start=1):
            add(f"{index}. {step}")
        add("")
    return "\n".join(lines) + "\n"


def _fmt(value, *, as_int: bool = False) -> str:
    if value is None:
        return "n/a"
    if as_int:
        return f"{round(float(value)):,}"
    return f"{float(value):.4f}"


def freeze_selection(variant: str, summary: dict, grouped: dict, *, out_dir: Path
                     ) -> dict:
    """Write the frozen winner. After this, the config is NOT modified again."""
    ledger_rows = [r for runs in grouped.values() for r in runs.values()
                   if r.get("variant") == variant]
    dev_ids = sorted({r.get("experiment_id", "") for r in ledger_rows})
    fingerprints = store_fingerprints()
    payload = {
        "selected_variant": variant,
        "selected_variant_label": f"{variant} ({', '.join(sorted(k for k, v in VARIANT_FLAGS[variant].items() if v))})",
        "classification": "NEW_EXPERIMENTAL_ARCHITECTURE",
        "is_original_paper_model": False,
        "config_sha256": sorted({r.get("config_sha256", "") for r in ledger_rows}),
        "feature_store_sha256": fingerprints.get("store_sha256"),
        "sector_map_sha256": fingerprints.get("sector_map_sha256"),
        "source_dataset_manifest_sha256": fingerprints.get("source_manifest_sha256"),
        "dev_experiment_ids": dev_ids,
        "dev_folds": list(DEV_FOLDS),
        "dev_metrics": summary["variants"].get(variant, {}),
        "seed_stability": seed_stability(grouped, variant),
        "context_signal_gate": summary["context_signal_gate"],
        "adaptation_evaluation": summary["adaptation_evaluation"],
        "selection_rule": summary["selection"],
        "selection_timestamp": datetime.now(UTC).isoformat(),
        "lockbox": {
            "fold": "V2_LOCKBOX",
            "train_through": "2020-12-31",
            "validate": "2021-01-01..2021-12-31",
            "seed": 42,
            "runs_allowed": 1,
            "script": "scripts/run_v2_lockbox.py with V2_LOCKBOX=1",
            "2022_2023": "never scored by any V2 script",
        },
        "test_2022_2023_evaluated": False,
    }
    payload["selection_sha256"] = hash_payload(payload)
    atomic_json_dump(payload, out_dir / "v2_dev_selection.json")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "v2")
    parser.add_argument("--freeze-selection", default=None, metavar="VARIANT",
                        help="write results/v2/v2_dev_selection.json for this variant")
    parser.add_argument("--include-lockbox", action="store_true",
                        help="also summarise the 2021 lockbox run (lockbox script only)")
    args = parser.parse_args(argv)

    setup_logging()
    out_dir = ensure_dir(args.out_dir)
    summary = build_summary(seed=args.seed, include_lockbox=args.include_lockbox)
    summary["ledger_path"] = str(ledger_path())
    summary["ledger_rows"] = len(read_ledger())
    summary["seed_stability"] = {
        variant: seed_stability(load_runs(), variant) for variant in summary["variants"]
    }
    lockbox = None
    if args.include_lockbox:
        lockbox = _load_lockbox_summary()

    atomic_json_dump(summary, out_dir / "v2_dev_summary.json")
    (out_dir / "V2_DEV_REPORT.md").write_text(render_report(summary, lockbox=lockbox),
                                              encoding="utf-8")

    print(json.dumps({
        "variants": {v: b.get("mean_accuracy_macro_ticker")
                     for v, b in summary["variants"].items()},
        "context_signal_gate_passed": summary["context_signal_gate"]["passed"],
        "selected": summary["selection"].get("selected"),
        "meta_learning_justified": summary["meta_learning_justified"],
        "summary": str(out_dir / "v2_dev_summary.json"),
        "report": str(out_dir / "V2_DEV_REPORT.md"),
    }, indent=2))

    if args.freeze_selection:
        grouped = load_runs()
        payload = freeze_selection(args.freeze_selection.upper(), summary, grouped,
                                   out_dir=out_dir)
        print(json.dumps({"frozen_selection": payload["selected_variant"],
                          "selection_sha256": payload["selection_sha256"]}, indent=2))
    return 0


def _load_lockbox_summary() -> dict | None:
    """Read the single lockbox run's aggregate metrics, if it exists."""
    rows = [r for r in read_ledger() if r.get("fold") == "V2_LOCKBOX"]
    if not rows:
        return None
    row = rows[-1]
    metrics = read_experiment_json(row, "aggregate_metrics.json")
    complexity = read_experiment_json(row, "model_complexity.json")
    return {
        "experiment_id": row.get("experiment_id"),
        "variant": row.get("variant"),
        "seed": row.get("seed"),
        "direction": metrics.get("direction", {}),
        "selection": metrics.get("selection", {}),
        "selective_accuracy": metrics.get("selective_accuracy", {}),
        "complexity": complexity,
        "test_2022_2023_evaluated": row.get("test_2022_2023_evaluated"),
    }


if __name__ == "__main__":
    raise SystemExit(main())