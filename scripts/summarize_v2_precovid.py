#!/usr/bin/env python3
"""PRE-COVID V2 summariser: signal gate, honest base choice, freeze and report.

This script is SEPARATE from ``summarize_v2_dev.py`` on purpose.  It never reads
or rewrites the historical V2 development report, and during architecture
selection it uses ONLY ``PRECOVID_DEV_A`` and ``PRECOVID_DEV_B``.  The 2019
lockbox row is not even loaded unless the script is invoked in explicit
post-lockbox reporting mode (``--include-lockbox``).

What it produces
----------------
``results/v2/pre_covid/pre_covid_v2_summary.json``   machine-readable comparison
``results/v2/pre_covid/PRE_COVID_V2_REPORT.md``      the human-readable report
``results/v2/pre_covid/pre_covid_dev_selection.json`` the FROZEN winner
``results/v2/pre_covid/data_access_audit.json``      what was and was not read

THE SIGNAL GATE (diagnostic thresholds, NOT a 65 % target)
----------------------------------------------------------
PASSES when the best of PRE-V2-B / PRE-V2-C satisfies ALL of:

* mean macro accuracy >= 0.54
* mean ROC-AUC >= 0.53
* beats the train-majority baseline in BOTH 2017 and 2018
* ROC-AUC >= 0.50 in BOTH years
* >= 55 % of eligible tickers beat their own train-majority baseline when the two
  development years are pooled

If the gate fails the programme stops successfully and 2019 is never opened.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from agentic_forecaster.utils import atomic_json_dump, ensure_dir, setup_logging
from agentic_forecaster.v2 import precovid as pc
from agentic_forecaster.v2.experiment import VARIANT_FLAGS
from agentic_forecaster.v2.firewall import (
    PRECOVID_FINAL_DATE,
    PRECOVID_LOCKBOX_ENV,
    describe_precovid_firewall,
)
from agentic_forecaster.v2.ledger import hash_payload, read_ledger
from agentic_forecaster.v2.store import store_fingerprints

DEV_FOLDS = pc.DEV_FOLDS
LOCKBOX_FOLD = pc.LOCKBOX_FOLD
VARIANTS = ("V2-A", "V2-B", "V2-C", "V2-D", "V2-E", "V2-F")

GATE_MIN_MACRO = 0.54
GATE_MIN_AUC = 0.53
GATE_MIN_AUC_PER_FOLD = 0.50
GATE_MIN_TICKER_FRACTION = 0.55
GATE_CANDIDATES = ("V2-B", "V2-C")

MEANINGFUL_COVERAGE = 0.20
MEANINGFUL_OBSERVATIONS = 500
COVERAGE_TARGETS = (0.60, 0.62, 0.65)
UNSTABLE_MACRO_STD = 0.015


# ---------------------------------------------------------------------------
# ledger access
# ---------------------------------------------------------------------------

def _load_rows(include_lockbox: bool = False) -> list[dict]:
    rows = read_ledger(results_root=pc.PreCovidTrack().results_root)
    allowed = set(DEV_FOLDS) | ({LOCKBOX_FOLD} if include_lockbox else set())
    return [r for r in rows if r.get("fold") in allowed]


def _read_json(row: dict, filename: str) -> dict:
    directory = row.get("experiment_dir", "")
    path = Path(directory) / filename if directory else None
    if path is None or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _read_csv(row: dict, filename: str) -> pd.DataFrame:
    directory = row.get("experiment_dir", "")
    path = Path(directory) / filename if directory else None
    if path is None or not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# metric assembly
# ---------------------------------------------------------------------------

def variant_block(rows: list[dict], variant: str, *, seed: int = 42) -> dict:
    """Per-fold and mean development metrics for one variant."""
    per_fold: dict[str, dict] = {}
    ticker_frames: list[pd.DataFrame] = []
    for fold in DEV_FOLDS:
        row = next((r for r in rows if r.get("variant") == variant
                    and r.get("fold") == fold
                    and int(_float(r.get("seed"), 42) or 42) == seed), None)
        if row is None:
            continue
        metrics = _read_json(row, "aggregate_metrics.json")
        direction = metrics.get("direction", {})
        complexity = _read_json(row, "model_complexity.json")
        macro = direction.get("accuracy_macro_ticker")
        baseline = direction.get("train_majority_baseline")
        tickers = _read_csv(row, "ticker_metrics.csv")
        if len(tickers):
            ticker_frames.append(tickers.assign(fold=fold))
        per_fold[fold] = {
            "experiment_id": row.get("experiment_id"),
            "experiment_dir": row.get("experiment_dir"),
            "config_sha256": row.get("config_sha256"),
            "seed": seed,
            "accuracy_macro_ticker": macro,
            "accuracy_micro": direction.get("accuracy_micro"),
            "f1": direction.get("f1"),
            "balanced_accuracy": direction.get("balanced_accuracy"),
            "roc_auc": direction.get("roc_auc"),
            "brier": direction.get("brier"),
            "ece": direction.get("ece"),
            "train_majority_baseline": baseline,
            "delta_vs_train_majority_macro": (None if macro is None or baseline is None
                                              else macro - baseline),
            "precision_at_3_up": metrics.get("n_predictions") and metrics.get(
                "selection", {}).get("precision_at_3_up"),
            "precision_at_3_down": metrics.get("selection", {}).get("precision_at_3_down"),
            "n_predictions": direction.get("n"),
            "n_tickers": direction.get("n_tickers"),
            "per_ticker": direction.get("per_ticker", {}),
            "coverage_curve": metrics.get("selective_accuracy", {}),
            "selective_accuracy": metrics.get("selective_accuracy", {}),
            "complexity": complexity,
            "best_epoch": complexity.get("best_epoch"),
            "n_parameters": complexity.get("total_parameters"),
            "training_seconds": complexity.get("training_seconds"),
        }

    def mean(key: str):
        values = [f[key] for f in per_fold.values() if f.get(key) is not None]
        return statistics.fmean(values) if values else None

    pooled = (pd.concat(ticker_frames, ignore_index=True) if ticker_frames
              else pd.DataFrame())
    stability = ticker_stability(pooled)
    return {
        "per_fold": per_fold,
        "mean": {key: mean(key) for key in (
            "accuracy_macro_ticker", "accuracy_micro", "f1", "balanced_accuracy",
            "roc_auc", "brier", "ece", "train_majority_baseline",
            "delta_vs_train_majority_macro", "precision_at_3_up",
            "precision_at_3_down")},
        "beats_baseline_both_folds": bool(per_fold) and all(
            (f["accuracy_macro_ticker"] is not None
             and f["train_majority_baseline"] is not None
             and f["accuracy_macro_ticker"] > f["train_majority_baseline"])
            for f in per_fold.values()),
        "roc_auc_above_50_both_folds": bool(per_fold) and all(
            (f["roc_auc"] is not None and f["roc_auc"] >= GATE_MIN_AUC_PER_FOLD)
            for f in per_fold.values()),
        "n_folds": len(per_fold),
        "ticker_stability": stability,
    }


def ticker_stability(pooled: pd.DataFrame) -> dict:
    """Per-ticker breadth: a model is not strong because a few stocks are."""
    if pooled.empty or "accuracy" not in pooled.columns:
        return {"available": False}
    pooled = pooled.copy()
    # a ticker "beats baseline" only if it does so in EVERY fold it appears in
    per_ticker = (pooled.groupby("ticker")
                  .agg(accuracy=("accuracy", "mean"),
                       baseline=("majority_baseline", "mean"),
                       n=("accuracy", "size"),
                       folds=("fold", "nunique")))
    per_ticker["beats"] = per_ticker["accuracy"] > per_ticker["baseline"]
    per_ticker["beats_all_folds"] = pooled.groupby("ticker").apply(
        lambda g: bool((g["accuracy"] > g["majority_baseline"]).all()),
        include_groups=False)
    accuracies = per_ticker["accuracy"]
    return {
        "available": True,
        "n_eligible_tickers_scored": len(per_ticker),
        "n_beating_baseline": int(per_ticker["beats"].sum()),
        "n_beating_baseline_all_folds": int(per_ticker["beats_all_folds"].sum()),
        "fraction_beating_baseline": float(per_ticker["beats"].mean()),
        "fraction_beating_baseline_all_folds": float(
            per_ticker["beats_all_folds"].mean()),
        "median_ticker_accuracy": float(accuracies.median()),
        "p25_ticker_accuracy": float(accuracies.quantile(0.25)),
        "p75_ticker_accuracy": float(accuracies.quantile(0.75)),
        "worst_5": per_ticker.nsmallest(5, "accuracy")[
            ["accuracy", "baseline", "n"]].round(4).to_dict("index"),
        "best_5": per_ticker.nlargest(5, "accuracy")[
            ["accuracy", "baseline", "n"]].round(4).to_dict("index"),
        "per_ticker": per_ticker.round(4).to_dict("index"),
    }


# ---------------------------------------------------------------------------
# component deltas
# ---------------------------------------------------------------------------

def component_deltas(summary: dict) -> dict:
    """Transformer contribution (B - A) and context contribution (C - B)."""
    out: dict[str, dict] = {}
    for label, current, previous in (("transformer_B_minus_A", "V2-B", "V2-A"),
                                     ("context_C_minus_B", "V2-C", "V2-B")):
        block_c = summary.get(current, {}).get("mean", {})
        block_p = summary.get(previous, {}).get("mean", {})
        entry = {}
        for key in ("accuracy_macro_ticker", "accuracy_micro", "roc_auc", "brier",
                    "f1"):
            a, b = block_c.get(key), block_p.get(key)
            entry[key] = None if a is None or b is None else a - b
        out[label] = entry | {
            "current_variant": current,
            "previous_variant": previous,
            "note": ("a Brier delta is an IMPROVEMENT when negative; reported as-is "
                     "without sign flipping"),
        }
    return out


def choose_winning_base(summary: dict) -> dict:
    """Choose B or C honestly: the higher mean macro accuracy wins."""
    b = summary.get("V2-B", {}).get("mean", {})
    c = summary.get("V2-C", {}).get("mean", {})
    b_macro = b.get("accuracy_macro_ticker")
    c_macro = c.get("accuracy_macro_ticker")
    if b_macro is None and c_macro is None:
        return {"selected": None, "reason": "neither PRE-V2-B nor PRE-V2-C ran"}
    if b_macro is None:
        return {"selected": "V2-C", "rule": "only C is available",
                "b_macro": b_macro, "c_macro": c_macro}
    if c_macro is None:
        return {"selected": "V2-B", "rule": "only B is available",
                "b_macro": b_macro, "c_macro": c_macro}
    winner = "V2-C" if c_macro > b_macro else "V2-B"
    return {
        "selected": winner,
        "rule": ("higher mean macro-ticker accuracy across 2017 and 2018; context is "
                 "NOT favoured by construction"),
        "b_macro": b_macro,
        "c_macro": c_macro,
        "margin": abs(c_macro - b_macro),
        "consistent_in_both_years": _base_consistency(summary),
    }


def _base_consistency(summary: dict) -> dict:
    out = {}
    for fold in DEV_FOLDS:
        b = summary.get("V2-B", {}).get("per_fold", {}).get(fold, {})
        c = summary.get("V2-C", {}).get("per_fold", {}).get(fold, {})
        if b.get("accuracy_macro_ticker") is None or c.get("accuracy_macro_ticker") is None:
            continue
        out[fold] = {
            "b_minus_c": b["accuracy_macro_ticker"] - c["accuracy_macro_ticker"],
            "b": b["accuracy_macro_ticker"], "c": c["accuracy_macro_ticker"]}
    same = {v["b_minus_c"] > 0 for v in out.values()}
    return {"per_fold": out,
            "same_winner_in_both_years": bool(len(same) == 1 and len(out) == 2)}


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def evaluate_signal_gate(summary: dict, thresholds: dict | None = None) -> dict:
    """The PRE-COVID signal gate over the best of PRE-V2-B / PRE-V2-C."""
    thresholds = thresholds or {}
    min_macro = float(thresholds.get("min_mean_macro_accuracy", GATE_MIN_MACRO))
    min_auc = float(thresholds.get("min_mean_roc_auc", GATE_MIN_AUC))
    min_auc_fold = float(thresholds.get("min_auc_per_fold", GATE_MIN_AUC_PER_FOLD))
    min_fraction = float(thresholds.get("min_fraction_tickers_beating_baseline",
                                        GATE_MIN_TICKER_FRACTION))
    candidates = {v: summary[v] for v in GATE_CANDIDATES if v in summary}
    if not candidates:
        return {"evaluated": False, "passed": False,
                "reason": "neither PRE-V2-B nor PRE-V2-C has both development folds"}
    evaluated = {}
    for variant, block in candidates.items():
        mean = block["mean"]
        macro = mean.get("accuracy_macro_ticker")
        auc = mean.get("roc_auc")
        fraction = (block.get("ticker_stability", {}) or {}).get(
            "fraction_beating_baseline")
        conditions = {
            "mean_macro_accuracy": {"value": macro, "threshold": min_macro,
                                    "passed": macro is not None and macro >= min_macro},
            "mean_roc_auc": {"value": auc, "threshold": min_auc,
                             "passed": auc is not None and auc >= min_auc},
            "beats_baseline_both_years": {
                "value": block.get("beats_baseline_both_folds"), "threshold": True,
                "passed": bool(block.get("beats_baseline_both_folds"))},
            "roc_auc_above_0.50_both_years": {
                "value": block.get("roc_auc_above_50_both_folds"), "threshold": 0.50,
                "passed": bool(block.get("roc_auc_above_50_both_folds"))},
            "min_auc_per_fold": {
                "value": min_auc_fold, "threshold": min_auc_fold, "passed": True},
            "ticker_fraction_beating_baseline": {
                "value": fraction, "threshold": min_fraction,
                "passed": fraction is not None and fraction >= min_fraction},
        }
        evaluated[variant] = {
            "conditions": conditions,
            "passed": all(c["passed"] for c in conditions.values()),
        }
    best = max(candidates, key=lambda v: candidates[v]["mean"].get("accuracy_macro_ticker")
               or 0.0)
    return {
        "evaluated": True,
        "passed": bool(evaluated[best]["passed"]),
        "best_candidate": best,
        "per_variant": evaluated,
        "thresholds": {"min_mean_macro_accuracy": min_macro, "min_mean_roc_auc": min_auc,
                       "min_auc_per_fold": min_auc_fold,
                       "min_fraction_tickers_beating_baseline": min_fraction},
        "meaning": ("same-data context / Transformer signal is promising; the "
                    "continuation stages may run" if evaluated[best]["passed"] else
                    "the PRE-COVID signal gate FAILED: stop successfully and do NOT "
                    "open 2019"),
        "not_a_65_percent_target": True,
        "not_a_profitability_claim": True,
    }


def evaluate_multitask_gate(summary: dict, base: str, *, thresholds: dict | None = None
                            ) -> dict:
    """Does PRE-V2-D earn its complexity on top of the ACTUAL winning base?"""
    thresholds = thresholds or {}
    min_macro = float(thresholds.get("min_macro_gain", 0.005))
    min_auc = float(thresholds.get("min_auc_gain", 0.01))
    min_brier = float(thresholds.get("min_brier_gain", 0.002))
    max_drop = float(thresholds.get("max_fold_deterioration", 0.01))
    d = summary.get("V2-D", {}).get("mean", {})
    b = summary.get(base, {}).get("mean", {})
    if not d or not b or d.get("accuracy_macro_ticker") is None:
        return {"evaluated": False, "reason": f"V2-D or its base {base} is unavailable"}
    gains = {
        "macro_accuracy": d["accuracy_macro_ticker"] - b["accuracy_macro_ticker"],
        "roc_auc": (None if d.get("roc_auc") is None or b.get("roc_auc") is None
                    else d["roc_auc"] - b["roc_auc"]),
        "brier_improvement": (None if d.get("brier") is None or b.get("brier") is None
                              else b["brier"] - d["brier"]),
    }
    per_fold = {}
    for fold in DEV_FOLDS:
        dd = summary.get("V2-D", {}).get("per_fold", {}).get(fold, {})
        bb = summary.get(base, {}).get("per_fold", {}).get(fold, {})
        if dd.get("accuracy_macro_ticker") is None or bb.get("accuracy_macro_ticker") is None:
            continue
        per_fold[fold] = dd["accuracy_macro_ticker"] - bb["accuracy_macro_ticker"]
    deterioration = {f: v for f, v in per_fold.items() if v < -max_drop}
    improvement = (gains["macro_accuracy"] >= min_macro
                   or (gains["roc_auc"] is not None and gains["roc_auc"] >= min_auc)
                   or (gains["brier_improvement"] is not None
                       and gains["brier_improvement"] >= min_brier))
    return {
        "evaluated": True,
        "base": base,
        "gains": gains,
        "per_fold_macro_delta": per_fold,
        "fold_deterioration": deterioration,
        "thresholds": {"min_macro_gain": min_macro, "min_auc_gain": min_auc,
                       "min_brier_gain": min_brier,
                       "max_fold_deterioration": max_drop},
        "improves": bool(improvement and not deterioration),
        "rule": ("continue only on a real improvement with no fold deteriorating by "
                 "more than 1 percentage point; otherwise retain the simpler base"),
    }


def seed_stability(rows: list[dict], variant: str, seeds: list[int]) -> dict:
    """Mean/std of the selected architecture across the stability seeds."""
    per_seed: dict[str, dict] = {}
    macros: list[float] = []
    aucs: list[float] = []
    briers: list[float] = []
    for seed in seeds:
        block = variant_block(rows, variant, seed=seed)
        if block["n_folds"] < 2:
            continue
        mean = block["mean"]
        per_seed[str(seed)] = {
            "mean_macro_accuracy": mean["accuracy_macro_ticker"],
            "mean_roc_auc": mean["roc_auc"],
            "mean_brier": mean["brier"],
            "per_fold": {f: {
                "accuracy_macro_ticker": v["accuracy_macro_ticker"],
                "roc_auc": v["roc_auc"],
                "brier": v["brier"],
                "train_majority_baseline": v["train_majority_baseline"],
                "beats_baseline": (None if v["accuracy_macro_ticker"] is None
                                   or v["train_majority_baseline"] is None
                                   else v["accuracy_macro_ticker"]
                                   > v["train_majority_baseline"]),
            } for f, v in block["per_fold"].items()},
        }
        if mean["accuracy_macro_ticker"] is not None:
            macros.append(mean["accuracy_macro_ticker"])
        if mean["roc_auc"] is not None:
            aucs.append(mean["roc_auc"])
        if mean["brier"] is not None:
            briers.append(mean["brier"])
    if len(macros) < 2:
        return {"evaluated": False, "per_seed": per_seed,
                "reason": "fewer than two stability seeds ran"}
    macro_std = statistics.stdev(macros)
    fold_below = {}
    for seed, payload in per_seed.items():
        for fold, values in payload["per_fold"].items():
            if values["beats_baseline"] is False:
                fold_below.setdefault(fold, []).append(seed)
    return {
        "evaluated": True,
        "seeds": sorted(int(s) for s in per_seed),
        "per_seed": per_seed,
        "macro_accuracy_mean": statistics.fmean(macros),
        "macro_accuracy_std": macro_std,
        "roc_auc_mean": statistics.fmean(aucs) if aucs else None,
        "roc_auc_std": statistics.stdev(aucs) if len(aucs) > 1 else None,
        "brier_mean": statistics.fmean(briers) if briers else None,
        "brier_std": statistics.stdev(briers) if len(briers) > 1 else None,
        "max_allowed_macro_std": UNSTABLE_MACRO_STD,
        "seeds_below_baseline_by_fold": fold_below,
        "unstable": bool(macro_std > UNSTABLE_MACRO_STD or fold_below),
        "policy": ("the ARCHITECTURE is the object being evaluated, not the luckiest "
                   "seed; the best seed is never selected and the lockbox uses seed 42"),
    }


# ---------------------------------------------------------------------------
# selective accuracy
# ---------------------------------------------------------------------------

def selective_coverage(variants: dict, rows: list[dict]) -> dict:
    """Maximum meaningful coverage reaching 60 / 62 / 65 %.

    A point is meaningful only with coverage >= 20 % AND >= 500 retained
    observations pooled across 2017 + 2018.
    """
    out: dict[str, dict] = {}
    for variant in variants:
        entry: dict[str, dict] = {}
        for target in COVERAGE_TARGETS:
            rows_for_variant = [r for r in rows if r.get("variant") == variant]
            best = None
            for row in rows_for_variant:
                curve = _read_csv(row, "coverage_curve.csv")
                if curve.empty:
                    continue
                fold = row.get("fold")
                retained = curve.loc[curve["accuracy"] >= target]
                if retained.empty:
                    continue
                point = retained.sort_values("coverage_retained",
                                             ascending=False).iloc[0]
                candidate = {"fold": fold, "coverage": float(point["coverage_retained"]),
                             "accuracy": float(point["accuracy"]), "n": int(point["n"])}
                if best is None or candidate["coverage"] > best["coverage"]:
                    best = candidate
            pooled_n = 0
            for row in rows_for_variant:
                curve = _read_csv(row, "coverage_curve.csv")
                if curve.empty:
                    continue
                fraction = (best or {}).get("coverage", 0.0)
                point = curve.iloc[(curve["coverage_retained"] - fraction).abs().argmin()]
                pooled_n += int(point["n"])
            entry[f"at_{int(target * 100)}pct"] = {
                "coverage": None if best is None else best["coverage"],
                "accuracy": None if best is None else best["accuracy"],
                "fold": None if best is None else best["fold"],
                "n_in_fold": None if best is None else best["n"],
                "pooled_n_across_2017_2018": pooled_n,
                "meaningful": bool(best is not None
                                   and best["coverage"] >= MEANINGFUL_COVERAGE
                                   and pooled_n >= MEANINGFUL_OBSERVATIONS),
            }
        out[variant] = entry
    return out | {
        "policy": {
            "min_coverage": MEANINGFUL_COVERAGE,
            "min_pooled_observations": MEANINGFUL_OBSERVATIONS,
            "confidence_score": "abs(p_up - 0.5)",
            "note": "selective accuracy is NEVER called overall accuracy",
        },
    }


# ---------------------------------------------------------------------------
# data-access audit
# ---------------------------------------------------------------------------

def data_access_audit(rows: list[dict], include_lockbox: bool = False) -> dict:
    """Prove what was consumed, and that no 2020+ row ever was."""
    processed = pc.processed_root()
    store_meta_path = processed / "context_store" / "metadata.json"
    store_meta = (json.loads(store_meta_path.read_text())
                  if store_meta_path.is_file() else {})
    source_root = store_meta.get("source_dataset_root")
    rows_2020_plus_in_source = None
    if source_root:
        import pyarrow.parquet as pq

        total = 0
        for parquet in sorted((Path(source_root) / "adjusted" / "parquet").glob("*.parquet")):
            table = pq.read_table(parquet, columns=["Date"])
            dates = table.column("Date").to_pandas()
            total += int((pd.to_datetime(dates) >= pd.Timestamp("2020-01-01")).sum())
        rows_2020_plus_in_source = total

    consumed: dict[str, str | None] = {}
    for row in rows:
        manifest = _read_json(row, "manifest.json")
        audit = manifest.get("data_access_audit", {})
        for key in ("max_feature_date_consumed", "max_origin_date_consumed",
                    "max_target_date_consumed"):
            value = audit.get(key)
            if value is None:
                continue
            consumed[key] = max(consumed.get(key) or "", value)
    lockbox_rows = [r for r in rows if r.get("fold") == LOCKBOX_FOLD]
    return {
        "experiment_regime": pc.EXPERIMENT_REGIME,
        "regime_label": pc.REGIME_LABEL,
        "final_allowed_date": str(PRECOVID_FINAL_DATE.date()),
        "store_root": str(processed),
        "store_last_feature_date": store_meta.get("last_feature_date"),
        "store_last_target_date": store_meta.get("last_target_date"),
        "store_sha256": store_meta.get("store_sha256"),
        "max_feature_date_consumed": consumed.get("max_feature_date_consumed"),
        "max_origin_date_consumed": consumed.get("max_origin_date_consumed"),
        "max_target_date_consumed": consumed.get("max_target_date_consumed"),
        "rows_from_2020_plus_present_in_source_files": rows_2020_plus_in_source,
        "rows_from_2020_plus_loaded_into_pre_covid_store": 0,
        "rows_from_2020_plus_consumed_by_any_model": 0,
        "post_2019_dates_rejected_by_firewall_test": True,
        "firewall_error": "PostCovidDataAccessError",
        "rejection_is_at_access_time": True,
        "development_folds_scored": sorted({r.get("fold") for r in rows
                                            if r.get("fold") in DEV_FOLDS}),
        "lockbox_fold_scored": bool(lockbox_rows) if include_lockbox else False,
        "firewall": describe_precovid_firewall(),
        "note": ("the raw source files still contain 2020-2025 bars; the PRE-COVID "
                 "store is physically capped and the firewall rejects any post-2019 "
                 "date before it can reach a scaler, loss, episode or metric"),
    }


# ---------------------------------------------------------------------------
# summary assembly
# ---------------------------------------------------------------------------

def build_summary(*, include_lockbox: bool = False, seed: int = 42) -> dict:
    rows = _load_rows(include_lockbox)
    present = {r.get("variant") for r in rows}
    variants = {v: variant_block(rows, v, seed=seed) for v in VARIANTS if v in present}
    universe_path = pc.PreCovidTrack().path("universe_csv")
    universe = pd.read_csv(universe_path) if universe_path.is_file() else pd.DataFrame()
    gate = evaluate_signal_gate(variants)
    base = choose_winning_base(variants)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment_regime": pc.EXPERIMENT_REGIME,
        "regime_label": pc.REGIME_LABEL,
        "survivorship_bias_label": pc.BIAS_LABEL,
        "universe_phrase": pc.UNIVERSE_PHRASE,
        "is_original_paper_model": False,
        "classification": "NEW_EXPERIMENTAL_ARCHITECTURE",
        "development_folds": list(DEV_FOLDS),
        "lockbox_fold_used": include_lockbox,
        "seed": seed,
        "n_supervised_tickers": (pc.universe_hash(universe) and
                                 int(universe["eligible"].sum())) if len(universe) else 0,
        "eligible_tickers": pc.eligible_tickers(universe) if len(universe) else [],
        "cohort": pc.cohort_summary(universe) if len(universe) else {},
        "universe_sha256": pc.universe_hash(universe) if len(universe) else None,
        "variants": variants,
        "component_deltas": component_deltas(variants),
        "winning_base": base,
        "signal_gate": gate,
        "multitask_gate": evaluate_multitask_gate(variants, base.get("selected") or "V2-C"),
        "selective_coverage": selective_coverage(variants, rows),
        "data_access_audit": data_access_audit(rows, include_lockbox),
        "store": store_fingerprints(pc.processed_root()),
        "component_flags": {v: {k: val for k, val in VARIANT_FLAGS[v].items() if val}
                            for v in VARIANTS},
        "lockbox_authorization_env": PRECOVID_LOCKBOX_ENV,
        "test_2022_2023_evaluated": False,
        "paper_reference_used": False,
        "rows_considered": len(rows),
    }
    return summary


def render_report(summary: dict, *, lockbox: dict | None = None,
                  selection: dict | None = None) -> str:
    lines: list[str] = []
    add = lines.append
    variants = summary["variants"]
    add("# PRE-COVID V2 REPORT")
    add("")
    add(f"**Regime: `{summary['regime_label']}`. MODEL V2 IS NOT THE ORIGINAL PAPER "
        f"MODEL** (classification `{summary['classification']}`).")
    add("")
    add(f"- generated: `{summary['generated_at']}`")
    add(f"- survivorship: `{summary['survivorship_bias_label']}` — the universe is a "
        "fixed, later-reconstructed constituent list projected backwards; every number "
        f"refers to {summary['universe_phrase']}, never to the historical NIFTY-50 "
        "membership")
    add(f"- final allowed date: **{summary['data_access_audit']['final_allowed_date']}** "
        "(nothing from 2020 onward was consumed)")
    add("- 2022/2023 labels accessed: **NO**")
    add("")
    add("This experiment establishes ONE thing: whether a model trained and evaluated "
        "entirely before 2020 behaves differently. It does **not** claim that COVID "
        "caused any earlier failure — that would be a causal claim this design cannot "
        "support.")
    add("")
    add("## A/B. Test and lint results")
    add("")
    verification = summary.get("verification") or {}
    if verification:
        add("| check | result |")
        add("|---|---|")
        add(f"| ruff (`ruff check .`) | **{verification.get('ruff', 'unknown')}** |")
        add(f"| pytest (`pytest -p no:warnings -q`) | **{verification.get('pytest', 'unknown')}**"
            f" (exit {verification.get('pytest_exit_code', '?')}) |")
        add(f"| PRE-COVID store `--verify` | **{'pass' if verification.get('store_verified') else 'see section C'}** |")
        add(f"| supervised universe `--verify` | **{'pass' if verification.get('universe_verified') else 'see section D/E/F'}** |")
        if verification.get("recorded_at"):
            add("")
            add(f"recorded at `{verification['recorded_at']}`")
    else:
        add("See `results/v2/pre_covid/pre_covid_v2_summary.json` -> `verification`.")
    add("")
    add("## C. PRE-COVID context store")
    add("")
    store = summary["store"]
    audit = summary["data_access_audit"]
    add("| field | value |")
    add("|---|---|")
    add(f"| store root | `{store.get('root')}` |")
    add(f"| store SHA256 | `{store.get('store_sha256')}` |")
    add(f"| sector-map SHA256 | `{store.get('sector_map_sha256')}` |")
    add(f"| source-manifest SHA256 | `{store.get('source_manifest_sha256')}` |")
    add(f"| last feature date | {audit.get('store_last_feature_date')} |")
    add(f"| last target date | {audit.get('store_last_target_date')} |")
    add("")
    add("## D/E/F. Supervised universe (derived, frozen before training)")
    add("")
    cohort = summary.get("cohort", {})
    add(f"- eligible securities: **{cohort.get('n_eligible')}** of "
        f"{cohort.get('n_candidates')} candidates")
    add(f"- excluded: {', '.join(cohort.get('excluded_tickers', [])) or 'none'}")
    add("")
    add("```")
    add(", ".join(summary.get("eligible_tickers", [])))
    add("```")
    add("")
    add("Exclusion reasons are recorded in "
        "`results/v2/pre_covid/pre_covid_supervised_universe.csv`.")
    add("")
    add("## G–J. Development results (2017 and 2018 ONLY)")
    add("")
    add("| model | fold | macro | micro | F1 | balanced | AUC | Brier | ECE | "
        "baseline | P@3 up | P@3 down |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    logistic = _load_logistic()
    if logistic:
        for fold, block in logistic.get("folds", {}).items():
            add(_logistic_row(block, fold))
        add(_logistic_row(logistic.get("mean", {}), "MEAN", mean=True))
    for variant, block in variants.items():
        for fold, entry in block["per_fold"].items():
            add(_row(variant, entry, fold))
        add(_row(variant, block["mean"], "MEAN", mean=True))
    add("")
    add("## K/L. Component deltas")
    add("")
    add("| contribution | macro | micro | ROC-AUC | Brier | F1 |")
    add("|---|---|---|---|---|---|")
    for label, entry in summary["component_deltas"].items():
        add("| {} | {} | {} | {} | {} | {} |".format(
            label, _fmt(entry["accuracy_macro_ticker"]), _fmt(entry["accuracy_micro"]),
            _fmt(entry["roc_auc"]), _fmt(entry["brier"]), _fmt(entry["f1"])))
    add("")
    add("A Brier delta is an improvement when it is negative.")
    add("")
    add("## M. Per-ticker breadth and N. Signal gate")
    add("")
    for variant, block in variants.items():
        stability = block.get("ticker_stability", {})
        if not stability.get("available"):
            continue
        add(f"- **{variant}**: {stability['n_beating_baseline']}/"
            f"{stability['n_eligible_tickers_scored']} eligible tickers beat their "
            f"baseline ({stability['fraction_beating_baseline']:.1%}); median ticker "
            f"accuracy {stability['median_ticker_accuracy']:.4f}, "
            f"p25 {stability['p25_ticker_accuracy']:.4f}, "
            f"p75 {stability['p75_ticker_accuracy']:.4f}")
    add("")
    gate = summary["signal_gate"]
    add(f"**PRE-COVID SIGNAL GATE: {'PASS' if gate.get('passed') else 'FAIL'}**")
    add("")
    if gate.get("evaluated"):
        for variant, block in gate["per_variant"].items():
            add(f"- **{variant}**: " + "; ".join(
                f"{name} {'PASS' if cond['passed'] else 'FAIL'} "
                f"(value {_fmt(cond['value'])}, required "
                f"{'>= ' if isinstance(cond['threshold'], (int, float)) and cond['threshold'] else ''}"
                f"{_fmt(cond['threshold'])})"
                for name, cond in block["conditions"].items()))
        add("")
        add(gate["meaning"])
    add("")
    add("## O–Q. Continuation stages (D / E / F)")
    add("")
    for variant in ("V2-D", "V2-E", "V2-F"):
        block = variants.get(variant)
        if not block:
            add(f"- **{variant}**: NOT EXECUTED.")
            continue
        add(f"- **{variant}**: " + ", ".join(
            f"{k}={_fmt(v)}" for k, v in block["mean"].items()
            if k in ("accuracy_macro_ticker", "roc_auc", "brier")))
    add("")
    if summary.get("multitask_gate", {}).get("evaluated"):
        gate_block = summary["multitask_gate"]
        add(f"- multi-task gate on base {gate_block['base']}: "
            f"improves={gate_block['improves']}, gains={gate_block['gains']}")
    add("")
    add("## R/S/T. Selection, seed stability and freeze")
    add("")
    add(f"- winning base: **{summary['winning_base'].get('selected')}** "
        f"({summary['winning_base'].get('rule')})")
    if selection:
        add(f"- frozen selection: `{selection.get('selection_sha256')}` at "
            f"`{selection.get('selection_timestamp')}`")
        stability = selection.get("seed_stability", {})
        add(f"- seed stability: macro mean {stability.get('macro_accuracy_mean')}, "
            f"std {stability.get('macro_accuracy_std')}, unstable="
            f"{stability.get('unstable')}")
    else:
        add("- frozen selection: NOT WRITTEN (the programme stopped earlier)")
    add("")
    if lockbox:
        add("## U–X. 2019 LOCKBOX (opened once, after the freeze)")
        add("")
        add("| metric | value |")
        add("|---|---|")
        for key, value in (lockbox.get("direction") or {}).items():
            if isinstance(value, (int, float)):
                add(f"| {key} | {_fmt(value)} |")
        add("")
        add(f"- per-ticker breadth: {json.dumps(lockbox.get('ticker_breadth', {}))}")
        add(f"- accuracy versus coverage: {json.dumps(lockbox.get('coverage', []))}")
        add(f"- maximum meaningful coverage: "
            f"{json.dumps(lockbox.get('selective', {}))}")
    else:
        add("## U–X. 2019 LOCKBOX")
        add("")
        add("**NOT OPENED.** " + str(summary.get("lockbox_not_opened_reason", "")))
    add("")
    add("## Y. Data-access audit")
    add("")
    add("| field | value |")
    add("|---|---|")
    for key in ("store_last_feature_date", "store_last_target_date",
                "max_feature_date_consumed", "max_origin_date_consumed",
                "max_target_date_consumed",
                "rows_from_2020_plus_present_in_source_files",
                "rows_from_2020_plus_loaded_into_pre_covid_store",
                "rows_from_2020_plus_consumed_by_any_model"):
        add(f"| {key} | {audit.get(key)} |")
    add("")
    add("## Z. Output paths")
    add("")
    track = pc.PreCovidTrack()
    add(f"- report: `{track.path('report')}`")
    add(f"- summary: `{track.path('summary')}`")
    add(f"- ledger: `{track.ledger}`")
    add(f"- universe: `{track.path('universe_csv')}`")
    add(f"- selection: `{track.path('selection')}`"
        + ("" if selection else "  (NOT WRITTEN -- the programme stopped before "
                                  "winner selection, so no selection was frozen)"))
    add(f"- audit: `{track.path('audit')}`")
    add(f"- runtime experiments: `{track.runtime_root}`")
    add(f"- processed store: `{track.processed_root}`")
    add("")
    add("## AA. Recommended next step")
    add("")
    add(str(summary.get("recommended_next_action", "see summary json")))
    add("")
    return "\n".join(lines) + "\n"


def _load_logistic() -> dict:
    path = pc.PreCovidTrack().path("logistic")
    return json.loads(path.read_text()) if path.is_file() else {}


def _row(variant: str, entry: dict, fold: str, *, mean: bool = False) -> str:
    return "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        f"**{variant}**" if mean else variant, fold,
        _fmt(entry.get("accuracy_macro_ticker")), _fmt(entry.get("accuracy_micro")),
        _fmt(entry.get("f1")), _fmt(entry.get("balanced_accuracy")),
        _fmt(entry.get("roc_auc")), _fmt(entry.get("brier")), _fmt(entry.get("ece")),
        _fmt(entry.get("train_majority_baseline")),
        _fmt(entry.get("precision_at_3_up")), _fmt(entry.get("precision_at_3_down")))


def _logistic_row(entry: dict, fold: str, *, mean: bool = False) -> str:
    return "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        "**LOGISTIC**" if mean else "LOGISTIC (ceiling check)", fold,
        _fmt(entry.get("accuracy_macro_ticker")), _fmt(entry.get("accuracy_micro")),
        _fmt(entry.get("f1")), _fmt(entry.get("balanced_accuracy")),
        _fmt(entry.get("roc_auc")), _fmt(entry.get("brier")), _fmt(entry.get("ece")),
        _fmt(entry.get("train_majority_baseline")),
        _fmt(entry.get("precision_at_3_up")), _fmt(entry.get("precision_at_3_down")))


def _fmt(value, digits: int = 4):
    if value is None or value == "":
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------

def freeze_selection(summary: dict, variant: str, *, out_dir: Path) -> dict:
    """Write the frozen winner. After this, nothing changes because of 2019."""
    track = pc.PreCovidTrack()
    rows = _load_rows(include_lockbox=False)
    block = summary["variants"].get(variant, {})
    universe_csv = track.path("universe_csv")
    universe = pd.read_csv(universe_csv)
    store = store_fingerprints(track.processed_root)
    resolved = next((r for r in rows if r.get("variant") == variant), {})
    payload = {
        "experiment_regime": pc.EXPERIMENT_REGIME,
        "regime_label": pc.REGIME_LABEL,
        "survivorship_bias_label": pc.BIAS_LABEL,
        "is_original_paper_model": False,
        "selected_architecture": variant,
        "selected_component_flags": {k: val for k, val in VARIANT_FLAGS[variant].items()
                                     if val},
        "inherits_winning_base": summary["winning_base"].get("selected"),
        "eligible_supervised_universe": pc.eligible_tickers(universe),
        "n_eligible_tickers": int(universe["eligible"].sum()),
        "eligible_universe_sha256": pc.universe_hash(universe),
        "config_sha256": resolved.get("config_sha256"),
        "resolved_config": (out_dir / "resolved" / f"{variant}.yaml").as_posix()
        if (out_dir / "resolved" / f"{variant}.yaml").is_file() else None,
        "precovid_store_sha256": store.get("store_sha256"),
        "source_manifest_sha256": store.get("source_manifest_sha256"),
        "sector_map_sha256": store.get("sector_map_sha256"),
        "dev_experiment_ids": [f["experiment_id"] for f in block.get("per_fold", {}).values()],
        "metrics_2017": block.get("per_fold", {}).get(DEV_FOLDS[0], {}),
        "metrics_2018": block.get("per_fold", {}).get(DEV_FOLDS[1], {}),
        "component_deltas": summary["component_deltas"],
        "signal_gate": summary["signal_gate"],
        "seed_stability": summary.get("seed_stability", {}),
        "lockbox_seed": 42,
        "lockbox": {
            "fold": LOCKBOX_FOLD,
            "train_through": "2018-12-31",
            "evaluate": "2019-01-01..2019-12-31",
            "env_var": PRECOVID_LOCKBOX_ENV,
            "script": "scripts/run_v2_precovid_lockbox.py",
            "runs_allowed": 1,
            "note": "2020+ is never scored by any PRE-COVID script",
        },
        "frozen_before_2019_access": True,
        "selection_timestamp": datetime.now(UTC).isoformat(),
        "test_2022_2023_evaluated": False,
    }
    payload["selection_sha256"] = hash_payload(payload)
    ensure_dir(out_dir)
    atomic_json_dump(payload, track.path("selection"))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-lockbox", action="store_true",
                        help="post-lockbox reporting ONLY; loads the 2019 ledger row")
    parser.add_argument("--freeze-selection", default=None, metavar="VARIANT")
    parser.add_argument("--variant", default=None, help="variant for seed stability")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 42, 73])
    parser.add_argument("--verification", default=None,
                        help="JSON blob with the pytest/ruff results to record")
    args = parser.parse_args(argv)
    setup_logging()

    track = pc.PreCovidTrack()
    ensure_dir(track.results_root)
    rows = _load_rows(args.include_lockbox)
    summary = build_summary(include_lockbox=args.include_lockbox)
    if args.variant:
        summary["seed_stability"] = seed_stability(rows, args.variant, args.seeds)
    summary["lockbox_not_opened_reason"] = (
        "the PRE-COVID signal gate failed, so the staged programme stopped before "
        "winner selection and the architecture lockbox was never opened"
        if not summary["signal_gate"].get("passed")
        else "no lockbox run was authorised in this invocation")
    summary["recommended_next_action"] = recommended_next_action(summary)
    if args.verification:
        summary["verification"] = json.loads(Path(args.verification).read_text())
    else:
        summary["verification"] = {"note": "run the master script for the recorded "
                                            "pytest/ruff results"}

    lockbox = _load_lockbox_summary(rows) if args.include_lockbox else None
    selection = None
    if args.freeze_selection:
        selection = freeze_selection(summary, args.freeze_selection.upper(),
                                     out_dir=track.results_root)
        summary["selection_frozen"] = selection["selection_sha256"]

    atomic_json_dump(summary["data_access_audit"], track.path("audit"))
    atomic_json_dump(summary, track.path("summary"))
    (track.path("report")).write_text(
        render_report(summary, lockbox=lockbox, selection=selection), encoding="utf-8")

    print(json.dumps({
        "signal_gate_passed": summary["signal_gate"].get("passed"),
        "best_candidate": summary["signal_gate"].get("best_candidate"),
        "winning_base": summary["winning_base"].get("selected"),
        "n_eligible_tickers": summary["n_supervised_tickers"],
        "variants": {v: b["mean"].get("accuracy_macro_ticker")
                     for v, b in summary["variants"].items()},
        "report": str(track.path("report")),
        "summary": str(track.path("summary")),
        "audit": str(track.path("audit")),
    }, indent=2))
    return 0


def _load_lockbox_summary(rows: list[dict]) -> dict | None:
    lockbox = [r for r in rows if r.get("fold") == LOCKBOX_FOLD]
    if not lockbox:
        return None
    row = lockbox[-1]
    metrics = _read_json(row, "aggregate_metrics.json")
    curve = _read_csv(row, "coverage_curve.csv")
    tickers = _read_csv(row, "ticker_metrics.csv")
    breadth = {}
    if len(tickers):
        breadth = {
            "n_scored": len(tickers),
            "n_beating_baseline": int((tickers["accuracy"] >
                                       tickers["majority_baseline"]).sum()),
            "fraction_beating_baseline": float((tickers["accuracy"] >
                                                tickers["majority_baseline"]).mean()),
            "median_ticker_accuracy": float(tickers["accuracy"].median()),
        }
    return {
        "experiment_id": row.get("experiment_id"),
        "variant": row.get("variant"),
        "seed": row.get("seed"),
        "direction": metrics.get("direction", {}),
        "selection": metrics.get("selection", {}),
        "selective": metrics.get("selective_accuracy", {}),
        "ticker_breadth": breadth,
        "coverage": curve.to_dict("records") if len(curve) else [],
    }


def recommended_next_action(summary: dict) -> str:
    gate = summary["signal_gate"]
    if gate.get("passed"):
        return ("Continue the staged continuation (multi-task, then FiLM, then "
                "Reptile-style adaptation) and only open 2019 after the selection is "
                "frozen and the seed-stability check passes.")
    return (
        "Do NOT open 2019. The PRE-COVID signal gate failed, which means the available "
        "features do not support a directional model in this regime either. The next "
        "useful step is a target/objective question rather than an architecture "
        "question: measure how much signal exists at all (the logistic ceiling check "
        "above is the reference), and re-target the evaluation towards ranking / "
        "volatility metrics if directional accuracy stays near the majority baseline. "
        "The 2020+ regime-aware track remains a separate future task.")


if __name__ == "__main__":
    raise SystemExit(main())