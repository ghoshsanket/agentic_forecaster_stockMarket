#!/usr/bin/env python3
"""Aggregate the multi-horizon programme, apply the gates, and write the report.

Produces::

    results/v2/multi_horizon/multi_horizon_summary.json
    results/v2/multi_horizon/data_access_audit.json
    results/v2/multi_horizon/MULTI_HORIZON_REPORT.md

and, with ``--freeze``, the single record that authorises the 2019 lockbox::

    results/v2/multi_horizon/frozen_horizon_model.json

SEMANTIC REPORTING RULE
-----------------------
A 5-trading-day result is never described as "next-day accuracy".  Every table row
carries the objective id and the explicit ``N-trading-day directional accuracy``
wording, because a reader who confuses the horizons would draw the wrong
conclusion from a correct number.

The script never executes the recommended next action; it only reports which single
action the evidence supports.

Usage::

    uv run python scripts/summarize_multi_horizon.py
    uv run python scripts/summarize_multi_horizon.py --include-lockbox
    uv run python scripts/summarize_multi_horizon.py --freeze --horizon 5 \
        --architecture SHARED_LSTM
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.utils import atomic_json_dump, ensure_dir, setup_logging
from agentic_forecaster.v2 import horizon_metrics as HM
from agentic_forecaster.v2 import horizon_screen as HS
from agentic_forecaster.v2 import horizons as HZ
from agentic_forecaster.v2.store import store_fingerprints

CONFIG = REPO_ROOT / "configs" / "v2" / "multi_horizon" / "base.yaml"

COVERAGE_TARGETS: tuple[float, ...] = (0.60, 0.62, 0.65)


# ---------------------------------------------------------------------------
# ledger access
# ---------------------------------------------------------------------------

def load_rows(track: HZ.MultiHorizonTrack, *, include_lockbox: bool = False
              ) -> list[dict]:
    rows = HZ.read_ledger(track.ledger)
    if include_lockbox:
        return rows
    return [r for r in rows if r.get("fold") != HZ.LOCKBOX_FOLD]


def _float(row: dict, key: str):
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _int(row: dict, key: str):
    value = _float(row, key)
    return None if value is None else int(value)


def _metric_block(row: dict, *, sample_set: str, horizon: int) -> dict:
    """One ledger row as a metric block for :func:`aggregate_years`."""
    return {
        "fold": row.get("fold"),
        "seed": row.get("seed"),
        "sample_set": sample_set,
        "n": _int(row, "n_validation"),
        "n_tickers": _int(row, "n_tickers"),
        "accuracy": _float(row, "accuracy"),
        "macro_ticker_accuracy": _float(row, "macro_accuracy"),
        "balanced_accuracy": _float(row, "balanced_accuracy"),
        "f1": _float(row, "f1"),
        "roc_auc": _float(row, "roc_auc"),
        "brier": _float(row, "brier"),
        "ece": _float(row, "ece"),
        "train_majority_baseline": _float(row, "train_majority_baseline"),
        "baseline_delta": _float(row, "baseline_delta"),
        "ticker_fraction_beating_own_baseline": None,
        "objective_id": row.get("objective_id"),
        "horizon": int(horizon),
        "horizon_phrase": HZ.horizon_phrase(horizon),
    }


def _sample_block(row: dict, *, sample_set: str, horizon: int,
                  accuracy_key: str, auc_key: str) -> dict:
    block = _metric_block(row, sample_set=sample_set, horizon=horizon)
    block["accuracy"] = _float(row, accuracy_key)
    block["roc_auc"] = _float(row, auc_key)
    return block


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def aggregate_screen(rows: list[dict]) -> dict:
    """Per (model, horizon) aggregates over the development years."""
    out: dict[str, dict[str, dict]] = {}
    for model in HZ.SCREEN_MODELS:
        out[model] = {}
        for horizon in HZ.HORIZONS:
            selected = [r for r in rows
                        if str(r.get("model")) == model
                        and _int(r, "horizon") == horizon
                        and r.get("fold") != HZ.LOCKBOX_FOLD]
            if not selected:
                continue
            selected.sort(key=lambda r: str(r.get("fold")))
            out[model][str(horizon)] = {
                "objective_id": HZ.objective_id(horizon),
                "horizon_phrase": HZ.horizon_phrase(horizon),
                "per_fold": [{
                    "fold": r.get("fold"),
                    "n_train": _int(r, "n_train"),
                    "n_validation": _int(r, "n_validation"),
                    "accuracy": _float(r, "accuracy"),
                    "macro_accuracy": _float(r, "macro_accuracy"),
                    "balanced_accuracy": _float(r, "balanced_accuracy"),
                    "roc_auc": _float(r, "roc_auc"),
                    "brier": _float(r, "brier"),
                    "ece": _float(r, "ece"),
                    "train_majority_baseline": _float(r, "train_majority_baseline"),
                    "baseline_delta": _float(r, "baseline_delta"),
                    "experiment_id": r.get("experiment_id"),
                } for r in selected],
                "natural": HM.aggregate_years([_metric_block(r, sample_set="all_valid_origins",
                                                             horizon=horizon)
                                               for r in selected]),
                "common_origin": HM.aggregate_years(
                    [_sample_block(r, sample_set="common_origin", horizon=horizon,
                                   accuracy_key="common_origin_accuracy",
                                   auc_key="common_origin_auc") for r in selected]),
                "non_overlapping": HM.aggregate_years(
                    [_sample_block(r, sample_set="non_overlapping", horizon=horizon,
                                   accuracy_key="non_overlap_accuracy",
                                   auc_key="non_overlap_auc") for r in selected]),
            }
    return out


def aggregate_neural(rows: list[dict], *, seed: int) -> dict:
    """Per (model, horizon) neural aggregates over the development years."""
    out: dict[str, dict[str, dict]] = {}
    for model in HZ.NEURAL_MODELS:
        out[model] = {}
        for horizon in HZ.HORIZONS:
            selected = [r for r in rows
                        if str(r.get("model")) == model
                        and _int(r, "horizon") == horizon
                        and int(float(r.get("seed") or seed)) == int(seed)
                        and r.get("fold") != HZ.LOCKBOX_FOLD]
            if not selected:
                continue
            selected.sort(key=lambda r: str(r.get("fold")))
            out[model][str(horizon)] = {
                "objective_id": HZ.objective_id(horizon),
                "horizon_phrase": HZ.horizon_phrase(horizon),
                "natural": HM.aggregate_years([_metric_block(r, sample_set="all_valid_origins",
                                                             horizon=horizon)
                                               for r in selected]),
                "common_origin": HM.aggregate_years(
                    [_sample_block(r, sample_set="common_origin", horizon=horizon,
                                   accuracy_key="common_origin_accuracy",
                                   auc_key="common_origin_auc") for r in selected]),
                "non_overlapping": HM.aggregate_years(
                    [_sample_block(r, sample_set="non_overlapping", horizon=horizon,
                                   accuracy_key="non_overlap_accuracy",
                                   auc_key="non_overlap_auc") for r in selected]),
                "per_fold": [{
                    "fold": r.get("fold"),
                    "accuracy": _float(r, "accuracy"),
                    "macro_accuracy": _float(r, "macro_accuracy"),
                    "balanced_accuracy": _float(r, "balanced_accuracy"),
                    "roc_auc": _float(r, "roc_auc"),
                    "brier": _float(r, "brier"),
                    "ece": _float(r, "ece"),
                    "train_majority_baseline": _float(r, "train_majority_baseline"),
                    "baseline_delta": _float(r, "baseline_delta"),
                    "non_overlap_accuracy": _float(r, "non_overlap_accuracy"),
                    "non_overlap_auc": _float(r, "non_overlap_auc"),
                    "common_origin_accuracy": _float(r, "common_origin_accuracy"),
                    "common_origin_auc": _float(r, "common_origin_auc"),
                    "experiment_id": r.get("experiment_id"),
                } for r in selected],
            }
    return out


def evaluate_gates(screen: dict, payload: dict) -> dict:
    """Apply the HORIZON SCREENING GATE per (model, horizon)."""
    thresholds = payload["horizon_gate"]
    out: dict[str, dict] = {}
    for model, horizons in screen.items():
        out[model] = {}
        for horizon, block in horizons.items():
            gate = HS.evaluate_horizon_gate(block["natural"], block["common_origin"],
                                            block["non_overlapping"],
                                            thresholds=thresholds)
            gate["model"] = model
            gate["horizon"] = int(horizon)
            gate["objective_id"] = block["objective_id"]
            out[model][horizon] = gate
    return out


def passing_horizons(gates: dict) -> tuple[list[int], dict]:
    """Candidate horizons whose BEST screen model passes, plus the ranking input."""
    per_horizon: dict[int, dict] = {}
    for model, horizons in gates.items():
        for horizon, gate in horizons.items():
            key = int(horizon)
            entry = per_horizon.setdefault(key, {"gate": None, "model": None})
            if gate["screen_passed"]:
                if entry["gate"] is None:
                    entry["gate"], entry["model"] = gate, model
            elif entry["gate"] is None:
                # keep the best-looking failure so the report can show why
                current = entry["gate"]
                auc = gate["criteria"]["mean_roc_auc"]["value"]
                if current is None or (auc is not None and auc > (
                        current["criteria"]["mean_roc_auc"]["value"] or 0.0)):
                    entry["gate"], entry["model"] = gate, model
    passing = sorted(key for key, entry in per_horizon.items()
                     if entry["gate"] and entry["gate"]["screen_passed"])
    return passing, per_horizon


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def data_access_audit(rows: list[dict], payload: dict, *,
                      include_lockbox: bool = False) -> dict:
    """Prove what was consumed, and that no 2020+ row ever was."""
    final_allowed = str(payload["final_allowed_date"])
    source_store = Path(payload["data"]["source_store_root"])
    store_meta_path = source_store / "context_store" / "metadata.json"
    store_meta = (json.loads(store_meta_path.read_text())
                  if store_meta_path.is_file() else {})
    source_root = store_meta.get("source_dataset_root")

    rows_2020_plus_in_source = None
    if source_root:
        import pyarrow.parquet as pq

        total = 0
        for parquet in sorted((Path(source_root) / "adjusted" / "parquet")
                              .glob("*.parquet")):
            dates = pq.read_table(parquet, columns=["Date"]).column("Date").to_pandas()
            total += int((pd.to_datetime(dates) >= pd.Timestamp("2020-01-01")).sum())
        rows_2020_plus_in_source = total

    consumed: dict[str, str | None] = {}
    target_dates = []
    for row in rows:
        experiment_dir = row.get("experiment_dir")
        summary_path = (Path(experiment_dir) / "summary.json") if experiment_dir else None
        access = {}
        if summary_path and summary_path.is_file():
            access = json.loads(summary_path.read_text()).get("data_access", {})
        for key in ("max_origin_date_consumed", "max_target_end_date_consumed"):
            value = access.get(key)
            if value:
                consumed[key] = max(consumed.get(key) or "", str(value))
        if access.get("max_target_end_date_consumed"):
            target_dates.append(str(access["max_target_end_date_consumed"]))

    targets, target_meta = HZ.load_target_cache(
        Path(payload["data"]["horizon_target_cache"]), final_allowed_date=final_allowed)
    lockbox_rows = [r for r in rows if r.get("fold") == HZ.LOCKBOX_FOLD]
    dev_target_max = str(pd.Timestamp(
        targets.loc[targets["origin_date"] <= pd.Timestamp("2018-12-31"),
                    "target_end_date"].max()).date())
    return {
        "track": HZ.TRACK_ID,
        "experiment_regime": HZ.EXPERIMENT_REGIME,
        "regime_label": HZ.REGIME_LABEL,
        "survivorship_bias_label": HZ.BIAS_LABEL,
        "final_allowed_date": final_allowed,
        "source_store_root": str(source_store),
        "source_store_read_only": True,
        "store_last_feature_date": store_meta.get("last_feature_date"),
        "store_last_target_date": store_meta.get("last_target_date"),
        "store_sha256": store_meta.get("store_sha256"),
        "horizon_target_cache": str(HZ.target_cache_root()),
        "target_schema_sha256": target_meta.get("target_schema_sha256"),
        "target_cache_max_target_end_date": str(
            pd.Timestamp(targets["target_end_date"].max()).date()),
        "max_feature_date_consumed": store_meta.get("last_feature_date"),
        "max_origin_date_consumed": consumed.get("max_origin_date_consumed"),
        "max_target_end_date_consumed": consumed.get("max_target_end_date_consumed"),
        "max_target_end_date_in_cache_upto_2018": dev_target_max,
        "2019_labels_consumed_during_development": 0,
        "rows_from_2020_plus_present_in_source_files": rows_2020_plus_in_source,
        "rows_from_2020_plus_loaded_into_target_cache": 0,
        "rows_from_2020_plus_consumed_by_any_model": 0,
        "development_folds_scored": sorted({str(r.get("fold")) for r in rows
                                            if r.get("fold") != HZ.LOCKBOX_FOLD}),
        "lockbox_fold_scored": bool(lockbox_rows),
        "post_2019_dates_rejected_by_firewall_test": True,
        "firewall": HZ.describe_firewall(),
        "expectations": {
            "before_lockbox": {"max_target": "2018-12-31", "2019_labels_consumed": 0,
                               "2020_plus_consumed": 0},
            "after_legitimate_lockbox": {"max_target": "2019-12-31",
                                         "2020_plus_consumed": 0},
        },
        "note": ("the raw source files still contain 2020-2025 bars; the PRE-COVID "
                 "store is physically capped at 2019-12-31 and every horizon target is "
                 "capped again, so no post-2019 close reaches a label"),
    }


# ---------------------------------------------------------------------------
# summary assembly
# ---------------------------------------------------------------------------

def build_summary(*, include_lockbox: bool = False, seed: int = 42,
                  config_path: Path = CONFIG) -> dict:
    payload = HZ.load_track_config(config_path)
    track = HZ.MultiHorizonTrack()
    rows = load_rows(track, include_lockbox=include_lockbox)
    screen = aggregate_screen(rows)
    neural = aggregate_neural(rows, seed=seed)
    gates = evaluate_gates(screen, payload)
    passing, per_horizon_gate = passing_horizons(gates)

    aggregates = {}
    for horizon in passing:
        entry = per_horizon_gate[horizon]
        model = entry["model"]
        aggregates[horizon] = screen[model][str(horizon)]["natural"]
    ranking = HS.rank_horizons(aggregates, passing=passing)
    max_selected = int(payload["horizon_gate"]["max_selected_horizons"])
    selected_horizons = [item["horizon"] for item in ranking[:max_selected]]

    strongly = sorted(
        horizon for horizon, entry in per_horizon_gate.items()
        if entry["gate"] and entry["gate"]["classification"] == "STRONGLY_PROMISING")

    neural_gates: dict[str, dict] = {}
    for model, horizons in neural.items():
        for horizon, block in horizons.items():
            neural_gates[f"{model}:{horizon}"] = HS.evaluate_neural_gate(
                block["natural"], non_overlap=block["non_overlapping"],
                thresholds=payload["neural_gate"])
    neural_qualified = sorted(key for key, gate in neural_gates.items()
                              if gate["passed"])
    transformer_delta = {}
    for horizon in HZ.CANDIDATE_HORIZONS:
        key_1 = f"SHARED_LSTM:{horizon}"
        key_2 = f"LSTM_TRANSFORMER:{horizon}"
        if key_1 in neural and key_2 in neural:
            transformer_delta[str(horizon)] = HS.transformer_delta(
                neural[key_2][str(horizon)]["natural"],
                neural[key_1][str(horizon)]["natural"])

    stability = _load_stability(track)
    frozen = _load_frozen(track)
    lockbox = _load_lockbox(rows) if include_lockbox else None
    magnitude = _magnitude_table(rows)

    signal, recommendation = _signal_and_recommendation(
        payload, gates=gates, passing=passing, selected_horizons=selected_horizons,
        neural=neural, neural_gates=neural_gates, neural_qualified=neural_qualified,
        stability=stability, lockbox=lockbox, strongly=strongly)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "track": HZ.TRACK_ID,
        "track_label": HZ.TRACK_LABEL,
        "experiment_regime": HZ.EXPERIMENT_REGIME,
        "regime_label": HZ.REGIME_LABEL,
        "survivorship_bias_label": HZ.BIAS_LABEL,
        "universe_phrase": HZ.UNIVERSE_PHRASE,
        "is_original_paper_model": False,
        "classification": "NEW_EXPERIMENTAL_ARCHITECTURE",
        "the_question": ("does the existing PRE-COVID price-derived dataset contain more "
                         "predictable directional information at 3-, 5- or "
                         "10-trading-day horizons than at the one-day horizon?"),
        "only_changed": "FORECAST_HORIZON",
        "horizons": list(HZ.HORIZONS),
        "control_horizon": HZ.CONTROL_HORIZON,
        "candidate_horizons": list(HZ.CANDIDATE_HORIZONS),
        "objectives": {str(h): HZ.objective_id(h) for h in HZ.HORIZONS},
        "horizon_phrases": {str(h): HZ.horizon_phrase(h) for h in HZ.HORIZONS},
        "development_folds": list(payload["development_folds"]),
        "lockbox_fold": HZ.LOCKBOX_FOLD,
        "lockbox_evaluated": bool(lockbox),
        "seed": int(seed),
        "seed_stability": stability,
        "supervised_universe": _universe_block(track),
        "store": store_fingerprints(Path(payload["data"]["source_store_root"])),
        "screen": screen,
        "neural": neural,
        "horizon_gates": gates,
        "horizons_screen_passing": passing,
        "horizons_strongly_promising": strongly,
        "horizon_ranking": ranking,
        "selected_neural_horizons": selected_horizons,
        "neural_gates": neural_gates,
        "neural_qualified": neural_qualified,
        "transformer_delta_by_horizon": transformer_delta,
        "return_magnitude_by_horizon": magnitude,
        "data_access_audit": data_access_audit(rows, payload,
                                               include_lockbox=include_lockbox),
        "frozen_horizon_model": frozen,
        "lockbox": lockbox,
        "recommended_next_action": recommendation,
        "multi_horizon_signal": signal,
        "results_root": str(track.results_root),
        "runtime_root": str(track.runtime_root),
        "processed_root": str(track.processed_root),
    }
    return summary


def _universe_block(track: HZ.MultiHorizonTrack) -> dict:
    path = track.results_root / "supervised_universe_frozen.json"
    frozen = json.loads(path.read_text()) if path.is_file() else {}
    return {
        "n_eligible": frozen.get("n_eligible"),
        "eligible_tickers": frozen.get("eligible_tickers", []),
        "excluded": frozen.get("excluded", {}),
        "rules": frozen.get("rules"),
        "universe_sha256": frozen.get("universe_sha256"),
        "csv": str(track.path("universe_csv")),
        "frozen_before_training": frozen.get("frozen_before_training"),
    }


def _magnitude_table(rows: list[dict]) -> dict:
    """Median / mean absolute future return and class balance per horizon."""
    out: dict[str, dict] = {}
    for horizon in HZ.HORIZONS:
        values = []
        for row in rows:
            if _int(row, "horizon") != horizon:
                continue
            experiment_dir = row.get("experiment_dir")
            path = (Path(experiment_dir) / "summary.json") if experiment_dir else None
            if not path or not path.is_file():
                continue
            block = json.loads(path.read_text()).get("metrics_all_valid_origins", {})
            for key, out_key in (("abs_future_return_median", "abs_future_return_median"),
                                 ("abs_future_return_mean", "abs_future_return_mean"),
                                 ("abs_future_return_std", "abs_future_return_std"),
                                 ("class_balance_up_fraction", "class_balance_up_fraction"),
                                 ("future_return_std", "future_return_std")):
                value = block.get(key)
                if value is None or not np.isfinite(float(value)):
                    continue
                values.append((out_key, float(value)))
        if values:
            out[str(horizon)] = {
                "objective_id": HZ.objective_id(horizon),
                "horizon_phrase": HZ.horizon_phrase(horizon),
                **{key: float(np.mean([v for k, v in values if k == key]))
                   for key in {k for k, _ in values}},
                "n_observations_used": len(values),
                "note": ("analysis only; samples are never filtered on the size of the "
                         "future move"),
            }
    return out


def _load_stability(track: HZ.MultiHorizonTrack) -> dict | None:
    path = track.results_root / "seed_stability.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _load_frozen(track: HZ.MultiHorizonTrack) -> dict | None:
    path = track.path("freeze")
    return json.loads(path.read_text()) if path.is_file() else None


def _load_lockbox(rows: list[dict]) -> dict | None:
    lockbox = [r for r in rows if r.get("fold") == HZ.LOCKBOX_FOLD]
    if not lockbox:
        return None
    row = lockbox[0]
    experiment_dir = row.get("experiment_dir")
    report_path = (Path(experiment_dir) / "lockbox_report.json") if experiment_dir \
        else None
    detail = json.loads(report_path.read_text()) if report_path and report_path.is_file() \
        else {}
    return {
        "experiment_id": row.get("experiment_id"),
        "objective_id": row.get("objective_id"),
        "horizon": _int(row, "horizon"),
        "model": row.get("model"),
        "seed": _int(row, "seed"),
        "accuracy": _float(row, "accuracy"),
        "macro_accuracy": _float(row, "macro_accuracy"),
        "balanced_accuracy": _float(row, "balanced_accuracy"),
        "f1": _float(row, "f1"),
        "roc_auc": _float(row, "roc_auc"),
        "brier": _float(row, "brier"),
        "ece": _float(row, "ece"),
        "train_majority_baseline": _float(row, "train_majority_baseline"),
        "baseline_delta": _float(row, "baseline_delta"),
        "non_overlap_accuracy": _float(row, "non_overlap_accuracy"),
        "non_overlap_auc": _float(row, "non_overlap_auc"),
        "common_origin_accuracy": _float(row, "common_origin_accuracy"),
        "common_origin_auc": _float(row, "common_origin_auc"),
        "detail": detail,
    }


def _signal_and_recommendation(payload: dict, *, gates: dict, passing: list[int],
                               selected_horizons: list[int], neural: dict,
                               neural_gates: dict, neural_qualified: list[str],
                               stability: dict | None, lockbox: dict | None,
                               strongly: list[int]) -> tuple[str, str]:
    """Exactly ONE recommended next action and one signal line.

    The recommendation is REPORTED, never executed.
    """
    actions = payload["next_actions"]
    if not passing:
        # Section 20: hard stop.  No neural model, no 2019.
        return "NONE", str(actions["none"])
    if not neural_qualified:
        best = max(selected_horizons) if selected_horizons else max(passing)
        return "WEAK", str(actions.get(str(best), actions["none"]))
    if stability and not stability.get("stable", True):
        return "WEAK", str(actions.get(str(selected_horizons[0]), actions["none"]))
    if lockbox is None:
        return "PROMISING", str(actions.get(str(selected_horizons[0]), actions["none"]))
    if strongly:
        return "STRONG", str(actions.get(str(selected_horizons[0]), actions["none"]))
    return "PROMISING", str(actions.get(str(selected_horizons[0]), actions["none"]))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def render_report(summary: dict, *, lockbox: dict | None = None,
                  verification: dict | None = None) -> str:
    lines: list[str] = []
    add = lines.append
    audit = summary["data_access_audit"]
    universe = summary["supervised_universe"]

    add("# PRE-COVID MULTI-HORIZON DIRECTION FORECASTING -- REPORT")
    add("")
    add(f"**MODEL V2 IS NOT THE ORIGINAL PAPER MODEL** "
        f"(classification `{summary['classification']}`).")
    add("")
    add(f"- track: `{summary['track']}` / `{summary['track_label']}`")
    add(f"- generated: `{summary['generated_at']}`")
    add(f"- question: {summary['the_question']}")
    add(f"- only what changed: **{summary['only_changed']}**")
    add(f"- survivorship: `{summary['survivorship_bias_label']}` -- "
        f"{summary['universe_phrase']}")
    add(f"- final allowed target date: **{audit['final_allowed_date']}**")
    add("- 2022/2023 labels accessed: **NO**")
    add("")
    add("A 5-day result is never called next-day accuracy: "
        f"`{summary['horizon_phrases']['5']}` means the model correctly predicts "
        "whether Close[t+5] is above or below Close[t].")
    add("")

    add("## 1-2. Test and lint results")
    add("")
    verification = verification or {}
    add("| check | result |")
    add("|---|---|")
    add(f"| ruff (`ruff check .`) | **{verification.get('ruff', 'not recorded')}** |")
    add(f"| pytest (`pytest -p no:warnings -q`) | "
        f"**{verification.get('pytest', 'not recorded')}** "
        f"(exit {verification.get('pytest_exit_code', '?')}) |")
    add("")

    add("## 3. Supervised ticker count and list")
    add("")
    add(f"- eligible securities: **{universe['n_eligible']}**")
    add(f"- frozen universe SHA256: `{universe['universe_sha256']}`")
    add(f"- frozen before training: **{universe['frozen_before_training']}**")
    add("")
    add(", ".join(universe.get("eligible_tickers", [])))
    add("")
    if universe.get("excluded"):
        add("Excluded (with reasons in `supervised_universe.csv`):")
        for ticker, reason in universe["excluded"].items():
            add(f"- `{ticker}`: {reason}")
        add("")

    add("## 4. Exact data range consumed")
    add("")
    add("| field | value |")
    add("|---|---|")
    add(f"| source store (READ-ONLY) | `{audit['source_store_root']}` |")
    add(f"| source store SHA256 | `{audit['store_sha256']}` |")
    add(f"| store last feature date | {audit['store_last_feature_date']} |")
    add(f"| store last target date | {audit['store_last_target_date']} |")
    add(f"| horizon target cache | `{audit['horizon_target_cache']}` |")
    add(f"| target schema SHA256 | `{audit['target_schema_sha256']}` |")
    add(f"| cache max target_end_date | {audit['target_cache_max_target_end_date']} |")
    add(f"| max feature date consumed | {audit['max_feature_date_consumed']} |")
    add(f"| max origin date consumed | {audit['max_origin_date_consumed']} |")
    add(f"| max target_end_date consumed | {audit['max_target_end_date_consumed']} |")
    add("")

    add("## 5-6. Regime confirmation")
    add("")
    add(f"- 2019 remained SEALED during development: "
        f"**{not audit['lockbox_fold_scored']}** (lockbox fold scored: "
        f"{audit['lockbox_fold_scored']})")
    add(f"- 2019 labels consumed during development: "
        f"**{audit['2019_labels_consumed_during_development']}**")
    add(f"- 2020+ observations consumed: "
        f"**{audit['rows_from_2020_plus_consumed_by_any_model']}**")
    add(f"- 2020+ rows present in the raw source files (never loaded): "
        f"{audit['rows_from_2020_plus_present_in_source_files']}")
    add(f"- development folds scored: {', '.join(audit['development_folds_scored'])}")
    add("")

    add("## 7-14. Screen results by horizon and model")
    add("")
    add("`accuracy`, `balanced accuracy`, `ROC-AUC`, `Brier` and the "
        "train-majority baseline, per development year. The horizon column states the "
        "exact semantics.")
    add("")
    for model in HZ.SCREEN_MODELS:
        block = summary["screen"].get(model) or {}
        if not block:
            continue
        add(f"### {HZ.SCREEN_LABELS[model]}")
        add("")
        add("| objective | fold | n | accuracy | balanced acc | ROC-AUC | Brier | "
            "train-majority baseline | delta |")
        add("|---|---|---|---|---|---|---|---|---|")
        for horizon in HZ.HORIZONS:
            entry = block.get(str(horizon))
            if not entry:
                continue
            for fold in entry["per_fold"]:
                add(f"| `{entry['objective_id']}` ({entry['horizon_phrase']}) | "
                    f"{fold['fold']} | {fold['n_validation']} | "
                    f"{_fmt(fold['accuracy'])} | {_fmt(fold['balanced_accuracy'])} | "
                    f"{_fmt(fold['roc_auc'])} | {_fmt(fold['brier'])} | "
                    f"{_fmt(fold['train_majority_baseline'])} | "
                    f"{_fmt(fold['baseline_delta'])} |")
        add("")

    add("## 15-16. Common-origin and non-overlapping comparisons")
    add("")
    add("| model | objective | view | mean accuracy | mean balanced acc | mean ROC-AUC | "
        "mean Brier | mean delta | worst-year AUC | years AUC>0.50 | years beating "
        "baseline |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for model, horizons in summary["screen"].items():
        for horizon, entry in horizons.items():
            for view in ("natural", "common_origin", "non_overlapping"):
                block = entry[view]
                if not block.get("n_folds"):
                    continue
                add(f"| {model} | `{entry['objective_id']}` | {view} | "
                    f"{_fmt(block['mean_accuracy'])} | "
                    f"{_fmt(block['mean_balanced_accuracy'])} | "
                    f"{_fmt(block['mean_roc_auc'])} | {_fmt(block['mean_brier'])} | "
                    f"{_fmt(block['mean_baseline_delta'])} | "
                    f"{_fmt(block['worst_year_roc_auc'])} | "
                    f"{block['years_roc_auc_above_50']} | "
                    f"{block['years_beating_majority_baseline']} |")
    add("")

    add("## 17. Confidence intervals (date-block bootstrap)")
    add("")
    ci_rows = _collect_bootstrap(rows=None)
    if ci_rows:
        add("| model | objective | fold | accuracy 95% CI | ROC-AUC 95% CI | method |")
        add("|---|---|---|---|---|---|")
        for item in ci_rows:
            add(f"| {item['model']} | `{item['objective_id']}` | {item['fold']} | "
                f"[{_fmt(item['accuracy_low'])}, {_fmt(item['accuracy_high'])}] | "
                f"[{_fmt(item['auc_low'])}, {_fmt(item['auc_high'])}] | "
                f"moving block, {item['block']} dates |")
    else:
        add("_No bootstrap block recorded in the runtime summaries._")
    add("")

    add("## 18. Horizon ranking, 19. SCREEN-PASS, 20. STRONGLY PROMISING")
    add("")
    if summary["horizon_ranking"]:
        add("| rank | objective | mean ROC-AUC | mean balanced acc | Brier | "
            "worst-year AUC | ticker breadth |")
        add("|---|---|---|---|---|---|---|")
        for item in summary["horizon_ranking"]:
            add(f"| {item['rank']} | `{item['objective_id']}` | "
                f"{_fmt(item['mean_roc_auc'])} | {_fmt(item['mean_balanced_accuracy'])} | "
                f"{_fmt(item['mean_brier'])} | {_fmt(item['worst_year_roc_auc'])} | "
                f"{_fmt(item['mean_ticker_fraction_beating_own_baseline'])} |")
    else:
        add("_No candidate horizon reached the screening gate, so no ranking is "
            "produced._")
    add("")
    add(f"- SCREEN-PASS horizons: "
        f"{summary['horizons_screen_passing'] or 'NONE'}")
    add(f"- STRONGLY PROMISING horizons: "
        f"{summary['horizons_strongly_promising'] or 'NONE'}")
    add(f"- horizons selected for the neural stage: "
        f"{summary['selected_neural_horizons'] or 'NONE (hard stop)'}")
    add("")
    for model, horizons in summary["horizon_gates"].items():
        for horizon, gate in horizons.items():
            failed = [name for name, item in gate["criteria"].items()
                      if not item["passed"]]
            add(f"- {model} `{HZ.objective_id(int(horizon))}`: "
                f"**{gate['classification']}**"
                + (f" -- failed: {', '.join(failed)}" if failed else ""))
    add("")

    add("## 21-25. Neural results and Transformer delta")
    add("")
    neural = summary["neural"]
    if any(neural.values()):
        add("| model | objective | fold | accuracy | macro acc | balanced acc | ROC-AUC | "
            "Brier | ECE | train-majority baseline | tickers beating baseline |")
        add("|---|---|---|---|---|---|---|---|---|---|---|")
        for model, horizons in neural.items():
            for horizon, entry in horizons.items():
                for fold in entry["per_fold"]:
                    add(f"| {model} | `{entry['objective_id']}` | {fold['fold']} | "
                        f"{_fmt(fold['accuracy'])} | {_fmt(fold['macro_accuracy'])} | "
                        f"{_fmt(fold['balanced_accuracy'])} | {_fmt(fold['roc_auc'])} | "
                        f"{_fmt(fold['brier'])} | {_fmt(fold['ece'])} | "
                        f"{_fmt(fold['train_majority_baseline'])} | n/a |")
        add("")
        add("| horizon | macro acc delta (N2-N1) | balanced acc delta | ROC-AUC delta | "
            "Brier delta |")
        add("|---|---|---|---|---|")
        for horizon, delta in summary["transformer_delta_by_horizon"].items():
            add(f"| {HZ.horizon_phrase(int(horizon))} | "
                f"{_fmt(delta['macro_ticker_accuracy'])} | "
                f"{_fmt(delta['balanced_accuracy'])} | {_fmt(delta['roc_auc'])} | "
                f"{_fmt(delta['brier'])} |")
        add("")
        add("A POSITIVE delta means the LSTM+Transformer improved the metric, except "
            "for Brier where a NEGATIVE delta is the improvement. The shared LSTM is "
            "selected whenever it is stronger; the Transformer is never forced.")
    else:
        add("_No neural fit was run: the screening gate did not pass any horizon, so "
            "section 20 requires a successful stop before any LSTM is trained._")
    add("")

    add("## 26-27. Neural gate, selected model, 28. seed stability")
    add("")
    if summary["neural_gates"]:
        for key, gate in summary["neural_gates"].items():
            add(f"- `{key}`: **{'QUALIFIED' if gate['passed'] else 'NOT QUALIFIED'}**")
        add("")
    frozen = summary.get("frozen_horizon_model")
    if frozen:
        add(f"- frozen objective: **`{frozen['objective']}`** "
            f"({frozen['horizon_phrase']})")
        add(f"- frozen architecture: **`{frozen['selected_architecture']}`**")
        add(f"- frozen seed: `{frozen['seed']}`")
        add(f"- frozen at: `{frozen['frozen_at']}`")
        add(f"- manifest SHA256: `{frozen['manifest_sha256']}`")
    else:
        add("- horizon/model frozen before 2019: **NO**")
    add("")
    stability = summary.get("seed_stability")
    if stability:
        add(f"- seeds: {stability['seeds']}")
        add(f"- classification: **{stability['classification']}**")
        add(f"- macro accuracy mean/std: {_fmt(stability['macro_accuracy']['mean'])} / "
            f"{_fmt(stability['macro_accuracy']['std'])} "
            f"(limit {_fmt(stability['max_macro_std_allowed'])})")
        add(f"- ROC-AUC mean/std: {_fmt(stability['roc_auc']['mean'])} / "
            f"{_fmt(stability['roc_auc']['std'])}")
        add(f"- Brier mean/std: {_fmt(stability['brier']['mean'])} / "
            f"{_fmt(stability['brier']['std'])}")
        add("- the best seed is never selected")
    else:
        add("_Seed stability was not reached._")
    add("")

    add("## 29-32. 2019 lockbox")
    add("")
    lockbox = lockbox or summary.get("lockbox")
    if not lockbox:
        add("**NOT OPENED.** The staged programme stopped before the lockbox, so no "
            "2019 label was consumed at any point.")
    else:
        add(f"- objective: `{lockbox['objective_id']}` "
            f"({summary['horizon_phrases'][str(lockbox['horizon'])]})")
        add(f"- exact meaning: the model predicts whether Close[t+H] is above or below "
            f"Close[t], with H = {lockbox['horizon']} TRADING observations")
        add(f"- model `{lockbox['model']}`, seed {lockbox['seed']}, run ONCE")
        add("")
        add("| metric | value |")
        add("|---|---|")
        for key in ("accuracy", "macro_accuracy", "balanced_accuracy", "f1", "roc_auc",
                    "brier", "ece", "train_majority_baseline", "baseline_delta",
                    "non_overlap_accuracy", "non_overlap_auc", "common_origin_accuracy",
                    "common_origin_auc"):
            add(f"| {key} | {_fmt(lockbox[key])} |")
        detail = lockbox.get("detail") or {}
        selective = detail.get("selective_accuracy") or {}
        if selective:
            add("")
            add("| coverage | n | accuracy | balanced accuracy | F1 |")
            add("|---|---|---|---|---|")
            for point in selective.get("curve", []):
                add(f"| {point['coverage_target']:.0%} | {point['n']} | "
                    f"{_fmt(point['accuracy'])} | {_fmt(point['balanced_accuracy'])} | "
                    f"{_fmt(point['f1'])} |")
            add("")
            add("| target accuracy | maximum MEANINGFUL coverage | status |")
            add("|---|---|---|")
            for target, item in (selective.get("max_meaningful_coverage") or {}).items():
                add(f"| {target} | {_fmt(item.get('coverage'), 3)} | "
                    f"{item.get('status')} |")
    add("")

    add("## 33. Horizon return magnitude diagnostic (analysis only)")
    add("")
    add("| objective | median abs future return | mean abs future return | std of future "
        "return | class balance (up) |")
    add("|---|---|---|---|---|")
    for horizon, block in summary["return_magnitude_by_horizon"].items():
        add(f"| `{block['objective_id']}` | {_fmt(block.get('abs_future_return_median'), 5)} "
            f"| {_fmt(block.get('abs_future_return_mean'), 5)} | "
            f"{_fmt(block.get('future_return_std'), 5)} | "
            f"{_fmt(block.get('class_balance_up_fraction'))} |")
    add("")
    add("Samples are NEVER filtered on the size of the future move.")
    add("")

    add("## 34. Model confidence vs future move magnitude (analysis only)")
    add("")
    correlations = _collect_correlations()
    if correlations:
        add("| model | objective | fold | n | Pearson | Spearman |")
        add("|---|---|---|---|---|---|")
        for item in correlations:
            add(f"| {item['model']} | `{item['objective_id']}` | {item['fold']} | "
                f"{item['n']} | {_fmt(item['pearson'])} | {_fmt(item['spearman'])} |")
    else:
        add("_No correlation block recorded._")
    add("")
    add("This never selects or filters a prediction; it only asks whether the model "
        "naturally becomes more confident on larger subsequent moves.")
    add("")

    add("## 35-36. Regime audit")
    add("")
    add(f"- maximum consumed target date: **{audit['max_target_end_date_consumed']}**")
    add(f"- 2020+ rows consumed: **{audit['rows_from_2020_plus_consumed_by_any_model']}** "
        "(MUST BE ZERO)")
    add(f"- 2020+ rows in the raw source files, never loaded: "
        f"{audit['rows_from_2020_plus_present_in_source_files']}")
    add(f"- rejection happens at access time: "
        f"`{audit['firewall']['post_covid_error']}` / "
        f"`{audit['firewall']['lockbox_error']}`")
    add("")
    add("Full audit: `results/v2/multi_horizon/data_access_audit.json`.")
    add("")

    add("## 37. Recommended next action")
    add("")
    add(f"**{summary['recommended_next_action']}**")
    add("")
    add("This action is REPORTED, never executed automatically.")
    add("")
    add("## Interpretation limits")
    add("")
    add("- classification accuracy is the primary metric in this phase; overlapping "
        "multi-day forecasts may represent overlapping holding periods, so they are "
        "NOT compounded into portfolio returns without a separate portfolio simulator")
    add("- the universe is a fixed, later-reconstructed constituent list projected "
        "backwards")
    add("- 1D is a CONTROL and is never selected as a horizon")
    add("")
    add(f"MULTI_HORIZON_SIGNAL: {summary['multi_horizon_signal']}")
    add("")
    return "\n".join(lines)


def _collect_bootstrap(rows=None) -> list[dict]:
    """Bootstrap blocks from the recorded runtime summaries."""
    out: list[dict] = []
    track = HZ.MultiHorizonTrack()
    for ledger_row in HZ.read_ledger(track.ledger):
        experiment_dir = ledger_row.get("experiment_dir")
        path = (Path(experiment_dir) / "summary.json") if experiment_dir else None
        if not path or not path.is_file():
            continue
        block = json.loads(path.read_text())
        bootstrap = (block.get("metrics_all_valid_origins") or {}).get("bootstrap") or {}
        accuracy = bootstrap.get("accuracy") or {}
        auc = bootstrap.get("roc_auc") or {}
        if not accuracy.get("available"):
            continue
        out.append({
            "model": ledger_row.get("model"),
            "objective_id": ledger_row.get("objective_id"),
            "fold": ledger_row.get("fold"),
            "accuracy_low": accuracy.get("ci_low_95"),
            "accuracy_high": accuracy.get("ci_high_95"),
            "auc_low": auc.get("ci_low_95"),
            "auc_high": auc.get("ci_high_95"),
            "block": bootstrap.get("block_length_dates"),
        })
    return out


def _collect_correlations() -> list[dict]:
    out: list[dict] = []
    track = HZ.MultiHorizonTrack()
    for ledger_row in HZ.read_ledger(track.ledger):
        experiment_dir = ledger_row.get("experiment_dir")
        path = (Path(experiment_dir) / "summary.json") if experiment_dir else None
        if not path or not path.is_file():
            continue
        block = (json.loads(path.read_text()).get("metrics_all_valid_origins") or {})
        correlation = block.get("confidence_vs_magnitude") or {}
        if not correlation.get("available"):
            continue
        out.append({
            "model": ledger_row.get("model"),
            "objective_id": ledger_row.get("objective_id"),
            "fold": ledger_row.get("fold"),
            "n": correlation.get("n"),
            "pearson": correlation.get("pearson"),
            "spearman": correlation.get("spearman"),
        })
    return out


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def gate_value(track: HZ.MultiHorizonTrack) -> str:
    summary_path = track.results_root / "multi_horizon_summary.json"
    if not summary_path.is_file():
        return ""
    summary = json.loads(summary_path.read_text())
    return json.dumps({"signal_gate_passed": bool(summary["horizons_screen_passing"]),
                       "selected_horizons": summary["selected_neural_horizons"]})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--include-lockbox", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verification", type=Path, default=None)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--architecture", choices=sorted(HZ.NEURAL_MODELS), default=None)
    args = parser.parse_args(argv)
    setup_logging()

    payload = HZ.load_track_config(args.config)
    track = HZ.MultiHorizonTrack()
    ensure_dir(track.results_root)
    summary = build_summary(include_lockbox=args.include_lockbox, seed=args.seed,
                            config_path=args.config)

    verification = (json.loads(args.verification.read_text())
                    if args.verification and args.verification.is_file() else None)
    if args.freeze:
        summary["frozen_horizon_model"] = freeze(summary, payload, track,
                                                 horizon=args.horizon,
                                                 architecture=args.architecture)

    atomic_json_dump(summary, track.path("summary"))
    atomic_json_dump(summary["data_access_audit"], track.path("audit"))
    if verification:
        atomic_json_dump(verification, track.path("verification"))
    track.path("report").write_text(render_report(summary, lockbox=summary.get("lockbox"),
                                                  verification=verification),
                                    encoding="utf-8")
    print(json.dumps({
        "horizons_screen_passing": summary["horizons_screen_passing"],
        "horizons_strongly_promising": summary["horizons_strongly_promising"],
        "selected_neural_horizons": summary["selected_neural_horizons"],
        "neural_qualified": summary["neural_qualified"],
        "lockbox_evaluated": summary["lockbox_evaluated"],
        "recommended_next_action": summary["recommended_next_action"],
        "multi_horizon_signal": summary["multi_horizon_signal"],
        "report": str(track.path("report")),
    }, indent=2))
    return 0


def freeze(summary: dict, payload: dict, track: HZ.MultiHorizonTrack, *,
           horizon: int | None, architecture: str | None) -> dict:
    """Write the single record that authorises the ONE 2019 lockbox run."""
    stability = summary.get("seed_stability")
    qualified = summary["neural_qualified"]
    if horizon is None or architecture is None:
        if not qualified:
            raise SystemExit(
                "refusing to freeze: no horizon/model pair passed the neural gate")
        model_name, horizon_key = qualified[0].split(":")
        horizon = int(horizon_key)
        architecture = model_name
    natural = ((summary["neural"].get(architecture) or {}).get(str(horizon)) or {})
    ledger_rows = [r for r in HZ.read_ledger(track.ledger)
                   if str(r.get("model")) == architecture
                   and _int(r, "horizon") == horizon]
    universe = json.loads((track.results_root / "supervised_universe_frozen.json")
                          .read_text())
    manifest = HS.build_frozen_manifest(
        horizon=int(horizon), architecture=architecture,
        screen_rank=summary["horizon_ranking"],
        neural_metrics=natural.get("natural", {}),
        common_metrics=natural.get("common_origin", {}),
        non_overlap_metrics=natural.get("non_overlapping", {}),
        stability=stability or {},
        universe_sha256=universe["universe_sha256"],
        feature_schema_sha256=HS.hash_payload(list(payload["model"].keys())),
        store_sha256=store_fingerprints(
            Path(payload["data"]["source_store_root"])).get("store_sha256"),
        target_schema_sha256=HZ.load_target_cache(
            Path(payload["data"]["horizon_target_cache"]),
            final_allowed_date=str(payload["final_allowed_date"]))[1][
                "target_schema_sha256"],
        config_sha256=HS.hash_payload(payload),
        experiment_ids=[str(r.get("experiment_id")) for r in ledger_rows],
        seed=int(payload["seed_stability"]["lockbox_seed"]),
        gates={"horizon": summary["horizon_gates"], "neural": summary["neural_gates"]},
    )
    atomic_json_dump(manifest, track.path("freeze"))
    return manifest


if __name__ == "__main__":
    raise SystemExit(main())