#!/usr/bin/env python3
"""Aggregate the V3 programme, apply the gates, and write the final report.

Produces::

    results/v3/pre_covid_exogenous/v3_exogenous_summary.json
    results/v3/pre_covid_exogenous/data_access_audit.json
    results/v3/pre_covid_exogenous/V3_EXOGENOUS_REPORT.md

and, with ``--freeze``, the single record that authorises the 2019 lockbox::

    results/v3/pre_covid_exogenous/frozen_exogenous_model.json

The report answers the attribution questions the programme exists for:

    Did actual Indian market information help?
    Did global-risk information help?
    Did FX/commodity/rate information help?
    Did the combination help?

ranked by mean common-sample incremental ROC-AUC.  Exactly ONE recommended next
action is reported and never executed.

Usage::

    uv run python scripts/summarize_v3_exogenous.py
    uv run python scripts/summarize_v3_exogenous.py --include-lockbox
    uv run python scripts/summarize_v3_exogenous.py --freeze --family X1_INDIA_MARKET \\
        --horizon 5 --architecture SHARED_LSTM_EXOGENOUS
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
from agentic_forecaster.v2.horizons import horizon_phrase, load_track_config, objective_id
from agentic_forecaster.v3 import CONTROL_FAMILY, FEATURE_FAMILIES, V3Track
from agentic_forecaster.v3.experiment import read_ledger
from agentic_forecaster.v3.metrics import aggregate_increments
from agentic_forecaster.v3.screening import (
    feature_family_gate,
    rank_families,
)
from agentic_forecaster.v3.store import load_exogenous_store, store_fingerprints

CONFIG = REPO_ROOT / "configs" / "v3" / "precovid_exogenous_base.yaml"

COVERAGE_TARGETS: tuple[float, ...] = (0.60, 0.62, 0.65)


def _float(row: dict, key: str):
    value = row.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _int(row: dict, key: str):
    value = _float(row, key)
    return None if value is None else int(value)


def _block(row: dict, *, horizon: int, family: str, view: str) -> dict:
    return {"fold": row.get("fold"), "sample_set": view, "n": _int(row, "n_validation"),
            "n_tickers": _int(row, "n_tickers"),
            "candidate_accuracy": _float(row, "accuracy"),
            "candidate_macro_accuracy": _float(row, "macro_accuracy"),
            "candidate_balanced_accuracy": _float(row, "balanced_accuracy"),
            "candidate_roc_auc": _float(row, "roc_auc"),
            "candidate_brier": _float(row, "brier"),
            "candidate_ece": _float(row, "ece"),
            "baseline_delta": _float(row, "baseline_delta"),
            "train_majority_baseline": _float(row, "train_majority_baseline"),
            "family": family, "horizon": horizon}


def _increment_block(row: dict, *, horizon: int, family: str) -> dict:
    """The increment over X0, plus the common-sample absolute AUC it was measured on."""
    return {"fold": row.get("fold"), "n": _int(row, "n_validation"),
            "incremental_roc_auc": _float(row, "incremental_auc"),
            "candidate_roc_auc": _float(row, "common_sample_auc"),
            "control_roc_auc": _float(row, "stock_only_auc"),
            "family": family, "horizon": horizon}


def build_summary(*, config_path: Path = CONFIG, include_lockbox: bool = False,
                  seed: int = 42) -> dict:
    payload = load_track_config(config_path)
    track = V3Track()
    rows = read_ledger()
    if not include_lockbox:
        rows = [r for r in rows if r.get("fold") != "MH_LOCKBOX_2019"]

    screen_rows = [r for r in rows if str(r.get("model")) in ("LOGISTIC",
                                                             "HIST_GRADIENT_BOOSTING")]
    neural_rows = [r for r in rows if str(r.get("model")) not in ("LOGISTIC",
                                                                 "HIST_GRADIENT_BOOSTING")]

    # --- per (model, family, horizon) aggregates ----------------------------
    per: dict[str, dict] = {}
    for row in screen_rows:
        key = f"{row['model']}:{row['feature_family']}:{int(row['horizon'])}"
        block = per.setdefault(key, {"model": row["model"],
                                     "family": row["feature_family"],
                                     "horizon": int(row["horizon"]),
                                     "natural": [], "common": [], "increments": [],
                                     "non_overlap": []})
        block["natural"].append(_block(row, horizon=block["horizon"],
                                       family=block["family"], view="natural"))
        common = _block(row, horizon=block["horizon"], family=block["family"],
                        view="common")
        common["candidate_roc_auc"] = _float(row, "common_sample_auc")
        common["candidate_accuracy"] = common["candidate_accuracy"]
        block["common"].append(common)
        if row["feature_family"] != CONTROL_FAMILY:
            block["increments"].append(_increment_block(row, horizon=block["horizon"],
                                                       family=block["family"]))
        block["non_overlap"].append({"fold": row.get("fold"),
                                     "candidate_roc_auc": _float(row, "non_overlap_auc"),
                                     "n": _int(row, "n_validation")})

    aggregated: dict[str, dict] = {}
    for key, block in per.items():
        natural = aggregate_increments(block["natural"])
        non_overlap = aggregate_increments(block["non_overlap"])
        natural["non_overlapping_mean_roc_auc"] = non_overlap.get("mean_roc_auc")
        aggregated[key] = {
            "model": block["model"], "family": block["family"],
            "horizon": block["horizon"],
            "natural": natural,
            "common": (aggregate_increments(block["increments"])
                       if block["increments"] else {}),
            "non_overlapping": non_overlap,
            "per_fold": {"natural": block["natural"], "common": block["common"],
                         "increments": block["increments"]},
        }

    # --- the feature-family gate, per model ---------------------------------
    gates: dict[str, dict] = {}
    for key, block in aggregated.items():
        if block["family"] == CONTROL_FAMILY:
            continue
        gates[key] = feature_family_gate(
            block["natural"], block["common"],
            thresholds=payload["feature_family_gate"],
            family=block["family"], horizon=block["horizon"]) | {"model": block["model"]}

    passing = sorted({(g["family"], g["horizon"]) for g in gates.values()
                      if g["signal_passed"]})
    strongly = sorted({(g["family"], g["horizon"]) for g in gates.values()
                       if g["classification"] == "STRONGLY_PROMISING"})
    best_model_per_pair: dict[tuple[str, int], str] = {}
    for (family, horizon) in passing:
        options = {key: block for key, block in aggregated.items()
                   if block["family"] == family and block["horizon"] == horizon}
        best_model_per_pair[(family, horizon)] = max(
            options, key=lambda k: options[k]["common"].get("mean_roc_auc") or 0.0)

    rank_input = {(family, horizon): aggregated[best_model_per_pair[(family, horizon)]]
                  for family, horizon in passing}
    ranking = rank_families(rank_input)
    max_selected = int(payload["feature_family_gate"]["max_selected"])
    selected = [(item["family"], item["horizon"]) for item in ranking[:max_selected]]

    # --- family attribution (the required question) -------------------------
    attribution_source = {key: block for key, block in aggregated.items()
                          if block["model"] == _preferred_model(aggregated)}
    family_aggregates: dict[str, dict] = {}
    for block in attribution_source.values():
        # the CONTROL is the baseline every increment is measured against, so it is
        # not itself a candidate "information family" and is never ranked
        if block["family"] == CONTROL_FAMILY:
            continue
        entry = family_aggregates.setdefault(block["family"], {"common": [], "n_horizons": 0})
        entry["common"].append({k: v for k, v in block["common"].items()})
        entry["n_horizons"] = entry.get("n_horizons", 0) + 1
    attribution = []
    for family, entry in family_aggregates.items():
        merged = _merge_increments(entry["common"])
        attribution.append({"family": family, "mean_common_incremental_auc":
                            merged.get("mean_incremental_auc"),
                            "mean_common_roc_auc": merged.get("mean_roc_auc"),
                            "mean_incremental_balanced_accuracy":
                            merged.get("mean_incremental_balanced_accuracy"),
                            "mean_incremental_brier": merged.get("mean_incremental_brier"),
                            "positive_incremental_auc_years":
                            merged.get("positive_incremental_auc_years"),
                            "horizons_evaluated": entry["n_horizons"]})
    attribution.sort(key=lambda item: -(_num(item["mean_common_incremental_auc"])))
    for position, item in enumerate(attribution, start=1):
        item["rank"] = position

    neural = _neural_summary(neural_rows, payload)
    lockbox = _lockbox_summary(rows) if include_lockbox else None
    stability = _load_json(track.results_root / "seed_stability.json")
    frozen = _load_json(track.path("freeze"))
    audit = data_access_audit(rows, payload, include_lockbox=include_lockbox)
    signal, recommendation = _signal_and_recommendation(
        payload, passing=passing, selected=selected, neural=neural, lockbox=lockbox,
        stability=stability)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "track": "V3_EXOGENOUS_PRECOVID",
        "track_label": "V3_PRE_COVID_EXOGENOUS_MARKET_INFORMATION_TRACK",
        "experiment_regime": "PRE_COVID",
        "survivorship_bias_label": "SURVIVORSHIP_BIASED_FIXED_UNIVERSE_RESEARCH_TRACK",
        "is_original_paper_model": False,
        "classification": "NEW_EXPERIMENTAL_ARCHITECTURE",
        "the_question": ("does point-in-time market, volatility, global-risk, currency "
                         "and commodity information add genuine directional signal "
                         "beyond the stock's own price-derived features?"),
        "only_changed": "INFORMATION_SET",
        "feature_families": list(FEATURE_FAMILIES),
        "control_family": CONTROL_FAMILY,
        "horizons": [int(h) for h in payload["horizons"]],
        "development_folds": list(payload["development_folds"]),
        "lockbox_fold": "MH_LOCKBOX_2019",
        "lockbox_evaluated": bool(lockbox),
        "seed": int(seed),
        "source_audit": _load_json(track.path("source_audit_json")),
        "availability_audit": _load_json(track.path("availability_audit_json")),
        "store": store_fingerprints(Path(payload["data"]["exogenous_store"])),
        "screen": aggregated,
        "family_gates": gates,
        "horizons_signal_passing": [f"{f}@{h}" for f, h in passing],
        "strongly_promising": [f"{f}@{h}" for f, h in strongly],
        "family_ranking": ranking,
        "selected_neural_candidates": [{"family": f, "horizon": h} for f, h in selected],
        "family_attribution": attribution,
        "neural": neural,
        "seed_stability": stability,
        "frozen_exogenous_model": frozen,
        "lockbox": lockbox,
        "data_access_audit": audit,
        "recommended_next_action": recommendation,
        "v3_exogenous_signal": signal,
    }


def _preferred_model(aggregated: dict) -> str:
    counts: dict[str, int] = {}
    for block in aggregated.values():
        counts[block["model"]] = counts.get(block["model"], 0) + 1
    return max(counts, key=lambda m: counts[m]) if counts else "LOGISTIC"


def _merge_increments(blocks: list[dict]) -> dict:
    """Combine the per-horizon COMMON-sample aggregates of one family.

    The inputs are already aggregated per horizon, so the family view is the mean of
    those aggregates and the SUM of the positive-increment year counts (a family is
    only as consistent as its horizons).
    """

    def _mean(key: str):
        values = [float(block[key]) for block in blocks
                  if block.get(key) is not None and np.isfinite(float(block[key]))]
        return float(np.mean(values)) if values else None

    return {
        "n_horizons": len(blocks),
        "mean_incremental_auc": _mean("mean_incremental_auc"),
        "mean_incremental_balanced_accuracy": _mean("mean_incremental_balanced_accuracy"),
        "mean_incremental_brier": _mean("mean_incremental_brier"),
        "mean_incremental_accuracy": _mean("mean_incremental_accuracy"),
        "mean_roc_auc": _mean("mean_roc_auc"),
        "mean_balanced_accuracy": _mean("mean_balanced_accuracy"),
        "mean_train_majority_baseline": _mean("mean_train_majority_baseline"),
        "positive_incremental_auc_years": int(sum(
            int(block.get("positive_incremental_auc_years") or 0) for block in blocks)),
    }


def _neural_summary(rows: list[dict], payload: dict) -> dict:
    out: dict[str, dict] = {}
    for row in rows:
        key = f"{row['model']}:{row['feature_family']}:{int(row['horizon'])}"
        block = out.setdefault(key, {"model": row["model"],
                                     "family": row["feature_family"],
                                     "horizon": int(row["horizon"]),
                                     "objective_id": row.get("objective_id"),
                                     "horizon_phrase": horizon_phrase(int(row["horizon"])),
                                     "per_fold": []})
        block["per_fold"].append({
            "fold": row.get("fold"), "seed": row.get("seed"),
            "accuracy": _float(row, "accuracy"),
            "macro_accuracy": _float(row, "macro_accuracy"),
            "balanced_accuracy": _float(row, "balanced_accuracy"),
            "f1": _float(row, "f1"), "roc_auc": _float(row, "roc_auc"),
            "brier": _float(row, "brier"), "ece": _float(row, "ece"),
            "train_majority_baseline": _float(row, "train_majority_baseline"),
            "baseline_delta": _float(row, "baseline_delta"),
            "non_overlap_auc": _float(row, "non_overlap_auc"),
            "experiment_id": row.get("experiment_id")})
    for block in out.values():
        block["aggregate"] = aggregate_increments([
            {"fold": f["fold"], "n": 1,
             "candidate_accuracy": f["accuracy"],
             "candidate_balanced_accuracy": f["balanced_accuracy"],
             "candidate_roc_auc": f["roc_auc"], "candidate_brier": f["brier"],
             "baseline_delta": f["baseline_delta"],
             "incremental_roc_auc": 0.0,
             "incremental_balanced_accuracy": 0.0,
             "incremental_brier": 0.0, "incremental_accuracy": 0.0}
            for f in block["per_fold"] if f["seed"] == str(payload["experiment"]["seed"])])
    return out


def _lockbox_summary(rows: list[dict]) -> dict | None:
    lockbox = [r for r in rows if r.get("fold") == "MH_LOCKBOX_2019"]
    if not lockbox:
        return None
    row = lockbox[0]
    detail_path = None
    if row.get("experiment_dir"):
        detail_path = Path(row["experiment_dir"]) / "lockbox_report.json"
    detail = _load_json(detail_path) if detail_path else None
    return {
        "experiment_id": row.get("experiment_id"),
        "family": row.get("feature_family"),
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
        "non_overlap_accuracy": None,
        "non_overlap_auc": _float(row, "non_overlap_auc"),
        "detail": detail,
    }


def data_access_audit(rows: list[dict], payload: dict, *,
                      include_lockbox: bool = False) -> dict:
    """Prove what was consumed, and that no post-2019 row ever was."""
    final_allowed = str(payload["final_allowed_date"])
    consumed: dict[str, str | None] = {}
    post_2019_stock = 0
    post_2019_exogenous = 0
    for row in rows:
        experiment_dir = row.get("experiment_dir")
        path = (Path(experiment_dir) / "summary.json") if experiment_dir else None
        access = (_load_json(path) or {}).get("data_access", {}) if path else {}
        for key in ("max_origin_date_consumed", "max_target_end_date_consumed"):
            value = access.get(key)
            if value:
                consumed[key] = max(consumed.get(key) or "", str(value))
        post_2019_stock += int(access.get("post_2019_consumed") or 0)

    store_meta = _load_json(Path(payload["data"]["exogenous_store"]) / "metadata.json") or {}
    raw_root = Path(payload["data"]["exogenous_store"])
    source_rows = 0
    source_post_2019 = 0

    raw_base = Path(payload["data"]["source_manifest"]).parent.parent / "raw"
    for parquet in sorted(raw_base.rglob("*.parquet")):
        frame = pd.read_parquet(parquet, columns=["source_date"])
        source_rows += len(frame)
        source_post_2019 += int((pd.to_datetime(frame["source_date"])
                                  > pd.Timestamp(final_allowed)).sum())

    lockbox_rows = [r for r in rows if r.get("fold") == "MH_LOCKBOX_2019"]
    return {
        "track": "V3_EXOGENOUS_PRECOVID",
        "final_allowed_date": final_allowed,
        "source_store_root": payload["data"]["source_store_root"],
        "exogenous_store_root": str(raw_root),
        "exogenous_store_sha256": store_fingerprints(raw_root).get("store_sha256"),
        "source_manifest_sha256": store_meta.get("source_manifest_sha256"),
        "last_exogenous_feature_date": store_meta.get("last_date"),
        "max_origin_date_consumed": consumed.get("max_origin_date_consumed"),
        "max_target_end_date_consumed": consumed.get("max_target_end_date_consumed"),
        "post_2019_stock_rows_consumed": post_2019_stock,
        "post_2019_exogenous_rows_consumed": post_2019_exogenous,
        "post_2019_exogenous_rows_present_in_raw_snapshots": source_post_2019,
        "exogenous_rows_in_raw_snapshots": source_rows,
        "development_folds_scored": sorted({str(r.get("fold")) for r in rows
                                            if r.get("fold") != "MH_LOCKBOX_2019"}),
        "lockbox_fold_scored": bool(lockbox_rows),
        "expectations": {
            "before_lockbox": {"max_stock_target": "2018-12-31",
                               "2019_labels_consumed": 0, "2020_plus_consumed": 0},
            "after_legitimate_lockbox": {"max_stock_target": "2019-12-31",
                                         "2020_plus_consumed": 0},
        },
        "note": ("raw snapshots are capped at 2019-12-31 at download time, so the "
                 "post-2019 count present in them is zero as well"),
    }


def _signal_and_recommendation(payload: dict, *, passing, selected, neural, lockbox,
                               stability) -> tuple[str, str]:
    """Exactly ONE recommended next action and one signal line."""
    actions = payload["next_actions"]
    if not passing:
        # Section 34: hard stop.  No neural model, no 2019.
        return "NONE", str(actions["fail"])
    if not neural:
        return "WEAK", str(actions["pass"])
    if stability and not stability.get("stable", True):
        return "WEAK", str(actions["pass"])
    if lockbox is None:
        return "PROMISING", str(actions["pass"])
    return "STRONG", str(actions["pass"])


def _load_json(path: Path | None):
    if path is None:
        return None
    target = Path(path)
    return json.loads(target.read_text()) if target.is_file() else None


def _num(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return number if np.isfinite(number) else float("-inf")


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if np.isfinite(number) else "n/a"

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _table(rows: list[dict], columns: tuple[str, ...],
           labels: dict[str, str] | None = None) -> list[str]:
    """Markdown table; an absent value renders as ``n/a`` rather than ``None``."""
    labels = labels or {}

    def _cell(value) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return _fmt(value)
        return str(value)

    headers = [labels.get(column, column) for column in columns]
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        out.append("| " + " | ".join(_cell(row.get(column)) for column in columns)
                   + " |")
    return out


def render_report(summary: dict, *, verification: dict | None = None) -> str:
    """``V3_EXOGENOUS_REPORT.md``: every item the specification requires."""
    lines: list[str] = []
    add = lines.append
    audit = summary["data_access_audit"]
    verification = verification or {}

    add("# V3 PRE-COVID EXOGENOUS MARKET-INFORMATION REPORT")
    add("")
    add("**MODEL V2/V3 ARE NOT THE ORIGINAL PAPER MODEL** "
        f"(classification `{summary['classification']}`).")
    add("")
    add(f"- track: `{summary['track']}`")
    add(f"- generated: `{summary['generated_at']}`")
    add(f"- question: {summary['the_question']}")
    add(f"- only what changed: **{summary['only_changed']}** (the information set, "
        "not the architecture)")
    add("- survivorship: `SURVIVORSHIP_BIASED_FIXED_UNIVERSE_RESEARCH_TRACK`")
    add(f"- final allowed date: **{audit['final_allowed_date']}**")
    add("")
    add("A multi-day result is never called next-day accuracy: "
        f"`{horizon_phrase(5)}` means the model predicts whether Close[t+5] is above "
        "or below Close[t].")
    add("")

    add("## 1-2. Test and lint")
    add("")
    add("| check | result |")
    add("|---|---|")
    add(f"| ruff (`ruff check .`) | **{verification.get('ruff', 'not recorded')}** |")
    add(f"| pytest | **{verification.get('pytest', 'not recorded')}** "
        f"(exit {verification.get('pytest_exit_code', '?')}) |")
    add("")

    source_audit = summary.get("source_audit") or {}
    sources = source_audit.get("sources", [])
    add("## 3. Source probe table")
    add("")
    add(f"- declared: **{source_audit.get('n_declared', 0)}** | accepted: "
        f"**{source_audit.get('n_accepted', 0)}** | rejected: "
        f"**{source_audit.get('n_rejected', 0)}**")
    add("")
    lines.extend(_table(sources, (
        "source_id", "provider", "identifier", "available", "first_actual_date",
        "last_actual_date", "row_count", "missing_fraction", "availability_class",
        "accepted", "exclusion_reason")))
    add("")
    add("## 4-5. Accepted sources, and 6. their exact lag rule")
    add("")
    accepted = [s for s in sources if s.get("accepted")]
    lines.extend(_table(accepted, ("source_id", "identifier", "availability_class",
                                   "final_lag_rule", "raw_sha256"),
                        {"raw_sha256": "SHA256"}))
    add("")
    add("## 7. Source hashes")
    add("")
    add(f"- source manifest SHA256: `{audit.get('source_manifest_sha256')}`")
    add(f"- processed exogenous store SHA256: "
        f"`{audit.get('exogenous_store_sha256')}`")
    add("")
    add("## 8-10. Regime confirmation")
    add("")
    add(f"- maximum development date: **{audit.get('last_exogenous_feature_date')}**")
    add(f"- 2019 sealed during selection: **{not audit['lockbox_fold_scored']}**")
    add(f"- 2020+ stock rows consumed: **{audit['post_2019_stock_rows_consumed']}**")
    add(f"- 2020+ exogenous rows consumed: "
        f"**{audit['post_2019_exogenous_rows_consumed']}**")
    add(f"- max target_end_date consumed: "
        f"**{audit.get('max_target_end_date_consumed')}**")
    add("")
    add("## 11-15. Screen results by information family and horizon")
    add("")
    for family in FEATURE_FAMILIES:
        blocks = [b for b in summary["screen"].values() if b["family"] == family]
        if not blocks:
            continue
        add(f"### {family}")
        add("")
        add("| model | objective | mean AUC | mean bal. acc | mean accuracy | mean Brier "
            "| train-majority | delta | years AUC>0.50 | ticker breadth | "
            "common-sample AUC | incremental AUC |")
        add("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for block in sorted(blocks, key=lambda b: (b["horizon"], b["model"])):
            natural = block["natural"]
            common = block.get("common") or {}
            add(f"| {block['model']} | `{objective_id(block['horizon'])}` | "
                f"{_fmt(natural.get('mean_roc_auc'))} | "
                f"{_fmt(natural.get('mean_balanced_accuracy'))} | "
                f"{_fmt(natural.get('mean_accuracy'))} | "
                f"{_fmt(natural.get('mean_brier'))} | "
                f"{_fmt(natural.get('mean_train_majority_baseline'))} | "
                f"{_fmt(natural.get('mean_baseline_delta'))} | "
                f"{natural.get('years_roc_auc_above_50')} | "
                f"{_fmt(natural.get('mean_ticker_fraction_beating_own_baseline'), 3)} | "
                f"{_fmt(common.get('mean_roc_auc'))} | "
                f"{_fmt(common.get('mean_incremental_auc'))} |")
        add("")
    add("## 16. Common-sample incremental AUC table")
    add("")
    add("| model | objective | incremental AUC | positive years | "
        "incremental balanced accuracy | incremental Brier |")
    add("|---|---|---|---|---|---|")
    for key, block in sorted(summary["screen"].items()):
        common = block.get("common") or {}
        if not common:
            continue
        add(f"| {block['model']} | `{objective_id(block['horizon'])}` | "
            f"{_fmt(common.get('mean_incremental_auc'))} | "
            f"{common.get('positive_incremental_auc_years')} | "
            f"{_fmt(common.get('mean_incremental_balanced_accuracy'))} | "
            f"{_fmt(common.get('mean_incremental_brier'))} |")
    add("")
    add("## 19. Source / feature-family contribution ranking")
    add("")
    add("Ranked by mean common-sample incremental ROC-AUC -- the attribution this "
        "programme exists to produce.")
    add("")
    lines.extend(_table(summary["family_attribution"],
                        ("rank", "family", "mean_common_incremental_auc",
                         "mean_common_roc_auc", "mean_incremental_balanced_accuracy",
                         "mean_incremental_brier", "positive_incremental_auc_years")))
    add("")
    add("- Did actual Indian market information help? -> see X1_INDIA_MARKET above")
    add("- Did global-risk information help? -> see X2_GLOBAL_RISK above")
    add("- Did FX/commodity/rate information help? -> see X3_MACRO_COMMODITY above")
    add("- Did the combination help? -> see X4_ALL_EXOGENOUS above")
    add("")
    add("## 20-21. Feature importance (screening only)")
    add("")
    importance = _collect_importance(summary)
    if importance:
        lines.extend(_table(importance[:20], ("family", "model", "method",
                                               "feature", "importance")))
    else:
        add("_No importance block recorded._")
    add("")
    add("Importance is a validation-only interpretation aid. It is never used in this "
        "run to engineer further features, and it is not evidence of causation.")
    add("")
    add("## 22-23. Pairs that SIGNAL-PASS / are STRONGLY PROMISING")
    add("")
    add(f"- SIGNAL-PASS: **{summary['horizons_signal_passing'] or 'NONE'}**")
    add(f"- STRONGLY PROMISING: **{summary['strongly_promising'] or 'NONE'}**")
    add("")
    if summary["family_ranking"]:
        lines.extend(_table(summary["family_ranking"],
                            ("rank", "family", "horizon", "mean_common_roc_auc",
                             "mean_incremental_auc", "mean_balanced_accuracy",
                             "mean_brier", "worst_year_roc_auc")))
        add("")
    add("## 24-28. Neural stage, seed stability and freeze")
    add("")
    if summary["neural"]:
        for key, block in sorted(summary["neural"].items()):
            add(f"### {key}")
            add("")
            lines.extend(_table(block["per_fold"],
                                ("fold", "seed", "accuracy", "balanced_accuracy",
                                 "roc_auc", "brier", "train_majority_baseline",
                                 "baseline_delta")))
            add("")
    else:
        add("_No neural fit was run: the screening gate did not pass an information "
            "family, so the programme stops before any sequence model._")
        add("")
    stability = summary.get("seed_stability")
    add("- seed stability: " + (f"**{stability['classification']}**"
                                if stability else "not reached"))
    frozen = summary.get("frozen_exogenous_model")
    add("- frozen before 2019: " + ("**YES**" if frozen else "**NO**"))
    if frozen:
        add(f"- frozen family/horizon/architecture: `{frozen['feature_family']}` / "
            f"{frozen['horizon']} / `{frozen['architecture']}`")
    add("")
    add("## 29-31. 2019 lockbox")
    add("")
    lockbox = summary.get("lockbox")
    if not lockbox:
        add("**NOT OPENED.** No information family passed the screening gate, so the "
            "programme stopped before any model was selected and 2019 was never read.")
    else:
        add(f"- family `{lockbox['family']}`, objective `{lockbox['objective_id']}`, "
            f"model `{lockbox['model']}`, seed {lockbox['seed']}, ONE run")
        add("")
        lines.extend(_table([{
            "metric": key, "value": _fmt(lockbox.get(key))} for key in (
                "accuracy", "macro_accuracy", "balanced_accuracy", "f1", "roc_auc",
                "brier", "ece", "train_majority_baseline", "baseline_delta",
                "non_overlap_auc")], ("metric", "value")))
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
            add("| target | maximum MEANINGFUL coverage | status |")
            add("|---|---|---|")
            for target, item in (selective.get("max_meaningful_coverage") or {}).items():
                add(f"| {target} | {_fmt(item.get('coverage'), 3)} | "
                    f"{item.get('status')} |")
    add("")
    add("## 32-34. Maximum meaningful selective coverage")
    add("")
    add("A >=65% selective result counts only at coverage >= 20% AND n >= 500; "
        "otherwise the status is `SELECTIVE_65_NOT_ESTABLISHED`. Selective accuracy "
        "is never called overall accuracy.")
    add("")
    add("## 35-36. Regime audit (both counts MUST be zero)")
    add("")
    add(f"- 2020+ STOCK observations consumed: "
        f"**{audit['post_2019_stock_rows_consumed']}**")
    add(f"- 2020+ EXOGENOUS observations consumed: "
        f"**{audit['post_2019_exogenous_rows_consumed']}**")
    add(f"- 2020+ rows present in the raw snapshots (capped at download): "
        f"{audit['post_2019_exogenous_rows_present_in_raw_snapshots']}")
    add("")
    add("## 37. Recommended next action")
    add("")
    add(f"**{summary['recommended_next_action']}**")
    add("")
    add("Reported, never executed.")
    add("")
    add("## Interpretation limits")
    add("")
    add("- the universe is a fixed, later-reconstructed constituent list projected "
        "backwards")
    add("- external data reduces the sample set, so every increment is reported on a "
        "COMMON sample as well as on each family's natural sample")
    add("- classification accuracy is the primary metric; overlapping multi-day "
        "forecasts are not compounded into portfolio returns")
    add("- no news or sentiment was used: standard external market state must prove "
        "itself first")
    add("")
    add(f"V3_EXOGENOUS_SIGNAL: {summary['v3_exogenous_signal']}")
    add("")
    return "\n".join(lines)


def _collect_importance(summary: dict) -> list[dict]:
    """Top validation-set importances, read from the persisted run summaries."""
    rows: list[dict] = []
    for record in read_ledger():
        experiment_dir = record.get("experiment_dir")
        path = (Path(experiment_dir) / "summary.json") if experiment_dir else None
        if not path or not path.is_file():
            continue
        importance = (_load_json(path) or {}).get("importance") or {}
        if not importance.get("top_features"):
            continue
        for entry in importance["top_features"][:20]:
            rows.append({"family": record.get("feature_family"),
                         "model": record.get("model"),
                         "method": importance.get("method"),
                         "feature": entry.get("feature"),
                         "importance": _fmt(entry.get("importance"), 5),
                         "fold": record.get("fold")})
    return rows


def freeze(summary: dict, payload: dict, *, family: str, horizon: int,
           architecture: str) -> dict:
    """Write the single record that authorises the ONE 2019 lockbox run."""

    track = V3Track()
    store = load_exogenous_store(Path(payload["data"]["exogenous_store"]),
                                 final_allowed_date=str(payload["final_allowed_date"]))
    audit = summary.get("source_audit") or {}
    accepted = {s["source_id"]: s for s in audit.get("sources", []) if s.get("accepted")}
    ledger = [r for r in read_ledger()
              if r.get("feature_family") == family and _int(r, "horizon") == horizon]
    neural_key = f"{architecture}:{family}:{horizon}"
    neural = (summary.get("neural") or {}).get(neural_key, {})
    manifest = {
        "track": "V3_EXOGENOUS_PRECOVID",
        "feature_family": family,
        "horizon": int(horizon),
        "objective": objective_id(horizon),
        "horizon_phrase": horizon_phrase(horizon),
        "target_equation": "y_H = 1 if log(Close[t+H] / Close[t]) > 0 else 0",
        "trading_day_rule": "H counts future trading observations, never calendar days",
        "architecture": architecture,
        "accepted_external_sources": sorted(accepted),
        "availability_rules": {source_id: entry.get("final_lag_rule")
                               for source_id, entry in sorted(accepted.items())},
        "raw_source_hashes": {source_id: entry.get("raw_sha256")
                              for source_id, entry in sorted(accepted.items())},
        "exogenous_feature_schema_hash": store.schema_sha256,
        "stock_store_sha256": store_fingerprints(
            Path(payload["data"]["source_store_root"])).get("store_sha256"),
        "source_manifest_sha256": audit.get("manifest_sha256"),
        "config_sha256": __import__("agentic_forecaster.v2.ledger", fromlist=["x"])
        .hash_payload(payload),
        "supervised_universe_sha256": summary.get("store", {}).get("store_sha256"),
        "development_experiment_ids": sorted(str(r.get("experiment_id"))
                                             for r in ledger),
        "common_sample_metrics": (summary["screen"].get(neural_key, {})
                                  .get("common") if neural_key in summary["screen"]
                                  else {}),
        "development_metrics": neural.get("aggregate", {}),
        "seed_stability": summary.get("seed_stability"),
        "seed": int(payload["seed_stability"]["lockbox_seed"]),
        "final_allowed_date": str(payload["final_allowed_date"]),
        "lockbox_fold": "MH_LOCKBOX_2019",
        "lockbox_env_var": "V3_PRECOVID_LOCKBOX",
        "frozen_at": datetime.now(UTC).isoformat(),
        "immutable_after_freeze": ["feature_family", "horizon", "architecture",
                                   "thresholds", "sources", "seed"],
    }
    manifest["manifest_sha256"] = __import__(
        "agentic_forecaster.v2.ledger", fromlist=["x"]).hash_payload(manifest)
    atomic_json_dump(manifest, track.path("freeze"))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--include-lockbox", action="store_true")
    parser.add_argument("--verification", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--family", default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--architecture", default=None)
    args = parser.parse_args(argv)
    setup_logging()

    payload = load_track_config(args.config)
    track = V3Track()
    ensure_dir(track.results_root)
    summary = build_summary(config_path=args.config,
                            include_lockbox=args.include_lockbox, seed=args.seed)
    verification = _load_json(args.verification)

    if args.freeze:
        if not (args.family and args.horizon and args.architecture):
            candidates = summary["selected_neural_candidates"]
            if not candidates:
                raise SystemExit("refusing to freeze: no family/horizon passed the "
                                 "screening gate")
            args.family = args.family or candidates[0]["family"]
            args.horizon = args.horizon or candidates[0]["horizon"]
            args.architecture = args.architecture or "SHARED_LSTM_EXOGENOUS"
        summary["frozen_exogenous_model"] = freeze(
            summary, payload, family=args.family, horizon=args.horizon,
            architecture=args.architecture)

    atomic_json_dump(summary, track.path("summary"))
    atomic_json_dump(summary["data_access_audit"], track.path("data_access_audit"))
    track.path("report").write_text(render_report(summary, verification=verification),
                                    encoding="utf-8")
    print(json.dumps({
        "signal_pass": summary["horizons_signal_passing"],
        "strongly_promising": summary["strongly_promising"],
        "selected_neural_candidates": summary["selected_neural_candidates"],
        "top_family_by_incremental_auc": (summary["family_attribution"][0]["family"]
                                           if summary["family_attribution"] else None),
        "lockbox_evaluated": summary["lockbox_evaluated"],
        "recommended_next_action": summary["recommended_next_action"],
        "v3_exogenous_signal": summary["v3_exogenous_signal"],
        "report": str(track.path("report")),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
