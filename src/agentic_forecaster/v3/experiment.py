"""V3 experiment orchestration: the cheap screen, the ledger, and the manifests.

WHAT ONE SCREEN RUN IS
----------------------
``(feature family, horizon, model, development year)`` fitted on TRAIN and scored
on that validation year, in two views:

NATURAL   every sample for which that family's exogenous block is finite;
COMMON    only samples finite for EVERY family, so X0..X4 see identical
          observations and an apparent gain cannot be a subset artefact.

``incremental_auc_Xk = AUC(Xk) - AUC(X0)`` is computed on the COMMON sample, which
is the number that decides whether exogenous INFORMATION helped.

THE TARGETS AND FOLDS ARE NOT REBUILT HERE
-------------------------------------------
The verified multi-horizon implementation supplies the trading-observation targets,
the origin/target split boundary rule and the 2019 seal.  V3 delegates to it.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from agentic_forecaster.utils import atomic_json_dump, ensure_dir
from agentic_forecaster.v2 import horizons as HZ
from agentic_forecaster.v2.dataset import resolve_fold_window
from agentic_forecaster.v2.horizon_dataset import build_horizon_sample_table
from agentic_forecaster.v2.horizons import load_track_config
from agentic_forecaster.v2.ledger import experiment_dir, hash_payload
from agentic_forecaster.v2.sectors import load_sector_map
from agentic_forecaster.v2.store import load_store

from . import CONTROL_FAMILY, FEATURE_FAMILIES, LEDGER_COLUMNS, V3Track
from .dataset import build_family_tables, common_family_mask, family_split_mask
from .metrics import aggregate_increments
from .screening import (
    SCREEN_SETTINGS,
    ScreenOutcome,
    attach_increment,
    non_overlap_metrics,
    run_screen,
)
from .sources import SourceRegistry
from .store import ExogenousStore

logger = logging.getLogger("agentic_forecaster.v3.experiment")

DEFAULT_CONFIG = Path("configs/v3/precovid_exogenous_base.yaml")


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------

def new_experiment_id(prefix: str = "V3") -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    import uuid

    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


def read_ledger(path: Path | None = None) -> list[dict]:
    target = Path(path) if path is not None else V3Track().ledger
    if not target.is_file():
        return []
    with target.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_row(record: dict, path: Path | None = None, *,
               lockbox_evaluated: bool = False) -> dict:
    """Append one immutable V3 row.

    The helper REFUSES a development row that claims the 2019 lockbox was scored,
    and refuses any row claiming a post-2019 evaluation, so a leak cannot be
    recorded as if it were legitimate.
    """
    target = Path(path) if path is not None else V3Track().ledger
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {c: record.get(c, "") for c in LEDGER_COLUMNS}
    row["experiment_id"] = row["experiment_id"] or new_experiment_id()
    row["timestamp"] = row["timestamp"] or datetime.now(UTC).isoformat()
    row["family"] = row.get("feature_family", "")
    row["model"] = str(row["model"]).upper()
    if row.get("horizon") not in (None, ""):
        row["horizon"] = int(row["horizon"])
        row["objective_id"] = row.get("objective_id") or HZ.objective_id(row["horizon"])
    if str(row["post_2019_evaluated"]).strip().lower() in ("true", "1", "yes"):
        raise ValueError(
            "Refusing to append a V3 row claiming post_2019_evaluated=true. Nothing "
            "from 2020 onward may be consumed by any V3 script.")
    row["post_2019_evaluated"] = "false"
    claimed = str(row["2019_lockbox_evaluated"]).strip().lower() in ("true", "1", "yes")
    if claimed and not lockbox_evaluated:
        raise ValueError(
            "Refusing to append a development row claiming 2019_lockbox_evaluated=true. "
            "Only scripts/run_v3_precovid_lockbox.py with V3_PRECOVID_LOCKBOX=1 may "
            "score 2019.")
    row["2019_lockbox_evaluated"] = "true" if lockbox_evaluated else "false"

    new_file = not target.exists()
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LEDGER_COLUMNS))
        if new_file:
            writer.writeheader()
        writer.writerow({c: row[c] for c in LEDGER_COLUMNS})
    return row


# ---------------------------------------------------------------------------
# shared inputs
# ---------------------------------------------------------------------------

@dataclass
class V3Inputs:
    """Everything the screen needs, assembled once and reused."""

    payload: dict
    arrays: object
    stock_features: pd.DataFrame
    targets: pd.DataFrame
    samples_by_horizon: dict
    family_tables: dict
    registry: SourceRegistry
    store: ExogenousStore
    config_sha256: str
    target_schema_sha256: str
    source_manifest_sha256: str
    feature_schema_sha256: str
    supervised_universe_sha256: str


def assemble_inputs(config_path: Path, *, locked: bool = False) -> V3Inputs:
    """Load the stock store, the shared targets, the exogenous store and the samples."""
    payload = load_track_config(config_path)
    data = payload["data"]
    final_allowed = str(payload["final_allowed_date"])
    sequence_length = int(data["sequence_length"])

    stock_store = load_store(Path(data["source_store_root"]),
                             final_allowed_date=final_allowed)
    sector_map = load_sector_map(Path(data["sector_map_csv"]))

    # The supervised universe is the FROZEN multi-horizon one: V3 deliberately keeps
    # the identical ticker set, so any difference in the result comes from the
    # information and not from a different population.
    universe_path = Path(data["supervised_universe"])
    if not universe_path.is_absolute():
        universe_path = Path("results/v2/multi_horizon") / universe_path
    universe = json.loads(universe_path.read_text())
    tickers = list(universe["eligible_tickers"])

    from agentic_forecaster.v2.dataset import build_feature_arrays
    from agentic_forecaster.v2.experiment import assemble_context_frame

    arrays = build_feature_arrays(stock_store.stock, assemble_context_frame(stock_store),
                                  sector_map, tickers=tickers, use_context=False)
    targets, target_metadata = HZ.load_target_cache(
        Path(payload["data"]["horizon_target_cache"]), final_allowed_date=final_allowed)

    horizons = tuple(int(h) for h in payload["horizons"])
    samples_by_horizon = {
        horizon: build_horizon_sample_table(
            arrays, targets, horizon=horizon, sequence_length=sequence_length,
            final_allowed_date=final_allowed, locked=locked)
        for horizon in horizons}

    from .store import load_exogenous_store

    exogenous_store = load_exogenous_store(final_allowed_date=final_allowed)
    registry = _load_registry(payload)
    stock_features = stock_store.stock.loc[stock_store.stock["ticker"].isin(tickers)]
    family_tables = build_family_tables(samples_by_horizon, exogenous_store, registry,
                                        stock_features=stock_features)

    manifest_path = Path(payload["data"]["source_manifest"])
    source_manifest_sha256 = (json.loads(manifest_path.read_text())["manifest_sha256"]
                              if manifest_path.is_file() else None)

    return V3Inputs(
        payload=payload, arrays=arrays, stock_features=stock_features, targets=targets,
        samples_by_horizon=samples_by_horizon, family_tables=family_tables,
        registry=registry, store=exogenous_store,
        config_sha256=hash_payload(payload),
        target_schema_sha256=str(target_metadata["target_schema_sha256"]),
        source_manifest_sha256=str(source_manifest_sha256),
        feature_schema_sha256=exogenous_store.schema_sha256,
        supervised_universe_sha256=str(universe["universe_sha256"]))


def _load_registry(payload: dict) -> SourceRegistry:
    from .sources import load_effective_registry

    data = payload["data"]
    return load_effective_registry(
        Path(data["resolved_registry"]), Path(data["declared_registry"]),
        Path("results/v3/pre_covid_exogenous/source_audit.json"))


# ---------------------------------------------------------------------------
# the screen
# ---------------------------------------------------------------------------

def run_screen_family(inputs: V3Inputs, *, model_name: str, horizons, folds,
                      families=FEATURE_FAMILIES, with_importance: bool = False,
                      reset_ledger: bool = False, pca_components: int | None = None
                      ) -> dict:
    """Run the whole cheap screen for ONE model and persist every row."""
    track = V3Track()
    ensure_dir(track.results_root)
    ensure_dir(track.runtime_root)
    if reset_ledger and track.ledger.is_file():
        track.ledger.unlink()

    payload = inputs.payload
    bootstrap = dict(payload["evaluation"]["bootstrap"])
    results: list[dict] = []
    for horizon in horizons:
        for fold in folds:
            window = resolve_fold_window(fold, payload["folds"])
            natural_masks = {
                family: family_split_mask(inputs.family_tables[(horizon, family)],
                                          window, fold=fold)
                for family in families}
            common_masks = common_family_mask(inputs.family_tables, window,
                                              families=families, horizon=horizon,
                                              fold=fold)

            outcomes: dict[tuple[str, str], ScreenOutcome] = {}
            for family in families:
                for view, masks in (("natural", natural_masks[family]),
                                    ("common", common_masks[family])):
                    outcome = run_screen(
                        inputs.family_tables[(horizon, family)], model_name=model_name,
                        window=window, fold=fold, train_mask=masks["train"],
                        val_mask=masks["val"],
                        pca_components=(pca_components if view == "natural" else None),
                        with_importance=(with_importance and view == "natural"
                                         and family != CONTROL_FAMILY),
                        bootstrap_kwargs=bootstrap)
                    outcome.view = view
                    outcome.metrics["sample_set"] = view
                    outcomes[(family, view)] = outcome

            for family in families:
                for view in ("natural", "common"):
                    outcome = outcomes[(family, view)]
                    if family != CONTROL_FAMILY:
                        outcome = attach_increment(outcome, outcomes[(CONTROL_FAMILY,
                                                                     view)])
                    else:
                        outcome.increment = {
                            "family": family, "horizon": horizon, "sample_set": view,
                            "n": outcome.metrics.get("n"), "comparable": True,
                            "incremental_roc_auc": 0.0 if view == "natural" else None,
                            "note": "X0 is the control: its increment over itself is zero",
                        }

            for family in families:
                natural = outcomes[(family, "natural")]
                baselines = {"global": natural.metrics["train_majority_baseline"],
                             "per_ticker": {}}
                non_overlap = non_overlap_metrics(natural.predictions,
                                                  horizon=horizon, baselines=baselines,
                                                  family=family)
                common = outcomes[(family, "common")]
                experiment_id = new_experiment_id("V3")
                out_dir = experiment_dir(experiment_id, runtime_root=track.runtime_root)
                natural.predictions.to_csv(out_dir / "predictions_natural.csv", index=False)
                common.predictions.to_csv(out_dir / "predictions_common.csv", index=False)
                atomic_json_dump({
                    "experiment_id": experiment_id,
                    "track": "V3_EXOGENOUS_PRECOVID",
                    "family": family,
                    "model": model_name,
                    "horizon": horizon,
                    "objective_id": HZ.objective_id(horizon),
                    "horizon_phrase": HZ.horizon_phrase(horizon),
                    "fold": fold,
                    "view": "natural+common",
                    "metrics_natural": natural.metrics,
                    "metrics_common": common.metrics,
                    "metrics_non_overlapping": non_overlap,
                    "increment_over_x0": natural.increment,
                    "importance": natural.importance,
                    "transform_state": natural.transform_state,
                    "data_access": {
                        "final_allowed_date": str(payload["final_allowed_date"]),
                        "max_origin_date_consumed": str(
                            pd.Timestamp(natural.predictions["origin_date"].max()).date()),
                        "max_target_end_date_consumed": str(
                            pd.Timestamp(natural.predictions["target_date"].max()).date()),
                        "lockbox_year_consumed": False,
                        "post_2019_consumed": 0,
                    },
                }, out_dir / "summary.json")

                append_row({
                    "experiment_id": experiment_id,
                    "experiment_dir": str(out_dir),
                    "model": model_name,
                    "feature_family": family,
                    "horizon": horizon,
                    "objective_id": HZ.objective_id(horizon),
                    "fold": fold,
                    "seed": int(payload["experiment"]["seed"]),
                    "n_train": natural.n_train,
                    "n_validation": natural.metrics.get("n"),
                    "n_tickers": natural.metrics.get("n_tickers"),
                    "accuracy": natural.metrics.get("accuracy"),
                    "macro_accuracy": natural.metrics.get("macro_ticker_accuracy"),
                    "balanced_accuracy": natural.metrics.get("balanced_accuracy"),
                    "f1": natural.metrics.get("f1"),
                    "roc_auc": natural.metrics.get("roc_auc"),
                    "brier": natural.metrics.get("brier"),
                    "ece": natural.metrics.get("ece"),
                    "train_majority_baseline": natural.metrics.get(
                        "train_majority_baseline"),
                    "baseline_delta": natural.metrics.get("baseline_delta"),
                    "stock_only_auc": outcomes[(CONTROL_FAMILY, "common")].metrics.get(
                        "roc_auc"),
                    "incremental_auc": common.increment.get("incremental_roc_auc"),
                    "common_sample_auc": common.metrics.get("roc_auc"),
                    "non_overlap_auc": non_overlap.get("roc_auc"),
                    "config_sha256": inputs.config_sha256,
                    "source_manifest_sha256": inputs.source_manifest_sha256,
                    "feature_schema_sha256": inputs.feature_schema_sha256,
                    "2019_lockbox_evaluated": False,
                    "post_2019_evaluated": False,
                }, path=track.ledger)

                results.append({
                    "family": family, "horizon": horizon, "model": model_name,
                    "fold": fold, "natural": natural.metrics, "common": common.metrics,
                    "non_overlapping": non_overlap, "increment": natural.increment,
                    "importance": natural.importance,
                    "transform_state": natural.transform_state,
                })
                print(f"[screen] {model_name:<22} {family:<18} "
                      f"{HZ.objective_id(horizon):<18} {fold}: "
                      f"auc={natural.metrics.get('roc_auc'):.4f} "
                      f"bal={natural.metrics.get('balanced_accuracy'):.4f} "
                      f"base={natural.metrics.get('train_majority_baseline'):.4f} "
                      f"inc_auc={common.increment.get('incremental_roc_auc')} "
                      f"(common n={common.metrics.get('n')}, "
                      f"nonoverlap n={non_overlap.get('n')})", flush=True)

    manifest = {
        "track": "V3_EXOGENOUS_PRECOVID",
        "stage": "CHEAP_SCREEN",
        "model": model_name,
        "horizons": [int(h) for h in horizons],
        "folds": list(folds),
        "families": list(families),
        "settings": dict(SCREEN_SETTINGS[model_name]),
        "hyperparameter_search": False,
        "transforms_fit_on": "TRAIN_ONLY",
        "config_sha256": inputs.config_sha256,
        "target_schema_sha256": inputs.target_schema_sha256,
        "source_manifest_sha256": inputs.source_manifest_sha256,
        "feature_schema_sha256": inputs.feature_schema_sha256,
        "supervised_universe_sha256": inputs.supervised_universe_sha256,
        "n_runs": len(results),
        "2019_lockbox_evaluated": False,
        "post_2019_evaluated": False,
        "results": results,
    }
    name = "logistic" if model_name == "LOGISTIC" else "hist_gradient_boosting"
    atomic_json_dump(manifest, track.results_root / f"screen_{name}_manifest.json")
    return manifest


def aggregate_screen(manifests: list[dict]) -> dict:
    """Aggregate the screen into per-(model, family, horizon) blocks."""
    per: dict[str, dict] = {}
    for manifest in manifests:
        model = manifest["model"]
        for row in manifest["results"]:
            key = f"{model}:{row['family']}:{row['horizon']}"
            block = per.setdefault(key, {"model": model, "family": row["family"],
                                         "horizon": row["horizon"], "natural": [],
                                         "common": [], "non_overlapping": [],
                                         "importance": []})
            block["natural"].append({"fold": row["fold"],
                                     "candidate_roc_auc": row["natural"].get("roc_auc"),
                                     "candidate_accuracy": row["natural"].get("accuracy"),
                                     "candidate_balanced_accuracy":
                                     row["natural"].get("balanced_accuracy"),
                                     "candidate_brier": row["natural"].get("brier"),
                                     "baseline_delta":
                                     row["natural"].get("baseline_delta"),
                                     "ticker_fraction_beating_own_baseline":
                                     row["natural"].get(
                                         "ticker_fraction_beating_own_baseline"),
                                     "n": row["natural"].get("n")})
            block["common"].append({"fold": row["fold"],
                                    "incremental_roc_auc":
                                    row["increment"].get("incremental_roc_auc"),
                                    "incremental_balanced_accuracy":
                                    row["increment"].get(
                                        "incremental_balanced_accuracy"),
                                    "incremental_brier":
                                    row["increment"].get("incremental_brier"),
                                    "incremental_accuracy":
                                    row["increment"].get("incremental_accuracy"),
                                    "candidate_roc_auc":
                                    row["common"].get("roc_auc"),
                                    "candidate_accuracy": row["common"].get("accuracy"),
                                    "candidate_balanced_accuracy":
                                    row["common"].get("balanced_accuracy"),
                                    "candidate_brier": row["common"].get("brier"),
                                    "baseline_delta": row["common"].get("baseline_delta"),
                                    "n": row["common"].get("n")})
            if row.get("non_overlapping"):
                block["non_overlapping"].append({
                    "fold": row["fold"],
                    "candidate_roc_auc": row["non_overlapping"].get("roc_auc"),
                    "n": row["non_overlapping"].get("n")})
            if row.get("importance"):
                block["importance"].append(row["importance"])

    out: dict[str, dict] = {}
    for key, block in per.items():
        non_overlap_agg = aggregate_increments(block["non_overlapping"])
        natural = aggregate_increments(block["natural"])
        out[key] = {
            "model": block["model"], "family": block["family"],
            "horizon": block["horizon"],
            "natural": natural,
            "common": aggregate_increments(block["common"]),
            "non_overlapping": non_overlap_agg,
            "importance": block["importance"],
        }
        out[key]["natural"]["non_overlapping_mean_roc_auc"] = non_overlap_agg.get(
            "mean_roc_auc")
    return out