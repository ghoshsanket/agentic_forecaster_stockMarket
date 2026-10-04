"""The V3 cheap screen: Logistic and HistGradientBoosting over X0..X4.

FIVE FAMILIES x FOUR HORIZONS x FIVE YEARS x TWO MODELS = 200 cheap fits.

TRAIN-ONLY TRANSFORMS
---------------------
Every ``StandardScaler`` is fitted on TRAIN ROWS ONLY and then applied unchanged to
the validation year.  No PCA is fitted here.  The fitted transformation is persisted
with the experiment so a result can be reproduced without refitting.

THE HEADLINE OUTPUT
-------------------
For each family/horizon/model the screen reports BOTH views:

* the NATURAL sample set for that family;
* the COMMON sample set, identical across all five families.

plus ``incremental_auc_Xk = AUC(Xk) - AUC(X0)`` on the common set.  A family that
looks better only because it lost the hard samples will show a positive raw
accuracy and a non-positive increment, which is exactly the failure this design is
built to catch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .dataset import FamilySampleTable, baseline_for, design_matrix
from .metrics import (
    block_bootstrap,
    direction_metrics,
    importance_block,
    incremental_block,
    non_overlapping_view,
)

logger = logging.getLogger("agentic_forecaster.v3.screening")

SCREEN_MODELS: tuple[str, ...] = ("LOGISTIC", "HIST_GRADIENT_BOOSTING")

#: Fixed, deterministic settings.  NO hyper-parameter search is performed.
SCREEN_SETTINGS: dict[str, dict] = {
    "LOGISTIC": {"C": 1.0, "max_iter": 2000, "solver": "lbfgs", "random_state": 42},
    "HIST_GRADIENT_BOOSTING": {"loss": "log_loss", "learning_rate": 0.05,
                               "max_iter": 200, "max_leaf_nodes": 31,
                               "min_samples_leaf": 50, "l2_regularization": 1.0,
                               "max_bins": 255, "early_stopping": False,
                               "random_state": 42},
    "PCA_LOGISTIC": {"C": 1.0, "max_iter": 2000, "solver": "lbfgs", "random_state": 42},
}


@dataclass
class ScreenOutcome:
    """One (family, horizon, model, fold, view) screen run."""

    family: str
    horizon: int
    model: str
    fold: str
    view: str
    n_train: int
    n_validation: int
    predictions: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    increment: dict = field(default_factory=dict)
    importance: dict = field(default_factory=dict)
    transform_state: dict = field(default_factory=dict)

    @property
    def objective_id(self) -> str:
        from agentic_forecaster.v2.horizons import objective_id

        return objective_id(self.horizon)


def fit_transform_train_only(model_name: str, x_train: np.ndarray, y_train: np.ndarray,
                             *, pca_components: int | None = None
                             ) -> tuple[dict, dict]:
    """Fit the scaler (and optional PCA) on TRAIN ROWS ONLY.

    Returns the fitted pipeline and a state dict that is persisted with the run, so
    the transformation can be re-applied without ever touching validation data.
    """
    settings = dict(SCREEN_SETTINGS[model_name])
    scaler = StandardScaler().fit(x_train) if x_train.shape[1] else None
    scaled = scaler.transform(x_train) if scaler is not None else x_train
    pca = None
    if pca_components:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=min(pca_components, scaled.shape[1]),
                  random_state=42).fit(scaled)
        scaled = pca.transform(scaled)
    if model_name == "HIST_GRADIENT_BOOSTING":
        # a tree ensemble needs no standardisation; the fact that no scaler was
        # fitted is part of the provenance
        estimator = HistGradientBoostingClassifier(**settings)
        fitted = {"scaler": None, "pca": None, "estimator": estimator}
        state = {"scaler": "NOT_APPLICABLE_TREE_MODEL", "pca": None,
                 "settings": settings, "fitted_on": "TRAIN_ONLY"}
        estimator.fit(x_train, y_train)
        return fitted, state
    estimator = LogisticRegression(C=settings["C"], max_iter=settings["max_iter"],
                                   solver=settings["solver"],
                                   random_state=settings["random_state"])
    estimator.fit(scaled, y_train)
    fitted = {"scaler": scaler, "pca": pca, "estimator": estimator}
    state = {
        "scaler": "StandardScaler" if scaler is not None else None,
        "scaler_fit_on": "TRAIN_ONLY",
        "scaler_n_rows": int(x_train.shape[0]),
        "scaler_n_columns": int(x_train.shape[1]),
        "scaler_mean_head": (None if scaler is None
                             else [round(float(v), 6)
                                   for v in np.asarray(scaler.mean_).ravel()[:5]]),
        "pca": (None if pca is None else
                {"n_components": int(pca.n_components_),
                 "explained_variance_ratio": [round(float(v), 6) for v in
                                              np.asarray(
                                                  pca.explained_variance_ratio_)[:5]]}),
        "settings": settings,
        "fitted_on": "TRAIN_ONLY",
        "validation_influence_on_transform": "NONE",
    }
    return fitted, state


def apply_transform(fitted: dict, x: np.ndarray) -> np.ndarray:
    matrix = x
    if fitted.get("scaler") is not None:
        matrix = fitted["scaler"].transform(matrix)
    if fitted.get("pca") is not None:
        matrix = fitted["pca"].transform(matrix)
    return matrix


def predict_proba(fitted: dict, x: np.ndarray) -> np.ndarray:
    return fitted["estimator"].predict_proba(apply_transform(fitted, x))[:, 1]


def _predictions(table: FamilySampleTable, mask: np.ndarray, proba: np.ndarray,
                 *, model: str, view: str) -> pd.DataFrame:
    frame = table.samples.frame.loc[mask].reset_index(drop=True)
    out = pd.DataFrame({
        "ticker": frame["ticker"].to_numpy(),
        "origin_date": frame["origin_date"].to_numpy(),
        "target_date": frame["target_date"].to_numpy(),
        "y_true": frame["y_direction"].to_numpy(int),
        "future_log_return": frame["future_log_return"].to_numpy(float)
        if "future_log_return" in frame.columns else np.nan,
        "p_up": proba,
        "model": model,
        "family": table.family,
        "view": view,
    })
    return out


def run_screen(table: FamilySampleTable, *, model_name: str, window, fold: str,
               train_mask: np.ndarray, val_mask: np.ndarray,
               pca_components: int | None = None,
               with_importance: bool = False,
               bootstrap_kwargs: dict | None = None) -> ScreenOutcome:
    """Fit on TRAIN, score the validation year, and report the increment over X0."""
    stock_train, exog_train, stock_columns, exog_columns = design_matrix(table, train_mask)
    stock_val, exog_val, _, _ = design_matrix(table, val_mask)
    x_train = np.hstack([stock_train, exog_train]) if exog_train.size else stock_train
    x_val = np.hstack([stock_val, exog_val]) if exog_val.size else stock_val
    frame = table.samples.frame
    y_train = frame.loc[train_mask, "y_direction"].to_numpy(int)

    fitted, state = fit_transform_train_only(model_name, x_train, y_train,
                                            pca_components=pca_components)
    proba = predict_proba(fitted, x_val)
    predictions = _predictions(table, val_mask, proba, model=model_name,
                               view=fold)
    baselines = baseline_for(table, train_mask)
    metrics = direction_metrics(predictions, baselines=baselines,
                                horizon=table.horizon, family=table.family,
                                sample_set="natural")
    metrics["bootstrap"] = block_bootstrap(predictions, horizon=table.horizon,
                                           family=table.family,
                                           **(bootstrap_kwargs or {}))

    importance: dict = {}
    if with_importance and x_val.shape[0] >= 50:
        names = stock_columns + exog_columns
        if model_name == "HIST_GRADIENT_BOOSTING":
            result = permutation_importance(
                fitted["estimator"], x_val, frame.loc[val_mask, "y_direction"].to_numpy(int),
                n_repeats=5, random_state=42, scoring="roc_auc")
            importance = importance_block(names, result.importances_mean,
                                          method="permutation importance (validation, "
                                                 "scoring=roc_auc)",
                                          family=table.family)
        else:
            coefficients = fitted["estimator"].coef_.ravel()
            importance = importance_block(names, coefficients,
                                          method="standardised coefficient magnitude",
                                          family=table.family)

    return ScreenOutcome(
        family=table.family, horizon=table.horizon, model=model_name, fold=fold,
        view="natural", n_train=len(frame.loc[train_mask]),
        n_validation=len(frame.loc[val_mask]), predictions=predictions,
        metrics=metrics, importance=importance,
        transform_state=state | {"n_features": int(x_train.shape[1]),
                                 "stock_features": len(stock_columns),
                                 "exogenous_features": len(exog_columns)})


def run_screen_view(table: FamilySampleTable, *, model_name: str, window, fold: str,
                    train_mask: np.ndarray, val_mask: np.ndarray,
                    view: str) -> ScreenOutcome:
    """Refit on the restricted masks of a COMMON or NON-OVERLAPPING view.

    A view that reuses the natural-sample fit would not be a fair comparison: the
    scaler must see only the rows the view is allowed to train on.
    """
    outcome = run_screen(table, model_name=model_name, window=window, fold=fold,
                         train_mask=train_mask, val_mask=val_mask)
    outcome.view = view
    outcome.metrics["sample_set"] = view
    return outcome


def attach_increment(candidate: ScreenOutcome, control: ScreenOutcome) -> ScreenOutcome:
    """Compute ``AUC(Xk) - AUC(X0)`` from two already-computed outcomes."""
    candidate.increment = incremental_block(candidate.metrics, control.metrics,
                                            family=candidate.family,
                                            horizon=candidate.horizon)
    return candidate


def non_overlap_metrics(predictions: pd.DataFrame, *, horizon: int, baselines: dict,
                        family: str) -> dict:
    """Score the NON-OVERLAPPING subset of an existing prediction frame.

    For ``H > 1`` adjacent origins share their future holding period, so the
    all-origins metrics treat dependent observations as independent.  This view
    keeps every H-th origin per security, from a FIXED starting offset.
    """
    from agentic_forecaster.v2.horizons import non_overlap_mask

    subset = predictions.loc[non_overlap_mask(predictions, horizon=horizon)]
    return non_overlapping_view(subset, horizon=horizon, baselines=baselines,
                                family=family)


def feature_family_gate(natural: dict, common: dict, *, thresholds: dict,
                        family: str, horizon: int) -> dict:
    """Apply the FEATURE-FAMILY GATE of section 30.

    A family/horizon pair SIGNAL-PASSES only when the absolute level, the increment
    over the stock-only control, the year-by-year robustness, the majority-baseline
    requirement, the common-sample support and the non-overlap sensitivity all hold.
    Gates are NOT lowered because nothing passed.
    """
    min_auc = float(thresholds["min_mean_roc_auc"])
    min_increment = float(thresholds["min_min_incremental_auc"])
    min_balanced = float(thresholds["min_mean_balanced_accuracy"])
    min_delta = float(thresholds["min_mean_accuracy_delta"])
    min_years = int(thresholds["min_years_positive_incremental_auc"])
    strong_increment = float(thresholds["strong_min_incremental_auc"])
    strong_auc = float(thresholds["strong_min_mean_roc_auc"])

    criteria = {
        "mean_roc_auc": {"value": natural.get("mean_roc_auc"), "threshold": min_auc,
                         "passed": _at_least(natural.get("mean_roc_auc"), min_auc)},
        "mean_incremental_auc_over_x0": {
            "value": common.get("mean_incremental_auc"), "threshold": min_increment,
            "passed": _at_least(common.get("mean_incremental_auc"), min_increment)},
        "mean_balanced_accuracy": {
            "value": natural.get("mean_balanced_accuracy"), "threshold": min_balanced,
            "passed": _at_least(natural.get("mean_balanced_accuracy"), min_balanced)},
        "mean_accuracy_delta_vs_train_majority": {
            "value": natural.get("mean_baseline_delta"), "threshold": min_delta,
            "passed": _at_least(natural.get("mean_baseline_delta"), min_delta)},
        "positive_incremental_auc_years": {
            "value": common.get("positive_incremental_auc_years"), "threshold": min_years,
            "passed": int(common.get("positive_incremental_auc_years") or 0) >= min_years},
        "common_sample_supports_improvement": {
            "value": common.get("mean_roc_auc"), "threshold": min_auc,
            "passed": _at_least(common.get("mean_roc_auc"), min_auc)},
    }
    non_overlap_auc = natural.get("non_overlapping_mean_roc_auc")
    if non_overlap_auc is not None:
        criteria["non_overlap_sensitivity"] = {
            "value": non_overlap_auc, "threshold": 0.50,
            "passed": _at_least(non_overlap_auc, 0.50)}
    passed = all(item["passed"] for item in criteria.values())
    strongly = (passed and _at_least(natural.get("mean_roc_auc"), strong_auc)
                and _at_least(common.get("mean_incremental_auc"), strong_increment)
                and int(common.get("positive_incremental_auc_years") or 0)
                >= int(thresholds["strong_min_years_positive_incremental_auc"]))
    return {
        "family": family,
        "horizon": int(horizon),
        "signal_passed": bool(passed),
        "classification": ("STRONGLY_PROMISING" if strongly
                           else ("SIGNAL_PASS" if passed else "SIGNAL_FAIL")),
        "criteria": criteria,
        "thresholds": dict(thresholds),
        "gates_lowered_because_nothing_passed": False,
    }


def _at_least(value, floor) -> bool:
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number >= float(floor))


def rank_families(aggregates: dict[tuple[str, int], dict]) -> list[dict]:
    """Rank signal-passing family/horizon pairs.

    Ranking keys: mean common-sample ROC-AUC, then mean incremental AUC, then
    balanced accuracy, then Brier, then worst-year robustness.  Closeness to 65 %
    is never a criterion.
    """
    ranked = []
    for (family, horizon), block in aggregates.items():
        common = block.get("common", {})
        natural = block.get("natural", {})
        ranked.append({
            "family": family,
            "horizon": int(horizon),
            "mean_common_roc_auc": common.get("mean_roc_auc"),
            "mean_incremental_auc": common.get("mean_incremental_auc"),
            "mean_balanced_accuracy": natural.get("mean_balanced_accuracy"),
            "mean_brier": natural.get("mean_brier"),
            "worst_year_roc_auc": natural.get("worst_year_roc_auc"),
        })
    ranked.sort(key=lambda item: (
        -(_num(item["mean_common_roc_auc"])), -(_num(item["mean_incremental_auc"])),
        -(_num(item["mean_balanced_accuracy"])), _num(item["mean_brier"]),
        -(_num(item["worst_year_roc_auc"]))))
    for position, item in enumerate(ranked, start=1):
        item["rank"] = position
    return ranked


def _num(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return number if np.isfinite(number) else float("inf")


def family_contribution_ranking(family_aggregates: dict[str, dict]) -> list[dict]:
    """Rank INFORMATION FAMILIES by mean common-sample incremental ROC-AUC.

    This is the attribution the report is required to make, and it matters more
    than any architecture decision.
    """
    out = []
    for family, block in family_aggregates.items():
        common = block.get("common", {})
        out.append({
            "family": family,
            "mean_common_incremental_auc": common.get("mean_incremental_auc"),
            "mean_common_roc_auc": common.get("mean_roc_auc"),
            "mean_incremental_balanced_accuracy": common.get(
                "mean_incremental_balanced_accuracy"),
            "mean_incremental_brier": common.get("mean_incremental_brier"),
            "positive_incremental_auc_years": common.get("positive_incremental_auc_years"),
            "horizons_evaluated": block.get("n_horizons"),
        })
    out.sort(key=lambda item: -(_num(item["mean_common_incremental_auc"])))
    for position, item in enumerate(out, start=1):
        item["rank"] = position
    return out