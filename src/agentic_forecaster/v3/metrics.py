"""V3 metrics: direction, breadth, and the INCREMENT over the stock-only control.

THE MOST IMPORTANT NUMBER IN V3
-------------------------------
Raw accuracy cannot decide whether exogenous information helps, because the
exogenous families are evaluated on slightly different samples and because
accuracy is dominated by class imbalance.  So the headline metric is::

    incremental_auc_Xk = AUC(Xk) - AUC(X0)

computed on a COMMON SAMPLE, where X0 is the stock-only control and Xk adds one
information family.  The same increment is reported for balanced accuracy and for
Brier, and every increment is paired with the number of years it stayed positive.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from agentic_forecaster.v2 import horizon_metrics as HM
from agentic_forecaster.v2.horizons import horizon_phrase, objective_id
from agentic_forecaster.v2.metrics import expected_calibration_error

from .dataset import ticker_breadth

logger = logging.getLogger("agentic_forecaster.v3.metrics")

#: Metrics for which an increment over X0 is reported.
INCREMENT_METRICS: tuple[str, ...] = ("roc_auc", "balanced_accuracy", "brier",
                                      "accuracy")


def direction_metrics(predictions: pd.DataFrame, *, baselines: dict,
                      horizon: int, family: str, sample_set: str) -> dict:
    """The full direction metric block for one experiment."""
    if predictions.empty:
        return {"n": 0, "sample_set": sample_set, "family": family}
    y = predictions["y_true"].to_numpy(int)
    p = predictions["p_up"].to_numpy(float)
    pred = (p >= 0.5).astype(int)
    two_class = len(np.unique(y)) == 2
    global_baseline = float(baselines.get("global", float("nan")))
    accuracy = float(accuracy_score(y, pred))
    breadth = ticker_breadth(predictions, baselines.get("per_ticker", {}))
    out = {
        "family": family,
        "horizon": int(horizon),
        "objective_id": objective_id(horizon),
        "horizon_phrase": horizon_phrase(horizon),
        "sample_set": sample_set,
        "n": len(y),
        "n_tickers": int(predictions["ticker"].nunique()),
        "accuracy": accuracy,
        "macro_ticker_accuracy": breadth.get("per_ticker_accuracy") and float(
            np.mean(list(breadth["per_ticker_accuracy"].values()))) or float("nan"),
        "balanced_accuracy": (float(balanced_accuracy_score(y, pred)) if two_class
                              else float("nan")),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if two_class else float("nan"),
        "brier": float(brier_score_loss(y, p)) if two_class else float("nan"),
        "ece": expected_calibration_error(y, p),
        "train_majority_baseline": global_baseline,
        "baseline_delta": (accuracy - global_baseline if np.isfinite(global_baseline)
                           else float("nan")),
        "validation_positive_rate": float(y.mean()),
        "ticker_fraction_beating_own_baseline": breadth["fraction"],
        "n_tickers_beating_own_baseline": breadth["beating"],
    }
    return out


def incremental_block(candidate: dict, control: dict, *, family: str,
                      horizon: int) -> dict:
    """``AUC(Xk) - AUC(X0)`` and its siblings, on the SAME samples.

    Both blocks must describe the same ``sample_set`` and the same ``n``; that is
    what proves they were scored on identical observations.
    """
    comparable = (candidate.get("n") == control.get("n")
                  and candidate.get("sample_set") == control.get("sample_set"))
    out = {
        "family": family,
        "horizon": int(horizon),
        "objective_id": objective_id(horizon),
        "sample_set": candidate.get("sample_set"),
        "n": candidate.get("n"),
        "control_family_sample_set": control.get("sample_set"),
        "comparable": bool(comparable),
        "control_n": control.get("n"),
    }
    for metric in INCREMENT_METRICS:
        left = candidate.get(metric)
        right = control.get(metric)
        delta = (None if left is None or right is None
                 else float(left) - float(right))
        # Brier is LOWER-is-better, so an improvement is a NEGATIVE delta
        out[f"incremental_{metric}"] = delta
        out[f"candidate_{metric}"] = left
        out[f"control_{metric}"] = right
    return out


def aggregate_increments(per_fold: list[dict]) -> dict:
    """Aggregate per-year increments into the gate inputs."""
    rows = [r for r in per_fold if r and r.get("n")]
    if not rows:
        return {"n_folds": 0}

    def _collect(key: str) -> np.ndarray:
        values = []
        for row in rows:
            value = row.get(key)
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(np.nan)
        return np.asarray(values, dtype=float)

    def _mean(values: np.ndarray) -> float:
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    def _std(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(finite.std(ddof=1)) if finite.size > 1 else float("nan")

    increments = {metric: _collect(f"incremental_{metric}") for metric in INCREMENT_METRICS}
    auc = _collect("candidate_roc_auc")
    return {
        "n_folds": len(rows),
        "folds": [str(r.get("fold")) for r in rows],
        "mean_incremental_auc": _mean(increments["roc_auc"]),
        "std_incremental_auc": _std(increments["roc_auc"]),
        "mean_incremental_balanced_accuracy": _mean(increments["balanced_accuracy"]),
        "mean_incremental_brier": _mean(increments["brier"]),
        "mean_incremental_accuracy": _mean(increments["accuracy"]),
        "positive_incremental_auc_years": int(np.nansum(increments["roc_auc"] > 0.0)),
        "mean_roc_auc": _mean(auc),
        "std_roc_auc": _std(auc),
        "worst_year_roc_auc": (float(np.nanmin(auc)) if np.isfinite(auc).any()
                               else float("nan")),
        "worst_year": str(rows[int(np.nanargmin(auc))].get("fold"))
        if np.isfinite(auc).any() else None,
        "mean_accuracy": _mean(_collect("candidate_accuracy")),
        "mean_balanced_accuracy": _mean(_collect("candidate_balanced_accuracy")),
        "mean_brier": _mean(_collect("candidate_brier")),
        # Reporting-only aggregate: the ledger already stores this per fit, it was
        # simply never carried into the aggregate, so the report rendered "n/a".
        "mean_train_majority_baseline": _mean(_collect("train_majority_baseline")),
        "mean_baseline_delta": _mean(_collect("baseline_delta")),
        "years_beating_majority_baseline": int(np.nansum(_collect("baseline_delta") > 0)),
        "mean_ticker_fraction_beating_own_baseline": _mean(
            _collect("ticker_fraction_beating_own_baseline")),
        "years_roc_auc_above_50": int(np.nansum(auc > 0.50)),
    }


def non_overlapping_view(predictions: pd.DataFrame, *, horizon: int, baselines: dict,
                         family: str) -> dict:
    """The non-overlapping sensitivity view for a multi-day horizon."""
    from agentic_forecaster.v2.horizons import non_overlap_mask

    return direction_metrics(
        predictions.loc[non_overlap_mask(predictions, horizon=horizon)],
        baselines=baselines, horizon=horizon, family=family,
        sample_set="non_overlapping")


#: Arguments :func:`agentic_forecaster.v2.horizon_metrics.block_bootstrap_ci` accepts.
_BOOTSTRAP_KEYS = ("date_col", "block_length", "n_bootstrap", "seed", "metrics", "where")


def block_bootstrap(predictions: pd.DataFrame, *, horizon: int, family: str,
                    **kwargs) -> dict:
    """Date-block bootstrap CI, carried over from the verified V2 implementation.

    The config's ``bootstrap`` block is descriptive (``method``) as well as
    operational, so only the operational keys are forwarded.
    """
    settings = {key: value for key, value in kwargs.items() if key in _BOOTSTRAP_KEYS}
    out = HM.block_bootstrap_ci(predictions, **settings)
    out["family"] = family
    out["objective_id"] = objective_id(horizon)
    out["declared_method"] = kwargs.get("method")
    return out


def importance_block(feature_names: list[str], importances: np.ndarray, *,
                     method: str, family: str, top_k: int = 20) -> dict:
    """Rank features by magnitude, with an explicit non-causal caveat.

    Importance is an INTERPRETATION aid computed on validation data.  It is never
    used in this run to engineer further features, and it is not evidence of
    causation.
    """
    values = np.asarray(importances, dtype=float).ravel()
    names = list(feature_names)[:len(values)]
    order = np.argsort(-np.abs(values))[:top_k]
    return {
        "family": family,
        "method": method,
        "n_features": len(values),
        "top_features": [{"feature": names[int(i)],
                          "importance": float(values[int(i)])} for i in order],
        "usage": ("validation-only interpretation; never used to select or engineer "
                  "features in this run, and not a causal claim"),
    }


def correlation_block(features: np.ndarray, feature_names: list[str]) -> dict:
    """Absolute pairwise correlation, so a reader can discount correlated features."""
    if features.shape[1] < 2:
        return {"computed": False, "reason": "fewer than two features"}
    frame = pd.DataFrame(features, columns=feature_names)
    matrix = frame.corr().abs()
    values = matrix.to_numpy(dtype=float)
    np.fill_diagonal(values, 0.0)
    high = np.argwhere(values > 0.95)
    return {
        "computed": True,
        "max_abs_correlation": float(np.nanmax(values)) if values.size else float("nan"),
        "n_pairs_above_0.95": int(len(high) // 2),
        "note": "correlated features share information; importance is not additive",
    }