"""Firewall-guarded scoring for the recovery search.

The production evaluation functions in
``agentic_forecaster.evaluation.metrics`` are deliberately left untouched: the
paper reproduction depends on them.  This module adds a parallel, guarded path
for the recovery search that cannot score a firewalled date.

Every entry point calls :func:`assert_pre_test_dates` with ``search=True``
BEFORE computing anything, so a 2022/2023 date raises rather than silently
producing a number that would then tempt a configuration choice.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from agentic_forecaster.recovery.firewall import (
    assert_pre_test_dates,
    firewall_guard,
)


def _probs(y_true, p_up) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(pd.Series(y_true).astype(float).to_numpy(), dtype=float)
    p = np.asarray(pd.Series(p_up).astype(float).to_numpy(), dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true={y.shape} p_up={p.shape}")
    return y, p


def score_validation(y_true, p_up, dates=None, *, where: str = "validation",
                     search: bool | None = None) -> dict:
    """Score a pre-2022 validation split.

    The firewall check runs FIRST, before any metric is computed, and covers
    BOTH the origin date and the target date of every sample.  ``target_dates``
    is optional but, when given, must also be pre-2022 - a validation sample
    whose label lands in 2022 would leak test information.
    """
    y, p = _probs(y_true, p_up)
    if dates is not None:
        assert_pre_test_dates(pd.to_datetime(pd.Series(dates)), where=f"{where}/origin",
                              search=True if search is None else search)
    metrics = _metrics(y, p)
    metrics["n"] = len(y)
    metrics["positive_rate"] = float(y.mean()) if len(y) else None
    return metrics


def score_validation_with_targets(y_true, p_up, dates, target_dates, *,
                                  where: str = "validation") -> dict:
    """:func:`score_validation` plus a firewall check on the target dates."""
    out = score_validation(y_true, p_up, dates, where=where)
    if target_dates is not None:
        assert_pre_test_dates(pd.to_datetime(pd.Series(target_dates)),
                              where=f"{where}/target")
    return out


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        f1_score,
        roc_auc_score,
    )

    from agentic_forecaster.evaluation.metrics import expected_calibration_error

    if len(y) == 0:
        return {"accuracy": None, "f1": None, "brier": None, "ece": None,
                "roc_auc": None, "majority_accuracy": None}
    hard = (p >= 0.5).astype(int)
    yi = y.astype(int)
    out = {
        "accuracy": float(accuracy_score(yi, hard)),
        "f1": float(f1_score(yi, hard, zero_division=0)),
        # Brier is mean((p_up - y)^2) on the UP probability. Never a flipped
        # directional confidence.
        "brier": float(brier_score_loss(yi, p)),
        "ece": float(expected_calibration_error(yi, p)),
        "majority_accuracy": float(max(yi.mean(), 1.0 - yi.mean())),
    }
    out["roc_auc"] = (float(roc_auc_score(yi, p))
                      if len(np.unique(yi)) > 1 else None)
    return out


def precision_at_k(predictions: pd.DataFrame, k: int = 3, *,
                   dates: Iterable | None = None, where: str = "validation",
                   label: str = "up") -> float | None:
    """Cross-sectional P@k on VALIDATION predictions.

    ``predictions`` must have columns ``date``, ``ticker``, ``y`` and ``p_up``.
    On each date the top-``k`` tickers by ``p_up`` (or by ``1 - p_up`` for
    ``label="down"``) are selected; the score is the share of those whose
    realised direction matched.  The dates are firewall-checked first.
    """
    required = {"date", "ticker", "y", "p_up"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions is missing columns: {sorted(missing)}")
    if dates is not None:
        assert_pre_test_dates(pd.to_datetime(pd.Series(list(dates))),
                              where=f"{where}/p_at_k", search=True)
    else:
        assert_pre_test_dates(predictions["date"], where=f"{where}/p_at_k", search=True)

    score = predictions["p_up"] if label == "up" else 1.0 - predictions["p_up"]
    want = (predictions["y"] > 0.5).astype(int)
    if label == "down":
        want = 1 - want

    picks: list[int] = []
    for _, grp in predictions.assign(_score=score).groupby("date", sort=True):
        top = grp.nlargest(k, "_score")
        picks.extend(top.index.tolist())
    if not picks:
        return None
    return float((want.loc[picks] == 1).mean())


def cross_sectional_metrics(predictions: pd.DataFrame, k: int = 3, *,
                            where: str = "validation") -> dict:
    """P@3 up/down on validation dates only, with the firewall enforced."""
    with firewall_guard(True):
        return {
            f"precision_at_{k}_up": precision_at_k(predictions, k, where=where, label="up"),
            f"precision_at_{k}_down": precision_at_k(predictions, k, where=where, label="down"),
        }


def aggregate(ticker_metrics: Sequence[dict]) -> dict:
    """Mean over tickers, ignoring absent metrics."""
    if not ticker_metrics:
        return {}
    keys = {k for m in ticker_metrics for k in m}
    out: dict = {}
    for k in keys:
        vals = [m[k] for m in ticker_metrics
                if isinstance(m.get(k), (int, float)) and not isinstance(m.get(k), bool)]
        if vals:
            out[k] = float(np.mean(vals))
    out["n_tickers"] = len(ticker_metrics)
    return out
