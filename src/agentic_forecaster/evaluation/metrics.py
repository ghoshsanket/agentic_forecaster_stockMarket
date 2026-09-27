"""Evaluation metrics: accuracy, Brier, ECE, Precision@3, F1, ROC-AUC."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    roc_auc_score,
)


def expected_calibration_error(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error with equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (proba >= bins[i]) & (proba < bins[i + 1] if i < n_bins - 1 else proba <= bins[i + 1])
        if mask.sum() == 0:
            continue
        conf = proba[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def precision_at_k(y_true: np.ndarray, proba: np.ndarray, k: int = 3) -> float:
    """Precision among the top-k most confident predictions."""
    n = len(y_true)
    if n == 0:
        return 0.0
    k = min(k, n)
    top_idx = np.argsort(proba)[::-1][:k]
    return float(precision_score(y_true[top_idx], np.ones(k), zero_division=0))


def compute_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute the full metric set used in the paper."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)
    pred = (proba >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, proba)),
        "ece": expected_calibration_error(y_true, proba, n_bins),
        "precision_at_3": precision_at_k(y_true, proba, 3),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    return out
