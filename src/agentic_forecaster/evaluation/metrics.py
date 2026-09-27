"""Evaluation metrics: accuracy, Brier, ECE, cross-sectional Precision@3, F1, ROC-AUC.

Precision@3 is CROSS-SECTIONAL BY DATE (PAPER-DEFINED):
    for every eligible prediction date:
        UP P@3:   sort available tickers by p_up descending, take top 3,
                  fraction with actual y=1
        DOWN P@3: sort available tickers by p_up ascending, take bottom 3,
                  fraction with actual y=0
    average across valid dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
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


def precision_at_3_cross_sectional(
    df: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    proba_col: str = "p_up",
    y_col: str = "y",
    k: int = 3,
) -> dict:
    """Cross-sectional Precision@3 UP and DOWN by date.

    Parameters
    ----------
    df : DataFrame with columns date, ticker, p_up, y.
    k : number of top/bottom tickers to select per date.

    Returns
    -------
    dict with precision_at_3_up, precision_at_3_down, n_dates, selections.
    """
    up_precisions = []
    down_precisions = []
    selections = []

    for date, group in df.groupby(date_col):
        if len(group) < k:
            continue
        sorted_desc = group.sort_values(proba_col, ascending=False)
        top_k = sorted_desc.head(k)
        up_prec = float((top_k[y_col] == 1).mean())
        up_precisions.append(up_prec)

        sorted_asc = group.sort_values(proba_col, ascending=True)
        bottom_k = sorted_asc.head(k)
        down_prec = float((bottom_k[y_col] == 0).mean())
        down_precisions.append(down_prec)

        selections.append({
            "date": str(date),
            "up_tickers": top_k[ticker_col].tolist(),
            "up_precision": up_prec,
            "down_tickers": bottom_k[ticker_col].tolist(),
            "down_precision": down_prec,
        })

    return {
        "precision_at_3_up": float(np.mean(up_precisions)) if up_precisions else 0.0,
        "precision_at_3_down": float(np.mean(down_precisions)) if down_precisions else 0.0,
        "n_dates": len(up_precisions),
        "selections": selections,
    }


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
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    return out
