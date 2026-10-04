"""PRECOVID_LOGISTIC_BASELINE: the cheap linear ceiling check.

Purpose
-------
Before spending compute on sequence models, establish whether the SAME V2
features expose ANY simple linear directional signal in 2017 / 2018.  A regularised
logistic regression on current-origin stationary stock features plus
current-origin context features answers that in seconds, on exactly the same
samples, with preprocessing fitted on TRAIN only.

This is NOT a replacement for the sequence model and it is not a competitor to be
beaten: it is a diagnostic ceiling.  If it cannot beat the train-majority
baseline, no amount of architecture is likely to help.

Causality and regime rules
--------------------------
* fit on TRAIN samples only, score the validation fold only;
* the scaler is fitted on TRAIN rows only;
* no 2019 access in development, and nothing after 2019-12-31 anywhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import metrics as M
from .dataset import SampleTable

logger = logging.getLogger("agentic_forecaster.v2.baseline")

#: Fixed baseline hyper-parameters (not searched).
DEFAULT_C = 1.0
DEFAULT_MAX_ITER = 2000
BASELINE_LABEL = "PRECOVID_LOGISTIC_BASELINE"


@dataclass
class LinearBaselineResult:
    """One fold of the logistic ceiling check."""

    label: str
    fold: str
    n_train: int
    n_val: int
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "fold": self.fold,
            "n_train": self.n_train,
            "n_val": self.n_val,
            **self.metrics,
        }


def _origin_matrix(samples: SampleTable, frame: pd.DataFrame, groups: list[str],
                   percentile: set[str]) -> np.ndarray:
    """Stack the ORIGIN-row features of every sample in ``frame``."""
    blocks = []
    for ticker, row in zip(frame["ticker"], frame["row"], strict=True):
        matrices = samples.arrays.matrices[str(ticker)]
        stock = matrices.stock[int(row), :]
        if not groups:
            blocks.append(stock)
            continue
        context = matrices.context[int(row), :]
        blocks.append(np.concatenate([stock, context]))
    matrix = np.vstack(blocks).astype(np.float64)
    if percentile:
        keep = np.array([c not in percentile for c in groups])
        matrix = matrix[:, keep]
    return matrix


def run_logistic_baseline(samples: SampleTable, *, fold: str, train_mask: np.ndarray,
                          val_mask: np.ndarray, use_context: bool = True,
                          c: float = DEFAULT_C, max_iter: int = DEFAULT_MAX_ITER,
                          random_state: int = 42) -> LinearBaselineResult:
    """Fit on TRAIN, score the validation fold, report the full direction set."""
    arrays = samples.arrays
    percentile = set(arrays.percentile_features) if use_context else set()
    stock_columns = list(arrays.stock_features)
    context_columns = list(arrays.context_features) if use_context else []
    # scaled columns only: percentile ranks stay in [0, 1] and are passed through,
    # exactly as they are for the neural models
    context_scaled = [c for c in context_columns if c not in percentile]

    train_frame = samples.frame.loc[train_mask].reset_index(drop=True)
    val_frame = samples.frame.loc[val_mask].reset_index(drop=True)
    if train_frame.empty or val_frame.empty:
        raise ValueError(f"{fold}: logistic baseline needs both TRAIN and validation rows")

    def design(frame: pd.DataFrame) -> np.ndarray:
        stock = _origin_matrix(samples, frame, [], set())
        if not context_scaled:
            return stock
        context = np.vstack([
            samples.arrays.matrices[str(t)].context[int(r), :]
            for t, r in zip(frame["ticker"], frame["row"], strict=True)
        ]).astype(np.float64)[:, [arrays.context_features.index(c)
                                   for c in context_scaled]]
        return np.concatenate([stock, context], axis=1)

    x_train, x_val = design(train_frame), design(val_frame)
    y_train = train_frame["y_direction"].to_numpy(int)
    y_val = val_frame["y_direction"].to_numpy(int)

    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0.0] = 1.0                     # TRAIN-only, and never divide by 0
    model = LogisticRegression(C=c, max_iter=max_iter, random_state=random_state)
    model.fit((x_train - mean) / scale, y_train)
    proba = model.predict_proba((x_val - mean) / scale)[:, 1]

    predictions = val_frame.copy()
    predictions["p_up"] = proba
    predictions["y_true"] = y_val
    direction = M.direction_metrics(predictions)
    baseline = M.train_majority_baseline(y_train)
    direction["train_majority_baseline"] = baseline
    direction["delta_vs_train_majority_macro"] = (
        direction["accuracy_macro_ticker"] - baseline)
    direction["delta_vs_train_majority_micro"] = direction["accuracy_micro"] - baseline
    p3 = M.precision_at_k(predictions, k=3)

    result = LinearBaselineResult(
        label=BASELINE_LABEL, fold=fold, n_train=len(train_frame), n_val=len(val_frame),
        metrics={
            "n_features": int(x_train.shape[1]),
            "n_stock_features": len(stock_columns),
            "n_context_features": len(context_scaled),
            "percentile_features_passthrough": sorted(percentile),
            "preprocessing_fit_on": "TRAIN_ONLY",
            "c": c,
            "max_iter": max_iter,
            "accuracy_macro_ticker": direction["accuracy_macro_ticker"],
            "accuracy_micro": direction["accuracy_micro"],
            "f1": direction["f1"],
            "balanced_accuracy": direction["balanced_accuracy"],
            "roc_auc": direction["roc_auc"],
            "brier": direction["brier"],
            "ece": direction["ece"],
            "train_majority_baseline": baseline,
            "delta_vs_train_majority_macro": direction["delta_vs_train_majority_macro"],
            "precision_at_3_up": p3["precision_at_3_up"],
            "precision_at_3_down": p3["precision_at_3_down"],
            "per_ticker": direction["per_ticker"],
            "beats_baseline_both_folds": None,
            "n_tickers": direction["n_tickers"],
        })
    logger.info("%s %s: macro=%.4f auc=%.4f baseline=%.4f", BASELINE_LABEL, fold,
                result.metrics["accuracy_macro_ticker"], result.metrics["roc_auc"], baseline)
    return result