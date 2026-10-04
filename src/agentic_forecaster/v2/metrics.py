"""V2 metrics: direction, coverage, cross-sectional selection, return and rank.

Every number reported here is reported together with its sample count and, for
selective accuracy, its COVERAGE.  A selective accuracy is never called overall
accuracy: the long-term 65% aspiration may be reached either as all-sample
accuracy or as explicitly labelled selective accuracy at a stated coverage, and
the distinction has to survive into the report.

Direction
    micro accuracy, macro-ticker accuracy, per-ticker accuracy, F1, balanced
    accuracy, ROC-AUC, Brier, ECE and the train-majority baseline.

Coverage
    accuracy and F1 at 100 / 75 / 50 / 30 / 20 / 10 percent coverage, ranked by
    ``|sigmoid(direction_logit) - 0.5|`` -- deliberately the simplest possible
    confidence, with no learned confidence head in this first programme.

Cross-sectional selection
    P@3 up / P@3 down, computed per date.

Return and rank
    MAE, RMSE, Pearson, Spearman; and for the rank head, MAE, daily Spearman
    rank IC and mean rank IC.  A negative IC is reported, never hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

#: Coverage levels for the accuracy-versus-coverage curve.
COVERAGE_LEVELS: tuple[float, ...] = (1.0, 0.75, 0.5, 0.3, 0.2, 0.1)

#: Minimum sample count before a selective-accuracy point is treated as
#: meaningful.  Together with the 10% coverage floor this is what stops a
#: 65% figure computed on a handful of observations being reported.
MIN_SELECTIVE_OBSERVATIONS = 200
MIN_SELECTIVE_COVERAGE = 0.10

#: Accuracies reported as "maximum meaningful coverage" in the final report.
COVERAGE_TARGETS: tuple[float, ...] = (0.60, 0.62, 0.65)


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray,
                               n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        upper_open = i < n_bins - 1
        mask = (proba >= edges[i]) & (proba < edges[i + 1] if upper_open
                                      else proba <= edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / len(y_true)) * abs(float(y_true[mask].mean())
                                            - float(proba[mask].mean()))
    return float(ece)


def macro_ticker_accuracy(frame: pd.DataFrame, *, ticker_col: str = "ticker",
                          y_col: str = "y_true", p_col: str = "p_up") -> float:
    """Mean of per-ticker accuracies: every security counts once."""
    if frame.empty:
        return float("nan")
    per_ticker = frame.groupby(ticker_col).apply(
        lambda g: accuracy_score(g[y_col].to_numpy(int),
                                 (g[p_col].to_numpy(float) >= 0.5).astype(int)),
        include_groups=False)
    return float(per_ticker.mean())


def direction_metrics(frame: pd.DataFrame, *, ticker_col: str = "ticker",
                      y_col: str = "y_true", p_col: str = "p_up",
                      ece_bins: int = 10) -> dict:
    """The full direction metric set for one validation split."""
    if frame.empty:
        return {"n": 0}
    y = frame[y_col].to_numpy(int)
    p = frame[p_col].to_numpy(float)
    pred = (p >= 0.5).astype(int)
    out: dict[str, float] = {
        "n": len(y),
        "n_tickers": int(frame[ticker_col].nunique()),
        "accuracy_micro": float(accuracy_score(y, pred)),
        "accuracy_macro_ticker": macro_ticker_accuracy(frame, ticker_col=ticker_col,
                                                      y_col=y_col, p_col=p_col),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred))
        if len(np.unique(y)) == 2 else float("nan"),
        "brier": float(brier_score_loss(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "ece": expected_calibration_error(y, p, ece_bins),
        "positive_rate": float(y.mean()),
        "mean_confidence": float(np.abs(p - 0.5).mean()),
    }
    out["roc_auc"] = (float(roc_auc_score(y, p)) if len(np.unique(y)) == 2
                      else float("nan"))
    per_ticker = frame.groupby(ticker_col).apply(
        lambda g: {
            "n": len(g),
            "accuracy": float(accuracy_score(g[y_col].to_numpy(int),
                                             (g[p_col].to_numpy(float) >= 0.5).astype(int))),
            "positive_rate": float(g[y_col].mean()),
        },
        include_groups=False)
    out["per_ticker"] = {str(k): v for k, v in per_ticker.items()}
    return out


def train_majority_baseline(train_labels: np.ndarray) -> float:
    """Accuracy of always predicting the majority TRAIN class."""
    labels = np.asarray(train_labels).astype(int)
    if len(labels) == 0:
        return float("nan")
    return float(max(labels.mean(), 1.0 - labels.mean()))


def coverage_curve(frame: pd.DataFrame, *, levels=COVERAGE_LEVELS,
                   ticker_col: str = "ticker", y_col: str = "y_true",
                   p_col: str = "p_up") -> pd.DataFrame:
    """Accuracy and F1 at decreasing coverage, ranked by directional confidence.

    Confidence is ``|p_up - 0.5|``: the more decisive the directional prediction,
    the earlier the observation is kept.
    """
    rows: list[dict] = []
    if frame.empty:
        return pd.DataFrame(columns=["coverage_target", "n", "coverage_retained",
                                     "accuracy", "f1", "n_tickers"])
    confidence = np.abs(frame[p_col].to_numpy(float) - 0.5)
    order = np.argsort(-confidence, kind="stable")
    n = len(frame)
    y = frame[y_col].to_numpy(int)
    p = frame[p_col].to_numpy(float)
    tickers = frame[ticker_col].to_numpy()
    for level in levels:
        keep_n = round(level * n)
        keep_n = max(keep_n, 1)
        idx = order[:keep_n]
        kept_pred = (p[idx] >= 0.5).astype(int)
        rows.append({
            "coverage_target": float(level),
            "n": int(keep_n),
            "coverage_retained": float(keep_n / n),
            "accuracy": float(accuracy_score(y[idx], kept_pred)),
            "f1": float(f1_score(y[idx], kept_pred, zero_division=0)),
            "n_tickers": len(np.unique(tickers[idx])),
            "mean_confidence": float(confidence[idx].mean()),
        })
    return pd.DataFrame(rows)


def max_coverage_at_accuracy(frame: pd.DataFrame, threshold: float, *,
                             levels=COVERAGE_LEVELS, ticker_col: str = "ticker",
                             y_col: str = "y_true", p_col: str = "p_up",
                             min_observations: int = MIN_SELECTIVE_OBSERVATIONS,
                             min_coverage: float = MIN_SELECTIVE_COVERAGE) -> dict:
    """Highest coverage whose accuracy reaches ``threshold``, if it is meaningful.

    A point counts only when it retains at least ``min_observations`` rows AND at
    least ``min_coverage`` of the split.  Otherwise the result is reported as
    unavailable rather than as a number that a reader could over-interpret.
    """
    curve = coverage_curve(frame, levels=levels, ticker_col=ticker_col, y_col=y_col,
                           p_col=p_col)
    if curve.empty:
        return {"threshold": threshold, "coverage": None, "reason": "no predictions"}
    eligible = curve.loc[(curve["accuracy"] >= threshold)
                         & (curve["n"] >= min_observations)
                         & (curve["coverage_retained"] >= min_coverage)]
    if eligible.empty:
        best = curve.loc[curve["accuracy"].idxmax()] if len(curve) else None
        return {
            "threshold": float(threshold),
            "coverage": None,
            "accuracy": None,
            "n": None,
            "reason": ("no coverage level reaches the threshold with at least "
                       f"{min_observations} observations and {min_coverage:.0%} coverage"),
            "max_accuracy_seen": None if best is None else float(best["accuracy"]),
        }
    best = eligible.sort_values(["coverage_retained", "accuracy"],
                                ascending=[False, False]).iloc[0]
    return {
        "threshold": float(threshold),
        "coverage": float(best["coverage_retained"]),
        "coverage_target": float(best["coverage_target"]),
        "accuracy": float(best["accuracy"]),
        "n": int(best["n"]),
        "n_tickers": int(best["n_tickers"]),
        "reason": "meaningful",
    }


def precision_at_k(frame: pd.DataFrame, *, k: int = 3, date_col: str = "target_date",
                   ticker_col: str = "ticker", p_col: str = "p_up",
                   y_col: str = "y_true") -> dict:
    """Cross-sectional P@k up and down, computed per date.

    Up: the ``k`` highest-probability securities on a date, scored by how many
    actually rose.  Down: the ``k`` lowest, scored by how many actually fell.
    """
    ups: list[float] = []
    downs: list[float] = []
    rows: list[dict] = []
    for date, group in frame.groupby(date_col, sort=True):
        if len(group) < k:
            continue
        ordered_up = group.sort_values(p_col, ascending=False).head(k)
        ordered_down = group.sort_values(p_col, ascending=True).head(k)
        up_precision = float((ordered_up[y_col].to_numpy(int) == 1).mean())
        down_precision = float((ordered_down[y_col].to_numpy(int) == 0).mean())
        ups.append(up_precision)
        downs.append(down_precision)
        rows.append({
            "target_date": str(pd.Timestamp(date).date()),
            f"p_at_{k}_up": up_precision,
            f"p_at_{k}_down": down_precision,
            "up_tickers": ",".join(ordered_up[ticker_col].astype(str)),
            "down_tickers": ",".join(ordered_down[ticker_col].astype(str)),
            "n_available": len(group),
        })
    return {
        f"precision_at_{k}_up": float(np.mean(ups)) if ups else float("nan"),
        f"precision_at_{k}_down": float(np.mean(downs)) if downs else float("nan"),
        "n_dates": len(ups),
        "k": k,
        "selections": rows,
    }


def return_metrics(frame: pd.DataFrame, *, y_col: str = "y_true",
                   pred_col: str = "prediction",
                   raw_y_col: str = "raw_next_return",
                   raw_pred_col: str = "predicted_next_return") -> dict:
    """MAE / RMSE / Pearson / Spearman for the return head.

    The head predicts a VOLATILITY-NORMALISED return, so the normalised errors
    are the training-scale numbers while the de-normalised errors are what a
    reader of a return forecast expects.  Both are reported; neither is silently
    substituted for the other.
    """
    valid = frame.loc[frame[[pred_col, y_col]].notna().all(axis=1)]
    if valid.empty:
        return {"n": 0}
    y = valid[y_col].to_numpy(float)
    p = valid[pred_col].to_numpy(float)
    error = p - y
    out: dict[str, float] = {
        "n": len(y),
        "return_mae": float(np.mean(np.abs(error))),
        "return_rmse": float(np.sqrt(np.mean(error ** 2))),
        "return_pearson": _safe_corr(y, p, method="pearson"),
        "return_spearman": _safe_corr(y, p, method="spearman"),
    }
    if raw_y_col in valid.columns and raw_pred_col in valid.columns:
        raw_valid = valid.loc[valid[[raw_y_col, raw_pred_col]].notna().all(axis=1)]
        if len(raw_valid):
            raw_y = raw_valid[raw_y_col].to_numpy(float)
            raw_p = raw_valid[raw_pred_col].to_numpy(float)
            raw_error = raw_p - raw_y
            out.update({
                "n_raw": len(raw_y),
                "raw_return_mae": float(np.mean(np.abs(raw_error))),
                "raw_return_rmse": float(np.sqrt(np.mean(raw_error ** 2))),
                "raw_return_spearman": _safe_corr(raw_y, raw_p, method="spearman"),
            })
    return out


def _safe_corr(x: np.ndarray, y: np.ndarray, *, method: str) -> float:
    """Correlation that returns NaN instead of raising on degenerate input."""
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    frame = pd.DataFrame({"x": x, "y": y})
    value = frame["x"].corr(frame["y"], method=method)
    return float(value) if pd.notna(value) else float("nan")


def _rank_ic(daily: pd.DataFrame, *, pred_col: str, y_col: str,
             date_col: str = "target_date") -> pd.DataFrame:
    """Per-date Spearman rank IC between the prediction and the realised rank."""
    rows = []
    for date, group in daily.groupby(date_col, sort=True):
        if len(group) < 3:
            continue
        ic = group[pred_col].corr(group[y_col], method="spearman")
        rows.append({"target_date": str(pd.Timestamp(date).date()),
                     "rank_ic": float(ic) if pd.notna(ic) else float("nan"),
                     "n": len(group)})
    return pd.DataFrame(rows, columns=["target_date", "rank_ic", "n"])


def rank_metrics(frame: pd.DataFrame, *, pred_col: str = "rank_prediction",
                 y_col: str = "y_rank", date_col: str = "target_date") -> dict:
    """MAE plus daily Spearman rank IC for the cross-sectional rank head."""
    valid = frame.loc[frame[[pred_col, y_col]].notna().all(axis=1)]
    if valid.empty:
        return {"n": 0}
    error = valid[pred_col].to_numpy(float) - valid[y_col].to_numpy(float)
    ic = _rank_ic(valid, pred_col=pred_col, y_col=y_col, date_col=date_col)
    finite_ic = ic["rank_ic"].dropna() if not ic.empty else pd.Series(dtype=float)
    return {
        "n": len(valid),
        "rank_mae": float(np.mean(np.abs(error))),
        "rank_rmse": float(np.sqrt(np.mean(error ** 2))),
        "daily_rank_ic": {str(r["target_date"]): r["rank_ic"]
                          for r in ic.to_dict("records")} if not ic.empty else {},
        "mean_rank_ic": float(finite_ic.mean()) if len(finite_ic) else float("nan"),
        "std_rank_ic": float(finite_ic.std(ddof=1)) if len(finite_ic) > 1 else float("nan"),
        "n_days_rank_ic": len(finite_ic),
        "positive_rank_ic_fraction": float((finite_ic > 0).mean())
        if len(finite_ic) else float("nan"),
    }


def multi_head_agreement(frame: pd.DataFrame, *, y_col: str = "y_true",
                         logit_col: str = "direction_logit",
                         return_col: str = "return_prediction",
                         rank_col: str = "rank_prediction") -> dict:
    """ANALYSIS-ONLY agreement indicator across the three heads.

    Reports the accuracy and the coverage of the subset on which the
    classification direction, the sign of the return prediction and the rank
    prediction all agree.  It is diagnostic only: it never trains or selects a
    model in this programme.
    """
    required = [logit_col, return_col, rank_col, y_col]
    if any(c not in frame.columns for c in required) or frame.empty:
        return {"available": False, "reason": "multi-task predictions absent"}
    sub = frame.dropna(subset=required)
    if sub.empty:
        return {"available": False, "reason": "no complete multi-head predictions"}
    from_direction = (sub[logit_col].to_numpy(float) >= 0.0).astype(int)
    from_return = (sub[return_col].to_numpy(float) >= 0.0).astype(int)
    from_rank = (sub[rank_col].to_numpy(float) >= 0.5).astype(int)
    agree = (from_direction == from_return) & (from_return == from_rank)
    y = sub[y_col].to_numpy(int)
    n_total = len(sub)
    n_agree = int(agree.sum())
    correct = int((y[agree] == from_direction[agree]).sum()) if n_agree else 0
    return {
        "available": True,
        "n": n_total,
        "n_agree": n_agree,
        "coverage": float(n_agree / n_total) if n_total else float("nan"),
        "accuracy_when_agreeing": float(correct / n_agree) if n_agree else float("nan"),
        "accuracy_all": float((from_direction == y).mean()),
        "pairwise_agreement": {
            "direction_vs_return": float((from_direction == from_return).mean()),
            "direction_vs_rank": float((from_direction == from_rank).mean()),
            "return_vs_rank": float((from_return == from_rank).mean()),
        },
        "note": "diagnostic only; never used to train or select a variant",
    }


def summarise_selection_metrics(frame: pd.DataFrame, *, k: int = 3,
                                coverage_levels=COVERAGE_LEVELS) -> dict:
    """P@k plus the coverage targets in one block."""
    out = {"p_at_k": precision_at_k(frame, k=k)}
    out["selective_accuracy"] = {
        f"coverage_at_{int(target * 100)}pct": max_coverage_at_accuracy(
            frame, target, levels=coverage_levels)
        for target in COVERAGE_TARGETS
    }
    return out