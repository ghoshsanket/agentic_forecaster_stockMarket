"""Multi-horizon metrics: direction, breadth, block-aware uncertainty, selective
accuracy and the horizon magnitude diagnostic.

Every number is reported together with its sample count, its ticker count and,
where it matters, its COVERAGE.  Two rules are enforced structurally:

* the baseline is always the TRAIN-majority predictor.  A validation-oracle
  majority class would leak the answer into the baseline;
* a multi-day number is never called "next-day accuracy".  The wording comes from
  :func:`agentic_forecaster.v2.horizons.horizon_phrase`.

OVERLAPPING MULTI-DAY TARGETS
-----------------------------
For ``H > 1`` adjacent samples share their future holding period, so the
ALL-ORIGINS metrics treat dependent observations as independent.  Every horizon
is therefore also reported on a NON-OVERLAPPING origin subset (origins spaced at
least ``H`` trading observations apart) and, where it matters, with a
DATE-BLOCK BOOTSTRAP confidence interval rather than an IID bootstrap over
individual overlapping rows.
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

from . import metrics as M

logger = logging.getLogger("agentic_forecaster.v2.horizon_metrics")

#: A selective point counts as meaningful only with coverage >= 20% AND at least
#: 500 retained observations pooled across the scored years.
MIN_MEANINGFUL_COVERAGE = 0.20
MIN_MEANINGFUL_OBSERVATIONS = 500

#: Coverage levels for the accuracy-versus-coverage table.
COVERAGE_LEVELS: tuple[float, ...] = (1.0, 0.75, 0.5, 0.3, 0.2, 0.1)

#: Block bootstrap settings.  The block length is expressed in TRADING DATES and
#: is comfortably longer than the longest horizon (10 observations), so a
#: resampled block carries the dependence structure of overlapping labels.
DEFAULT_BLOCK_LENGTH = 21
DEFAULT_N_BOOTSTRAP = 400
BOOTSTRAP_SEED = 42


def ticker_train_majority(train_frame: pd.DataFrame) -> dict[str, float]:
    """Per-ticker TRAIN-majority accuracy: each security's own baseline."""
    if train_frame.empty:
        return {}
    return {str(ticker): float(M.train_majority_baseline(group["y_direction"].to_numpy()))
            for ticker, group in train_frame.groupby("ticker", sort=True)}


def direction_block(predictions: pd.DataFrame, *, train_majority: float,
                    ticker_baselines: dict[str, float] | None = None,
                    horizon: int = 1, sample_label: str = "all_valid_origins"
                    ) -> dict:
    """The full direction metric set for one horizon / model / fold.

    ``predictions`` needs ``ticker``, ``y_true`` and ``p_up``; ``future_log_return``
    is optional and only feeds the magnitude diagnostic.
    """
    from . import horizons as HZ

    if predictions.empty:
        return {"n": 0, "sample_set": sample_label}
    y = predictions["y_true"].to_numpy(int)
    p = predictions["p_up"].to_numpy(float)
    pred = (p >= 0.5).astype(int)
    ticker_baselines = ticker_baselines or {}

    per_ticker_accuracy = predictions.groupby("ticker", sort=True).apply(
        lambda g: float(accuracy_score(g["y_true"].to_numpy(int),
                                       (g["p_up"].to_numpy(float) >= 0.5).astype(int))),
        include_groups=False)
    beating, breadth_details = [], {}
    for ticker, accuracy in per_ticker_accuracy.items():
        baseline = ticker_baselines.get(str(ticker))
        if baseline is None or not np.isfinite(baseline):
            continue
        wins = bool(accuracy > baseline)
        beating.append(wins)
        breadth_details[str(ticker)] = {"accuracy": float(accuracy),
                                         "train_majority_baseline": float(baseline),
                                         "beats_own_baseline": wins}

    out = {
        "sample_set": sample_label,
        "horizon": int(horizon),
        "objective_id": HZ.objective_id(horizon),
        "horizon_phrase": HZ.horizon_phrase(horizon),
        "n": len(y),
        "n_tickers": int(predictions["ticker"].nunique()),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_ticker_accuracy": float(per_ticker_accuracy.mean()),
        "balanced_accuracy": (float(balanced_accuracy_score(y, pred))
                              if len(np.unique(y)) == 2 else float("nan")),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "brier": float(brier_score_loss(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "ece": M.expected_calibration_error(y, p),
        "train_majority_baseline": float(train_majority),
        "baseline_delta": float(accuracy_score(y, pred) - train_majority),
        "macro_baseline_delta": (float(per_ticker_accuracy.mean() - train_majority)
                                 if np.isfinite(train_majority) else float("nan")),
        "validation_positive_rate": float(y.mean()),
        "train_positive_rate": (None if "y_train" not in predictions
                                else float(predictions["y_train"].mean())),
        "mean_confidence": float(np.abs(p - 0.5).mean()),
        "tickers_beating_own_baseline": int(sum(beating)),
        "ticker_fraction_beating_own_baseline": (float(np.mean(beating))
                                                  if beating else float("nan")),
        "ticker_breadth": breadth_details,
    }
    out.update(return_magnitude_stats(predictions))
    out.update(confidence_vs_magnitude(predictions))
    return out


def return_magnitude_stats(predictions: pd.DataFrame, *,
                           return_col: str = "future_log_return") -> dict:
    """ANALYSIS ONLY: the size of the move the model is being asked to call.

    This explains whether predictability changes because the signal-to-noise ratio
    improves with the horizon.  It NEVER filters samples.
    """
    if return_col not in predictions.columns:
        return {}
    values = pd.to_numeric(predictions[return_col], errors="coerce").dropna().to_numpy(float)
    if values.size == 0:
        return {}
    absolute = np.abs(values)
    return {
        "future_return_median": float(np.median(values)),
        "future_return_mean": float(np.mean(values)),
        "future_return_std": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
        "abs_future_return_median": float(np.median(absolute)),
        "abs_future_return_mean": float(np.mean(absolute)),
        "abs_future_return_std": (float(np.std(absolute, ddof=1))
                                  if absolute.size > 1 else float("nan")),
        "class_balance_up_fraction": float((values > 0).mean()),
        "magnitude_note": ("diagnostic only; samples are never filtered on the size of "
                           "the future move"),
    }


def confidence_vs_magnitude(predictions: pd.DataFrame, *,
                            return_col: str = "future_log_return") -> dict:
    """ANALYSIS ONLY: does the model become more confident on larger moves?"""
    if return_col not in predictions.columns:
        return {}
    block = predictions.loc[:, ["p_up", return_col]].replace([np.inf, -np.inf], np.nan)
    block = block.dropna()
    if len(block) < 3:
        return {"confidence_vs_magnitude": {"available": False,
                                           "reason": "too few complete rows"}}
    confidence = (block["p_up"].to_numpy(float) - 0.5).__abs__()
    magnitude = block[return_col].to_numpy(float).__abs__()
    frame = pd.DataFrame({"confidence": confidence, "magnitude": magnitude})
    pearson = float(frame["confidence"].corr(frame["magnitude"], method="pearson"))
    spearman = float(frame["confidence"].corr(frame["magnitude"], method="spearman"))
    return {
        "confidence_vs_magnitude": {
            "available": True,
            "n": len(frame),
            "pearson": pearson,
            "spearman": spearman,
            "question": ("does the model naturally become more confident on larger "
                         "subsequent moves?"),
            "usage": "analysis only; never used to select or filter predictions",
        }
    }


# ---------------------------------------------------------------------------
# block-aware uncertainty
# ---------------------------------------------------------------------------

def block_bootstrap_ci(predictions: pd.DataFrame, *, date_col: str = "origin_date",
                      block_length: int = DEFAULT_BLOCK_LENGTH,
                      n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
                      seed: int = BOOTSTRAP_SEED,
                      metrics: tuple[str, ...] = ("accuracy", "roc_auc"),
                      where: str = "block bootstrap") -> dict:
    """95% confidence intervals from a MOVING-BLOCK bootstrap over dates.

    An IID bootstrap over individual rows would treat overlapping multi-day
    targets as independent and would therefore report intervals that are far too
    narrow.  This resamples contiguous blocks of trading DATES with replacement
    (circular moving blocks), which preserves the local dependence created by
    overlapping holding periods.
    """
    if predictions.empty:
        return {"available": False, "reason": "no predictions"}
    frame = predictions.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    dates = np.sort(frame[date_col].unique())
    n_dates = len(dates)
    if n_dates < max(2, block_length):
        return {"available": False,
                "reason": f"only {n_dates} distinct dates; block length {block_length} too long"}
    position = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    date_index = frame[date_col].map(position).to_numpy(int)
    y = frame["y_true"].to_numpy(int)
    p = frame["p_up"].to_numpy(float)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_dates / block_length))
    max_start = n_dates - block_length + 1
    collected: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(int(n_bootstrap)):
        starts = rng.integers(0, max_start, size=n_blocks)
        drawn = np.concatenate([(np.arange(s, s + block_length) % n_dates)
                                for s in starts])[:n_dates]
        weight = np.bincount(date_index[drawn], minlength=n_dates).astype(float)
        sample_weight = weight[date_index]
        if sample_weight.sum() == 0:
            continue
        keep = sample_weight > 0
        if len(np.unique(y[keep])) < 2:
            continue
        for name in metrics:
            if name == "accuracy":
                collected[name].append(float(accuracy_score(
                    y[keep], (p[keep] >= 0.5).astype(int),
                    sample_weight=sample_weight[keep])))
            elif name == "roc_auc":
                collected[name].append(float(roc_auc_score(
                    y[keep], p[keep], sample_weight=sample_weight[keep])))
            elif name == "balanced_accuracy":
                collected[name].append(float(balanced_accuracy_score(
                    y[keep], (p[keep] >= 0.5).astype(int),
                    sample_weight=sample_weight[keep])))
            elif name == "brier":
                collected[name].append(float(brier_score_loss(
                    y[keep], p[keep], sample_weight=sample_weight[keep])))
            else:
                raise ValueError(f"unsupported bootstrap metric {name!r}")

    out: dict = {
        "method": "circular moving-block bootstrap over trading dates",
        "block_length_dates": int(block_length),
        "n_bootstrap_requested": int(n_bootstrap),
        "n_dates": int(n_dates),
        "n_observations": len(frame),
        "seed": int(seed),
        "why_not_iid": ("multi-day targets overlap in their future holding period, so an "
                        "IID bootstrap over rows would understate the interval"),
    }
    for name, values in collected.items():
        if not values:
            out[name] = {"available": False}
            continue
        array = np.asarray(values, dtype=float)
        out[name] = {
            "available": True,
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if len(array) > 1 else float("nan"),
            "ci_low_95": float(np.percentile(array, 2.5)),
            "ci_high_95": float(np.percentile(array, 97.5)),
            "n_effective_replicates": len(array),
        }
    logger.info("%s: block bootstrap over %d dates -> %s", where, n_dates,
                {k: v.get("ci_low_95") for k, v in out.items() if isinstance(v, dict)})
    return out


# ---------------------------------------------------------------------------
# selective accuracy
# ---------------------------------------------------------------------------

def confidence_values(probabilities) -> np.ndarray:
    """``|p_up - 0.5|``: the simplest possible confidence, no learned head."""
    return np.abs(np.asarray(probabilities, dtype=float) - 0.5)


def selective_table(predictions: pd.DataFrame, *,
                    levels=COVERAGE_LEVELS) -> pd.DataFrame:
    """Accuracy / balanced accuracy / F1 at decreasing coverage."""
    if predictions.empty:
        return pd.DataFrame(columns=["coverage_target", "n", "coverage_retained",
                                     "accuracy", "balanced_accuracy", "f1"])
    y = predictions["y_true"].to_numpy(int)
    p = predictions["p_up"].to_numpy(float)
    confidence = confidence_values(p)
    order = np.argsort(-confidence, kind="stable")
    rows = []
    for level in levels:
        keep_n = max(round(level * len(y)), 1)
        idx = order[:keep_n]
        pred = (p[idx] >= 0.5).astype(int)
        rows.append({
            "coverage_target": float(level),
            "n": int(keep_n),
            "coverage_retained": float(keep_n / len(y)),
            "accuracy": float(accuracy_score(y[idx], pred)),
            "balanced_accuracy": (float(balanced_accuracy_score(y[idx], pred))
                                  if len(np.unique(y[idx])) == 2 else float("nan")),
            "f1": float(f1_score(y[idx], pred, zero_division=0)),
            "mean_confidence": float(confidence[idx].mean()),
        })
    return pd.DataFrame(rows)


def max_meaningful_coverage(predictions: pd.DataFrame, threshold: float, *,
                            levels=COVERAGE_LEVELS,
                            min_coverage: float = MIN_MEANINGFUL_COVERAGE,
                            min_observations: int = MIN_MEANINGFUL_OBSERVATIONS
                            ) -> dict:
    """Highest coverage whose accuracy reaches ``threshold`` AND is meaningful.

    A selective result counts only with coverage >= 20% AND n >= 500; otherwise the
    answer is ``SELECTIVE_65_NOT_ESTABLISHED`` rather than a number a reader could
    over-interpret.
    """
    curve = selective_table(predictions, levels=levels)
    if curve.empty:
        return {"threshold": float(threshold), "coverage": None, "n": None,
                "status": "SELECTIVE_NOT_ESTABLISHED"}
    eligible = curve.loc[(curve["accuracy"] >= threshold)
                         & (curve["coverage_retained"] >= min_coverage)
                         & (curve["n"] >= min_observations)]
    if eligible.empty:
        best = curve.loc[curve["accuracy"].idxmax()]
        return {
            "threshold": float(threshold),
            "coverage": None,
            "accuracy": None,
            "n": None,
            "max_accuracy_seen": float(best["accuracy"]),
            "max_coverage_seen": float(best["coverage_retained"]),
            "status": "SELECTIVE_65_NOT_ESTABLISHED"
            if threshold >= 0.65 else "SELECTIVE_TARGET_NOT_ESTABLISHED",
            "rule": f"coverage >= {min_coverage:.0%} AND n >= {min_observations}",
        }
    best = eligible.sort_values(["coverage_retained", "accuracy"],
                                ascending=[False, False]).iloc[0]
    return {
        "threshold": float(threshold),
        "coverage": float(best["coverage_retained"]),
        "accuracy": float(best["accuracy"]),
        "n": int(best["n"]),
        "status": "ESTABLISHED",
        "rule": f"coverage >= {min_coverage:.0%} AND n >= {min_observations}",
    }


# ---------------------------------------------------------------------------
# aggregation across development years
# ---------------------------------------------------------------------------

def aggregate_years(per_fold: list[dict]) -> dict:
    """Aggregate one horizon/model across the development years.

    Reports the mean of every headline metric, the standard deviation across
    years, the WORST year, and the two robustness counts the screening gate needs.
    """
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

    accuracy = _collect("accuracy")
    macro = _collect("macro_ticker_accuracy")
    balanced = _collect("balanced_accuracy")
    auc = _collect("roc_auc")
    brier = _collect("brier")
    baseline = _collect("train_majority_baseline")
    delta = _collect("baseline_delta")
    breadth = _collect("ticker_fraction_beating_own_baseline")

    def _mean(values: np.ndarray) -> float:
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    def _std(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(finite.std(ddof=1)) if finite.size > 1 else float("nan")

    worst_index = int(np.nanargmin(auc)) if np.isfinite(auc).any() else 0
    return {
        "n_folds": len(rows),
        "folds": [str(r.get("fold")) for r in rows],
        "mean_accuracy": _mean(accuracy),
        "std_accuracy": _std(accuracy),
        "mean_macro_accuracy": _mean(macro),
        "std_macro_accuracy": _std(macro),
        "mean_balanced_accuracy": _mean(balanced),
        "std_balanced_accuracy": _std(balanced),
        "mean_roc_auc": _mean(auc),
        "std_roc_auc": _std(auc),
        "mean_brier": _mean(brier),
        "std_brier": _std(brier),
        "mean_train_majority_baseline": _mean(baseline),
        "mean_baseline_delta": _mean(delta),
        "std_baseline_delta": _std(delta),
        "worst_year_roc_auc": float(auc[worst_index]) if np.isfinite(auc).any() else float("nan"),
        "worst_year": str(rows[worst_index].get("fold")) if rows else None,
        "years_roc_auc_above_50": int(np.nansum(auc > 0.50)),
        "years_beating_majority_baseline": int(np.nansum(delta > 0.0)),
        "mean_ticker_fraction_beating_own_baseline": _mean(breadth),
        "min_year_roc_auc": float(np.nanmin(auc)) if np.isfinite(auc).any() else float("nan"),
        "max_year_roc_auc": float(np.nanmax(auc)) if np.isfinite(auc).any() else float("nan"),
    }


def cross_year_block_bootstrap(predictions: pd.DataFrame, *, horizon: int,
                               n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
                               block_length: int = DEFAULT_BLOCK_LENGTH,
                               seed: int = BOOTSTRAP_SEED) -> dict:
    """Block-bootstrap CI on the POOLED validation predictions of one horizon."""
    return block_bootstrap_ci(predictions, block_length=block_length,
                              n_bootstrap=n_bootstrap, seed=seed,
                              where=f"{horizon}-trading-day pooled")