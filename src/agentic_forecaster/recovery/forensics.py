"""Pre-2022 forensic diagnostics for the paper reconstruction.

Why this module exists
----------------------
The faithful reconstruction trains at the author-confirmed schedule
(10 epochs, patience 10) and lands *below* the majority-class baseline on
pre-2022 validation. Conventional hyperparameter search is therefore the wrong
next move: it would tune noise. The open question is *structural* -- which part
of the lost implementation explains a high reported accuracy?

The diagnostics are grouped by what they can explain:

===========================  ===========================================
Hypothesis                  Diagnostic
===========================  ===========================================
DATA_ADJUSTMENT             adjusted vs unadjusted
MODEL_FORM                  simple models vs Attention-LSTM
PER_STOCK_VS_POOLED         one model per stock vs one pooled model
SPLIT_PROTOCOL              official walk-forward vs 85/15 chronological
TARGET_ALIGNMENT            next-day target proven from raw OHLCV
AGGREGATION                 micro / macro-ticker / macro-date / fold
CONFIDENCE_FILTERING        accuracy at a confidence floor
LEAKAGE / OFF_BY_ONE        deliberately invalid probes (L1-L4)
===========================  ===========================================

TWO HARD RULES, enforced here rather than by convention
-----------------------------------------------------
1. `assert_targets_pre_test` is called before *every* metric computation in this
   module, and raises `TestSetFirewallError` on a protected date. A protected
   date aborts the run rather than being silently excluded.
2. Nothing here imports `PAPER_REFERENCE` or compares against a published
   metric. These diagnostics are explanatory, not number-matching; optimising
   toward 0.815 would recreate exactly the bias the firewall exists to prevent.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .firewall import TestSetFirewallError, assert_pre_test_dates

#: The firewall boundary. No legitimate or deliberately invalid diagnostic in
#: this module may score a target date on or after this day.
PRE_TEST_CUTOFF = "2022-01-01"

#: Confidence floors probed by the confidence-subset diagnostic. Conditional
#: accuracy at a floor is NOT overall accuracy and is never reported as such.
CONFIDENCE_THRESHOLDS = (0.55, 0.60, 0.70, 0.80)

#: Marks every deliberately invalid probe so it can never be mistaken for a
#: result. Written into each probe's own artifacts as well as the summary.
INVALID_MARKER = {
    "VALID_FOR_FINAL_MODEL": False,
    "SCIENTIFICALLY_INVALID": True,
}

#: Earliest raw history to consider for the long-history analog. The dataset
#: itself begins in 2000; this bound only guards against a malformed source.
LONG_HISTORY_EARLIEST = "1990-01-01"

#: Marks forensic variants that are diagnostics, not author-confirmed designs.
POOLED_LABEL = "FORENSIC_POOLED_NOT_AUTHOR_CONFIRMED"
ANALOG_85_15_LABEL = "FORENSIC_PRE2022_LONG_HISTORY_85_15"


class ForensicFirewallAbort(RuntimeError):
    """Raised to ABORT the entire forensic run on a protected date.

    Distinct from `TestSetFirewallError` so callers can distinguish "this one
    diagnostic is wrong" from "stop everything now".
    """


# ---------------------------------------------------------------------------
# firewall
# ---------------------------------------------------------------------------

def assert_targets_pre_test(train_target_dates: Iterable,
                            val_target_dates: Iterable,
                            *, where: str = "forensic") -> None:
    """Assert BOTH train and validation target dates are pre-2022.

    Raises
    ------
    ForensicFirewallAbort
        If any target date is on/after the cutoff. The caller is expected to
        let this propagate, which aborts the whole run.
    """
    try:
        assert_pre_test_dates(train_target_dates, where=f"{where}/train target",
                              search=True)
        assert_pre_test_dates(val_target_dates, where=f"{where}/val target",
                              search=True)
    except TestSetFirewallError as exc:
        raise ForensicFirewallAbort(str(exc)) from exc


def max_target_date(*date_sets: Iterable) -> str | None:
    """Latest date across several date collections, as an ISO string."""
    best: pd.Timestamp | None = None
    for dates in date_sets:
        for d in dates:
            ts = pd.Timestamp(d)
            if pd.isna(ts):
                continue
            if best is None or ts > best:
                best = ts
    return None if best is None else best.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# aggregation conventions
# ---------------------------------------------------------------------------

def _acc(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> float:
    if len(y) == 0:
        return float("nan")
    return float(((p >= thr).astype(int) == y).mean())


def _f1(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> float:
    if len(y) == 0:
        return float("nan")
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return float(2 * prec * rec / (prec + rec))


def aggregation_metrics(predictions: pd.DataFrame) -> dict:
    """Every aggregation convention, reported side by side and unranked.

    Columns expected: ``ticker``, ``fold``, ``y_true``, ``p_up`` and optionally
    ``target_date``. The question is whether the *choice* of convention could
    plausibly explain an inflated headline number, so all of them are reported
    and the largest is never singled out.
    """
    assert_targets_pre_test(
        predictions.get("target_date", []), predictions.get("target_date", []),
        where="aggregation")
    out: dict = {}
    y = predictions["y_true"].to_numpy(int)
    p = predictions["p_up"].to_numpy(float)

    # micro: every observation counts once (the largest-n convention)
    out["micro"] = {"accuracy": _acc(y, p), "f1": _f1(y, p), "n": len(y)}

    # macro over tickers
    per_t = predictions.groupby("ticker").apply(
        lambda g: _acc(g["y_true"].to_numpy(int), g["p_up"].to_numpy(float)),
        include_groups=False)
    out["macro_ticker"] = {"accuracy": float(per_t.mean()), "n_tickers": len(per_t)}

    # macro over dates (a date shared by many tickers is one observation)
    if "target_date" in predictions.columns:
        per_d = predictions.groupby("target_date").apply(
            lambda g: _acc(g["y_true"].to_numpy(int), g["p_up"].to_numpy(float)),
            include_groups=False)
        out["macro_date"] = {"accuracy": float(per_d.mean()), "n_dates": len(per_d)}

    # fold conventions
    per_fold = predictions.groupby("fold").apply(
        lambda g: {"accuracy": _acc(g["y_true"].to_numpy(int), g["p_up"].to_numpy(float)),
                   "f1": _f1(g["y_true"].to_numpy(int), g["p_up"].to_numpy(float)),
                   "n": len(g)},
        include_groups=False)
    folds = list(per_fold.index)
    out["equal_fold_average"] = {
        "accuracy": float(np.mean([per_fold[f]["accuracy"] for f in folds])),
        "f1": float(np.mean([per_fold[f]["f1"] for f in folds])),
        "n_folds": len(folds),
    }
    sizes = np.array([per_fold[f]["n"] for f in folds], dtype=float)
    accs = np.array([per_fold[f]["accuracy"] for f in folds], dtype=float)
    f1s = np.array([per_fold[f]["f1"] for f in folds], dtype=float)
    w = sizes / sizes.sum()
    out["sample_weighted_fold_average"] = {
        "accuracy": float((accs * w).sum()),
        "f1": float((f1s * w).sum()),
        "n_folds": len(folds),
    }
    out["note"] = ("All conventions are reported so a spread between them is "
                   "visible. The largest value is NOT selected: a convention "
                   "that flatters the result is an aggregation artefact, not "
                   "evidence of signal.")
    return out


# ---------------------------------------------------------------------------
# confidence filtering
# ---------------------------------------------------------------------------

def confidence_subset_metrics(predictions: pd.DataFrame,
                              thresholds: Sequence[float] = CONFIDENCE_THRESHOLDS
                              ) -> dict:
    """Accuracy/F1 conditional on the model being confident.

    ``confidence = max(p_up, 1 - p_up)``. Accuracy at a floor measures only the
    confident subset and is explicitly NOT overall accuracy: reporting it as
    such would overstate performance whenever the model is merely confident.
    """
    assert_targets_pre_test(
        predictions.get("target_date", []), predictions.get("target_date", []),
        where="confidence")
    y = predictions["y_true"].to_numpy(int)
    p = predictions["p_up"].to_numpy(float)
    conf = np.maximum(p, 1.0 - p)
    out: dict = {
        "all_predictions": {
            "threshold": 0.0, "n": len(y),
            "retained_pct": 100.0, "accuracy": _acc(y, p), "f1": _f1(y, p),
        }
    }
    for t in thresholds:
        m = conf >= t
        n = int(m.sum())
        out[f">={t:.2f}"] = {
            "threshold": float(t), "n": n,
            "retained_pct": float(n / len(y) * 100) if len(y) else 0.0,
            "accuracy": _acc(y[m], p[m]) if n else None,
            "f1": _f1(y[m], p[m]) if n else None,
        }
    out["note"] = ("These are CONDITIONAL accuracies on the retained subset. "
                   "They are not overall accuracy and must never be quoted as "
                   "the model's accuracy. A high value with low retention is a "
                   "small-sample artefact, not signal.")
    return out


# ---------------------------------------------------------------------------
# label balance
# ---------------------------------------------------------------------------

def train_majority_baseline(train_y: np.ndarray, val_y: np.ndarray) -> float:
    """LEGITIMATE deployable baseline: always predict the TRAIN majority class.

    The class is chosen from TRAIN labels only, so no validation label is used
    to build the predictor. This is the only quantity a model's accuracy should
    be compared against.
    """
    tr = np.asarray(train_y, dtype=int)
    va = np.asarray(val_y, dtype=int)
    if not len(tr) or not len(va):
        return float("nan")
    cls = 1 if float(np.mean(tr)) >= 0.5 else 0
    # accuracy ON VALIDATION of the constant predictor chosen from TRAIN
    return float(np.mean(va == cls))


def validation_oracle_majority_rate(val_y: np.ndarray) -> float:
    """CLASS-BALANCE DESCRIPTION ONLY -- not a deployable predictor.

    This is ``max(validation positive rate, 1 - positive rate)``: it peeks at
    validation labels, so no trained model can legitimately be compared against
    it. Reporting it as a "baseline" would overstate a model.
    """
    va = np.asarray(val_y, dtype=int)
    if not len(va):
        return float("nan")
    pos = float(np.mean(va))
    return float(max(pos, 1 - pos))


def label_balance(train_y: np.ndarray, val_y: np.ndarray) -> dict:
    """Class-balance description, with the two 'majority' notions kept apart."""
    def _one(a: np.ndarray) -> dict:
        a = np.asarray(a, dtype=int)
        n = len(a)
        pos = float(np.mean(a)) if n else float("nan")
        return {"n": n, "positive_rate": pos,
                "majority_class": "UP" if pos >= 0.5 else "DOWN"}
    return {
        "train": _one(train_y),
        "validation": _one(val_y),
        # legitimate, deployable, TRAIN-derived
        "train_majority_baseline_accuracy": train_majority_baseline(
            train_y, val_y),
        # class-balance description only; never a baseline to beat
        "validation_oracle_majority_rate": validation_oracle_majority_rate(val_y),
        "terminology_note": (
            "train_majority_baseline_accuracy is the only legitimate "
            "comparison. validation_oracle_majority_rate uses validation "
            "labels and is a class-balance statistic, not a predictor."),
    }


# ---------------------------------------------------------------------------
# target alignment audit
# ---------------------------------------------------------------------------

@dataclass
class AlignmentRow:
    ticker: str
    fold: str
    sample_index: int
    sequence_final_date: str
    target_date: str
    close_at_sequence_final: float
    close_at_target: float
    expected_label: int
    actual_label: int
    label_matches: bool
    target_is_next_trading_day: bool
    sequence_contains_target_row: bool
    sequence_excludes_target: bool
    ok: bool


def validate_target_alignment(dataset, raw_frame: pd.DataFrame, *,
                              fold: str, ticker: str,
                              n_samples: int = 100,
                              probe_sequence_includes_target: bool = False,
                              ) -> tuple[list[AlignmentRow], int]:
    """Prove next-day target construction directly from raw OHLCV.

    For each sampled example this independently re-derives, from the raw
    untransformed price frame:

    * the sequence's final row date ``t``
    * the next ACTUAL trading date after ``t`` (so weekends/holidays resolve to
      the next real bar, not a calendar day)
    * ``y == int(Close[target_date] > Close[t])``
    * that the sequence does NOT contain the target row

    ``probe_sequence_includes_target=True`` deliberately builds the invalid L1
    alignment (target row appended to the window) and is used by tests to prove
    the validator actually detects the off-by-one.
    """
    frame = raw_frame.copy()
    # Yahoo CSVs use `Date`; be tolerant of either spelling.
    date_col = "date" if "date" in frame.columns else "Date"
    close_col = "Close" if "Close" in frame.columns else "close"
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values(date_col).reset_index(drop=True)
    closes = frame[close_col].to_numpy(dtype=float)
    dates = frame[date_col].to_numpy()

    train = dataset.train
    # sample deterministically from TRAIN only: alignment is a TRAIN-side
    # property, and validation rows would be scored for no benefit.
    n = min(n_samples, len(train.dates))
    idx = np.linspace(0, len(train.dates) - 1, n).astype(int) if n else np.array([], dtype=int)

    rows: list[AlignmentRow] = []
    mismatches = 0
    for k, i in enumerate(idx):
        # coerce: numpy string scalars are rejected by pd.Timestamp
        origin = pd.Timestamp(str(train.dates[i]))
        # find the origin row and the next actual trading row
        pos = int(np.searchsorted(dates, np.datetime64(origin), side="left"))
        if pos >= len(dates) - 1:
            continue
        nxt_pos = pos + 1
        tdate = pd.Timestamp(dates[nxt_pos])
        c_t, c_next = float(closes[pos]), float(closes[nxt_pos])
        expected = int(c_next > c_t)
        actual = int(train.y[i])

        # does the sequence window contain the target row's features?
        seq_start = pos - (dataset.train.X.shape[1] - 1)
        if probe_sequence_includes_target:
            contains_target = True
            excludes = False
        else:
            contains_target = bool(seq_start <= nxt_pos <= pos)
            excludes = not contains_target

        is_next = bool(tdate > origin)
        label_ok = bool(expected == actual)
        ok = bool(label_ok and is_next and excludes)
        if not ok:
            mismatches += 1
        rows.append(AlignmentRow(
            ticker=ticker, fold=fold, sample_index=k,
            sequence_final_date=origin.strftime("%Y-%m-%d"),
            target_date=tdate.strftime("%Y-%m-%d"),
            close_at_sequence_final=c_t, close_at_target=c_next,
            expected_label=expected, actual_label=actual,
            label_matches=label_ok, target_is_next_trading_day=is_next,
            sequence_contains_target_row=contains_target,
            sequence_excludes_target=excludes, ok=ok))
    return rows, mismatches


# ---------------------------------------------------------------------------
# 85/15 chronological analog
# ---------------------------------------------------------------------------

def pre2022_85_15_split(dates: Sequence, *,
                        cutoff: str = PRE_TEST_CUTOFF) -> tuple[list, list, str]:
    """First chronological 85% -> train, last 15% -> validation, pre-2022 only.

    Source data is truncated at the cutoff FIRST, so the split cannot reach a
    protected year even if the caller passes a longer frame. No shuffle.
    """
    ts = pd.to_datetime(pd.Series(list(dates)))
    keep = ts[ts < pd.Timestamp(cutoff)].drop_duplicates().sort_values()
    if keep.empty:
        return [], [], ""
    n = len(keep)
    cut = max(1, min(n - 1, round(n * 0.85)))
    tr = [d.strftime("%Y-%m-%d") for d in keep.iloc[:cut]]
    va = [d.strftime("%Y-%m-%d") for d in keep.iloc[cut:]]
    return tr, va, (va[-1] if va else "")


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def write_forensic_csv(rows: Sequence[dict], path: Path, *,
                       marker: dict | None = None) -> Path:
    """Write forensic rows, prefixing an explicit invalidity marker if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    if marker:
        for k, v in marker.items():
            df[k] = v
    df.to_csv(path, index=False)
    return path


def write_forensic_json(payload: dict, path: Path, *,
                        marker: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    if marker:
        body.update(marker)
    path.write_text(json.dumps(body, indent=2, default=str))
    return path


def forensic_dir(sub: str = "") -> Path:
    """Runtime forensics root, outside every immutable dataset."""
    from .ledger import runtime_dir
    root = runtime_dir() / "forensics"
    if sub:
        root = root / sub
    root.mkdir(parents=True, exist_ok=True)
    return root


def repo_forensic_dir(sub: str = "") -> Path:
    from .ledger import repo_root
    root = repo_root() / "results" / "reproduction_recovery" / "forensics"
    if sub:
        root = root / sub
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# L1 probe: a GENUINE target-day-in-window off-by-one
# ---------------------------------------------------------------------------

class L1AlignmentError(AssertionError):
    """Raised when the L1 leaked sequence is not actually target-aligned."""


def build_target_day_leak(X, y, origin_dates, target_dates, *,
                          date_values=None) -> dict:
    """Build the DELIBERATELY INVALID "target day is in the window" sequences.

    Why the naive version is wrong
    -----------------------------
    Shifting an existing window left and leaving the final timestep alone does
    NOT leak: the final timestep is still the origin row, so the target day's
    information is absent and the probe measures nothing. The off-by-one being
    hypothesised requires the final timestep to BE the target day.

    This builds a genuine mapping ``date -> feature vector at that date`` from
    the samples themselves (each sample's final timestep is the feature vector
    at its own origin date), then for every sample appends the vector belonging
    to its **target date** after shifting the rest left:

        [f(t-L+1) ... f(t-1), f(t), f(t+1)]      <- final element is f(t+1)

    while leaving the label as the true next-day direction
    ``int(Close[t+1] > Close[t])``.

    Returns aligned ``X_leaked``, ``y``, origin dates, target dates and the
    final-timestep date actually realised for each kept sample. Every kept
    sample is asserted: the final timestep must be ``target_date`` and
    ``target_date > origin_date``.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    origins = [str(pd.Timestamp(d).date()) for d in origin_dates]
    targets = [str(pd.Timestamp(d).date()) for d in target_dates]

    # date -> feature vector at that date, taken from each sample's last row
    vec_by_date: dict[str, np.ndarray] = {}
    for i, d in enumerate(origins):
        vec_by_date[d] = X[i, -1, :]

    seq_len = X.shape[1]
    keep, leaked, last_dates = [], [], []
    for i, (o, t) in enumerate(zip(origins, targets)):
        if t <= o:
            raise L1AlignmentError(f"{o}: target {t} is not after origin")
        if t not in vec_by_date:
            continue          # genuinely unavailable -> drop, do not invent
        out = np.empty_like(X[i])
        out[:-1, :] = X[i, 1:, :]     # shift left
        out[-1, :] = vec_by_date[t]    # append the TRUE target-day vector
        keep.append(i)
        leaked.append(out)
        last_dates.append(t)

    if not keep:
        raise L1AlignmentError("no sample had an available target-day vector")

    X_leaked = np.stack(leaked).astype(np.float32)
    assert X_leaked.shape[1] == seq_len

    # Per-sample assertions, as required: the final element must genuinely be
    # the target date, and the target must be strictly after the origin.
    for j, i in enumerate(keep):
        # `i` indexes the ORIGINAL sample list, `j` indexes the KEPT list. Any
        # sample dropped above makes the two diverge, so the target must come
        # from `targets[i]` (the returned target_dates already do this).
        o, t = origins[i], targets[i]
        if not (last_dates[j] == t):
            raise L1AlignmentError(
                f"final timestep {last_dates[j]} != target_date {t}")
        if not (t > o):
            raise L1AlignmentError(f"target {t} is not after origin {o}")
        # the appended vector must be the one registered for the target date
        if not np.allclose(X_leaked[j, -1, :], vec_by_date[t]):
            raise L1AlignmentError(
                f"{o}: appended final timestep is not the target-day vector")

    return {
        "X_leaked": X_leaked,
        "y": y[keep],
        "origin_dates": [origins[i] for i in keep],
        "target_dates": [targets[i] for i in keep],
        "final_timestep_dates": last_dates,
        "n_kept": len(keep),
        "n_dropped": len(origins) - len(keep),
        "date_values": date_values,
    }
