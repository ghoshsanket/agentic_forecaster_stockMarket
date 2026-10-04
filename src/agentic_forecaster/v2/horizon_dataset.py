"""Multi-horizon sample construction: one supervised sample per (security, origin)
with an ``H``-trading-day label.

WHAT IS DIFFERENT FROM THE ONE-DAY V2 SAMPLE TABLE
--------------------------------------------------
Only the LABEL and the split boundary:

``target_date``  is the H-th FUTURE TRADING OBSERVATION (``target_end_date``);
``y_direction``  is ``1 if log(Close[t+H]/Close[t]) > 0 else 0``.

The INPUT is byte-for-byte the existing V2 stationary stock schema and the same
``sequence_length``, and the input window still ends at the origin row.  For a
10-trading-day target the model therefore still sees only ``<= t``; the
intermediate closes exist solely while the label is derived, and they are never
features, never context and never scaler inputs.

SPLIT BOUNDARY SAFETY (the multi-day hazard)
---------------------------------------------
A sample belongs to a split only when BOTH dates are inside it::

    train      : origin in [train_start, train_end]
                 AND target_end_date in [train_start, train_end]
    validation : origin in [val_start, val_end]
                 AND target_end_date in [val_start, val_end]

So an origin of 2018-12-24 whose 10-trading-day target ends in January 2019 is
NOT part of 2018 validation, and a December training origin whose target enters
the next validation year is dropped from training.  With the one-day horizon the
rule is unchanged, which is exactly why the control stays comparable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import horizons as HZ
from .dataset import (
    FeatureArrays,
    SampleTable,
    SplitWindow,
    build_feature_arrays,
    split_masks,
)
from .firewall import assert_pre_covid_dates

logger = logging.getLogger("agentic_forecaster.v2.horizon_dataset")

#: Columns of the multi-horizon sample frame.  ``target_date`` is the H-th future
#: trading observation; ``y_direction`` is ``y_H``; ``future_log_return`` is kept
#: for the magnitude DIAGNOSTIC only and is never a model input.
SAMPLE_COLUMNS: tuple[str, ...] = (
    "ticker",
    "origin_date",
    "target_date",
    "row",
    "ticker_id",
    "sector_id",
    "y_direction",
    "y_return",
    "y_rank",
    "horizon",
    "future_log_return",
)


def build_horizon_sample_table(arrays: FeatureArrays, targets: pd.DataFrame, *,
                               horizon: int, sequence_length: int,
                               final_allowed_date: str = HZ.FINAL_ALLOWED_DATE,
                               locked: bool = False,
                               where: str = "horizon sample table") -> SampleTable:
    """Enumerate every usable supervised sample for ONE horizon.

    A sample exists only when the security has a full window of finite inputs at
    the origin, the regime vector is finite, and the H-trading-day target exists.

    ``locked`` unlocks the 2019 lockbox year for the ONE legitimate lockbox run.
    Without it the lockbox year is EXCLUDED here, so a development sample table
    physically cannot contain a 2019 label; the count of excluded rows is recorded
    in the diagnostics.  A development SPLIT that tried to consume a 2019 target
    would still raise
    :class:`~agentic_forecaster.v2.horizons.MultiHorizonLockboxError` in
    :func:`horizon_split`.
    """
    horizon = int(horizon)
    objective = HZ.objective_id(horizon)
    block = targets.loc[targets["horizon"] == horizon].copy()
    if block.empty:
        raise ValueError(f"{objective}: no target rows for horizon {horizon}")
    block["origin_date"] = pd.to_datetime(block["origin_date"])
    block["target_end_date"] = pd.to_datetime(block["target_end_date"])

    # Boundary 1: the absolute cap.  A target that would end in 2020 is REJECTED
    # loudly, which is stronger than dropping it silently.
    assert_pre_covid_dates(origin_dates=block["origin_date"],
                           target_dates=block["target_end_date"],
                           final_allowed_date=final_allowed_date,
                           where=f"{where}/{objective}")
    # Boundary 2: the lockbox year.  The target cache legitimately holds 2019 rows
    # (the data regime ends 2019-12-31), but a DEVELOPMENT sample table must not be
    # able to read them, so they are excluded here and counted.
    in_lockbox_year = (block["target_end_date"] >= HZ.LOCKBOX_YEAR_START) & (
        block["target_end_date"] <= HZ.LOCKBOX_YEAR_END)
    dropped_lockbox_rows = int(in_lockbox_year.sum()) if not locked else 0
    if not locked:
        block = block.loc[~in_lockbox_year].copy()
    HZ.assert_no_lockbox_year_targets(block["target_end_date"],
                                      where=f"{where}/{objective}", unlocked=locked)

    block = block.loc[block["ticker"].isin(arrays.matrices)].copy()
    if block.empty:
        raise ValueError(f"{objective}: no target row belongs to the feature universe")

    records: list[dict] = []
    dropped = {"no_full_window": 0, "non_finite_inputs": 0,
               "no_future_observation": 0,
               "lockbox_year_excluded": dropped_lockbox_rows}
    for ticker, group in block.groupby("ticker", sort=True):
        matrices = arrays.matrices[str(ticker)]
        positions = matrices.position_of
        n_context = matrices.context.shape[1]
        group = group.sort_values("origin_date")
        for record in group.itertuples(index=False):
            row = positions.get(str(pd.Timestamp(record.origin_date).date()))
            if row is None or row < sequence_length - 1:
                dropped["no_full_window"] += 1
                continue
            if not _window_is_usable(matrices, row, sequence_length, n_context):
                dropped["non_finite_inputs"] += 1
                continue
            records.append({
                "ticker": str(ticker),
                "origin_date": pd.Timestamp(record.origin_date),
                "target_date": pd.Timestamp(record.target_end_date),
                "row": int(row),
                "ticker_id": arrays.ticker_id(str(ticker)),
                "sector_id": arrays.sector_id(str(ticker)),
                "y_direction": int(record.y),
                # the multi-horizon programme is a DIRECTION-only study: the
                # auxiliary heads are switched off, so their placeholders are
                # finite zeros rather than NaN, which would poison the loss
                "y_return": 0.0,
                "y_rank": 0.0,
                "horizon": horizon,
                "future_log_return": float(record.future_log_return),
            })

    frame = pd.DataFrame(records, columns=list(SAMPLE_COLUMNS))
    frame = frame.sort_values(["ticker", "origin_date"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{objective}: no usable supervised sample survived the "
                         "causal input-window requirement")
    diagnostics = {
        "objective_id": objective,
        "horizon": horizon,
        "horizon_phrase": HZ.horizon_phrase(horizon),
        "n_samples": len(frame),
        "n_tickers": int(frame["ticker"].nunique()),
        "sequence_length": int(sequence_length),
        "dropped": dropped,
        "first_origin_date": str(frame["origin_date"].min().date()),
        "last_origin_date": str(frame["origin_date"].max().date()),
        "last_target_date": str(frame["target_date"].max().date()),
        "max_origin_date_consumed": str(frame["origin_date"].max().date()),
        "max_target_date_consumed": str(frame["target_date"].max().date()),
        "direction_positive_rate": float(frame["y_direction"].mean()),
        "final_allowed_date": final_allowed_date,
        "lockbox_unlocked": bool(locked),
        "lockbox_year_excluded_rows": dropped_lockbox_rows,
        "max_lockbox_year_target_in_table": (
            None if locked or block.empty
            else str(pd.Timestamp(block["target_end_date"].max()).date())),
    }
    return SampleTable(frame=frame, arrays=arrays, sequence_length=int(sequence_length),
                       diagnostics=diagnostics)


def _window_is_usable(matrices, row: int, sequence_length: int, n_context: int) -> bool:
    """True when every INPUT the sample needs exists for the whole window.

    A sample is dropped, never repaired: no forward fill, no interpolation, no
    shortened window.  This is what keeps a late listing from being backfilled and
    what guarantees no future row can enter an input sequence.
    """
    start = row - sequence_length + 1
    if start < 0:
        return False
    stop = row + 1
    if not np.isfinite(matrices.stock[start:stop]).all():
        return False
    if n_context and not np.isfinite(matrices.context[start:stop]).all():
        return False
    return bool(np.isfinite(matrices.regime[row]).all())


def build_all_horizon_samples(arrays: FeatureArrays, targets: pd.DataFrame, *,
                              horizons=HZ.HORIZONS, sequence_length: int,
                              final_allowed_date: str = HZ.FINAL_ALLOWED_DATE,
                              locked: bool = False) -> dict[int, SampleTable]:
    """One sample table per horizon, sharing the same feature arrays."""
    return {
        int(horizon): build_horizon_sample_table(
            arrays, targets, horizon=int(horizon), sequence_length=sequence_length,
            final_allowed_date=final_allowed_date, locked=locked)
        for horizon in horizons
    }


@dataclass(frozen=True)
class HorizonSplit:
    """Train / validation masks for ONE horizon and ONE fold."""

    horizon: int
    fold: str
    window: SplitWindow
    train: np.ndarray
    val: np.ndarray

    @property
    def objective_id(self) -> str:
        return HZ.objective_id(self.horizon)

    def counts(self, samples: SampleTable) -> dict:
        return {
            "n_train": int(self.train.sum()),
            "n_validation": int(self.val.sum()),
            "n_train_tickers": int(samples.frame.loc[self.train, "ticker"].nunique()),
            "n_validation_tickers": int(samples.frame.loc[self.val, "ticker"].nunique()),
        }


def horizon_split(samples: SampleTable, window: SplitWindow, *, fold: str = "",
                  locked: bool = False, where: str = "horizon split"
                  ) -> HorizonSplit:
    """Origin AND target-end masks for one horizon, with the boundaries proved.

    ``split_masks`` already implements the "both dates inside the window" rule;
    this wrapper adds the horizon-specific boundary audit so a fold cannot quietly
    contain a target that crosses a year or a fold edge.
    """
    # A development fold may not even DECLARE the lockbox year.  Checking the
    # declared window (not only the rows it happens to contain) is what makes
    # "screening cannot read 2019" a structural property rather than a consequence
    # of which rows survived.
    if not locked:
        HZ.assert_no_lockbox_year_targets(
            [window.val_start, window.val_end], unlocked=False,
            where=f"{where}/{fold or 'fold'}/declared validation window")
    masks = split_masks(samples, window)
    horizon = int(samples.frame["horizon"].iloc[0]) if len(samples.frame) else 0
    for name, mask in masks.items():
        block = samples.frame.loc[mask]
        if block.empty:
            continue
        HZ.assert_horizon_boundary(block["target_date"],
                                   origin_dates=block["origin_date"],
                                   where=f"{where}/{fold or 'fold'}/{HZ.objective_id(horizon)}"
                                   f"/{name}",
                                   unlocked=locked or name == "train")
    return HorizonSplit(horizon=horizon, fold=fold or window.name, window=window,
                        train=masks["train"], val=masks["val"])


def boundary_report(samples: SampleTable, window: SplitWindow, *, horizon: int,
                    where: str = "boundary") -> dict:
    """Evidence that both dates of every split sample are inside the window."""
    split = horizon_split(samples, window, fold=window.name, where=where)
    frame = samples.frame
    out = {"horizon": int(horizon), "objective_id": HZ.objective_id(horizon),
           "fold": window.name, "train_samples": 0, "validation_samples": 0}
    for name, mask in (("train", split.train), ("validation", split.val)):
        block = frame.loc[mask]
        if block.empty:
            continue
        origin = pd.to_datetime(block["origin_date"])
        target = pd.to_datetime(block["target_date"])
        start = pd.Timestamp(window.train_start if name == "train" else window.val_start)
        end = pd.Timestamp(window.train_end if name == "train" else window.val_end)
        assert bool((origin >= start).all() and (origin <= end).all())
        assert bool((target >= start).all() and (target <= end).all())
        out[f"{name}_samples"] = len(block)
        out[f"{name}_max_origin"] = str(origin.max().date())
        out[f"{name}_max_target_end"] = str(target.max().date())
    out["rule"] = "origin AND target_end_date must both lie inside the split window"
    return out


def origin_feature_matrix(samples: SampleTable, mask: np.ndarray, *,
                          columns=None) -> np.ndarray:
    """Stack the ORIGIN-row stationary features of the selected samples.

    This is the design matrix of the cheap screen: the input is exactly what the
    sequence models see at their last timestep, so the screen cannot benefit from
    information the neural models do not have.
    """
    frame = samples.frame.loc[mask]
    if frame.empty:
        return np.zeros((0, len(columns or samples.arrays.stock_features)),
                        dtype=np.float64)
    blocks = [samples.arrays.matrices[str(t)].stock[int(r), :]
              for t, r in zip(frame["ticker"], frame["row"], strict=True)]
    matrix = np.vstack(blocks).astype(np.float64)
    if columns is not None:
        keep = [samples.arrays.stock_features.index(c) for c in columns]
        matrix = matrix[:, keep]
    return matrix


def common_origin_keys(targets: pd.DataFrame, horizons=HZ.HORIZONS) -> pd.MultiIndex:
    """Origins whose target is available at EVERY tested horizon."""
    return HZ.horizon_target_keys(targets, horizons)


def common_origin_masks(samples_by_horizon: dict[int, SampleTable], window: SplitWindow,
                        *, horizons=HZ.HORIZONS, fold: str = "",
                        locked: bool = False) -> dict[int, dict[str, np.ndarray]]:
    """Per-horizon masks restricted to the COMMON-ORIGIN sample set.

    The comparison that answers "is one horizon genuinely easier, rather than
    merely evaluated on a different subset?": every horizon is scored on exactly
    the origins that are valid for all of them.
    """
    frames = {h: samples_by_horizon[h].frame.loc[:, ["ticker", "origin_date",
                                                      "target_date"]].copy()
             for h in horizons if h in samples_by_horizon}
    keys, train_keys, val_keys = _common_valid_keys(frames, window)
    out: dict[int, dict[str, np.ndarray]] = {}
    for horizon, samples in samples_by_horizon.items():
        frame = samples.frame
        split = horizon_split(samples, window, fold=fold, locked=locked,
                              where="common origin split")
        in_keys = HZ.common_origin_mask(frame, keys)
        out[horizon] = {"train": split.train & in_keys, "val": split.val & in_keys,
                        "n_train": int((split.train & in_keys).sum()),
                        "n_val": int((split.val & in_keys).sum()),
                        # the shared key sets, so a caller holding only a
                        # validation SUBSET can still restrict itself to the common
                        # origins of that split
                        "keys": keys,
                        "train_keys": keys[keys.isin(train_keys)],
                        "val_keys": keys[keys.isin(val_keys)]}
    return out


def _common_valid_keys(frames: dict[int, pd.DataFrame], window: SplitWindow
                       ) -> tuple[pd.MultiIndex, pd.MultiIndex, pd.MultiIndex]:
    """Origins inside the window whose target exists for EVERY horizon.

    Returns the shared key set plus its TRAIN-side and VALIDATION-side subsets,
    which is what makes the common-origin comparison apples-to-apples: the
    validation figure of one horizon is only comparable with the validation figure
    of another, never with a train+validation total.
    """
    empty = pd.MultiIndex.from_arrays([[], []], names=["ticker", "origin_date"])
    if not frames:
        return empty, empty, empty
    per_horizon: dict[str, list[pd.MultiIndex]] = {"train": [], "val": [], "all": []}
    for block in frames.values():
        frame = block.copy()
        frame["origin_date"] = pd.to_datetime(frame["origin_date"])
        frame["target_date"] = pd.to_datetime(frame["target_date"])
        in_train = ((frame["origin_date"] >= pd.Timestamp(window.train_start))
                    & (frame["origin_date"] <= pd.Timestamp(window.train_end))
                    & (frame["target_date"] >= pd.Timestamp(window.train_start))
                    & (frame["target_date"] <= pd.Timestamp(window.train_end)))
        in_val = ((frame["origin_date"] >= pd.Timestamp(window.val_start))
                  & (frame["origin_date"] <= pd.Timestamp(window.val_end))
                  & (frame["target_date"] >= pd.Timestamp(window.val_start))
                  & (frame["target_date"] <= pd.Timestamp(window.val_end)))
        for name, mask in (("train", in_train), ("val", in_val),
                           ("all", in_train | in_val)):
            per_horizon[name].append(pd.MultiIndex.from_frame(
                frame.loc[mask, ["ticker", "origin_date"]].drop_duplicates()))

    def _intersect(blocks: list[pd.MultiIndex]) -> pd.MultiIndex:
        common = blocks[0]
        for other in blocks[1:]:
            common = common.intersection(other)
        return common

    return (_intersect(per_horizon["all"]), _intersect(per_horizon["train"]),
            _intersect(per_horizon["val"]))


def build_horizon_arrays(store, sector_map, *, tickers, use_context: bool = False
                         ) -> FeatureArrays:
    """Stationary feature arrays for this track.

    ``use_context`` defaults to False: the multi-horizon experiment isolates the
    HORIZON, and the previous PRE-COVID run already found context added little or
    hurt.  No new feature is introduced anywhere.
    """
    from .experiment import assemble_context_frame

    return build_feature_arrays(store.stock, assemble_context_frame(store), sector_map,
                                tickers=sorted(tickers), use_context=use_context)