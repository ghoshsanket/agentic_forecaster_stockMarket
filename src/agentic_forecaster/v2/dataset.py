"""V2 sample construction, split assignment, global train-only scaling and the
torch ``Dataset``.

SAMPLE STRUCTURE
----------------
One supervised sample per (security, prediction origin ``t``)::

    stock_sequence     [T, F_stock]   stationary features of the security
    context_sequence   [T, F_context] market/sector/relative/rank context
    ticker_id, sector_id
    regime_vector      [F_regime]     origin-date market state
    y_direction, y_return, y_rank
    ticker, origin_date, target_date

``T`` defaults to 60.  The last timestep of every sequence is the ORIGIN date
``t``; the ``t+1`` row is never part of a sequence, and ``target_date`` is kept
explicitly on every sample.

SPLIT RULE
----------
A sample belongs to a split only when BOTH ``origin_date`` and ``target_date``
fall inside the split window.  The 60 rows of history that precede a January
window are allowed to come from December: a real forecaster standing on
31 December has that history available.  Scaler fitting, label fitting and loss
computation, however, use TRAIN samples only.

SCALING
-------
V2 is ONE SHARED model, so it gets ONE global feature scaler per feature group,
fitted on TRAIN samples only.  It is explicitly NOT one scaler per security.
Percentile rank features are already bounded in [0, 1] and are passed through
unscaled; every other continuous feature is standardised.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from . import context as ctx
from . import features as feat
from .firewall import (
    assert_no_lockbox_targets,
    assert_no_paper_test_targets,
    assert_pre_covid_dates,
)
from .sectors import UNKNOWN_LABEL, SectorMap, build_vocabulary, vocabulary_index

logger = logging.getLogger("agentic_forecaster.v2.dataset")

#: V2 sequence length. The paper's 30-day window is untouched; V2 deliberately
#: starts at 60 because the LSTM carries local dynamics while the Transformer
#: needs a wider history to be useful. Sequence length is NOT searched here.
DEFAULT_SEQUENCE_LENGTH = 60


@dataclass(frozen=True)
class SplitWindow:
    """A chronological split declared by date."""

    name: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str

    def contains(self, date: pd.Series | pd.Timestamp, start: str, end: str) -> pd.Series:
        ts = pd.to_datetime(date)
        return (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))


#: The three V2 folds.  Only ``V2_DEV_FOLD_A``/``V2_DEV_FOLD_B`` may be used for
#: architecture selection; ``V2_LOCKBOX`` requires the explicit lockbox switch.
V2_DEV_FOLD_A = SplitWindow("V2_DEV_FOLD_A", "2005-01-01", "2018-12-31",
                            "2019-01-01", "2019-12-31")
V2_DEV_FOLD_B = SplitWindow("V2_DEV_FOLD_B", "2005-01-01", "2019-12-31",
                            "2020-01-01", "2020-12-31")
V2_LOCKBOX = SplitWindow("V2_LOCKBOX", "2005-01-01", "2020-12-31",
                         "2021-01-01", "2021-12-31")

#: Backward-compatible defaults for the ORIGINAL V2 configs and tests.  New
#: tracks (for example the PRE-COVID regime) declare their own windows under
#: ``folds:`` in their config and are resolved by :func:`resolve_fold_window`,
#: so no new hard-coded global fold set is ever added here.
V2_FOLDS: dict[str, SplitWindow] = {
    V2_DEV_FOLD_A.name: V2_DEV_FOLD_A,
    V2_DEV_FOLD_B.name: V2_DEV_FOLD_B,
    V2_LOCKBOX.name: V2_LOCKBOX,
}


def resolve_fold_window(name: str, declared: dict | None = None) -> SplitWindow:
    """Resolve a split window for ``name``.

    A config-declared ``folds:`` block WINS over the built-in defaults, which is
    what makes the fold definitions genuinely config-driven: changing the YAML
    changes the resolved ``SplitWindow``.
    """
    key = str(name).upper()
    if declared:
        lowered = {str(k).lower(): v for k, v in declared.items()}
        block = (declared.get(key) or declared.get(name)
                 or lowered.get(key.lower()) or lowered.get(str(name).lower()))
        if block:
            missing = [f for f in ("train_start", "train_end", "val_start", "val_end")
                       if f not in block]
            if missing:
                raise ValueError(f"fold {key!r} is missing {missing}")
            return SplitWindow(
                name=key,
                train_start=str(block["train_start"]),
                train_end=str(block["train_end"]),
                val_start=str(block["val_start"]),
                val_end=str(block["val_end"]),
            )
    if key not in V2_FOLDS:
        raise ValueError(
            f"unknown fold {key!r}: declare it under `folds:` in the config or use one "
            f"of the built-in defaults {sorted(V2_FOLDS)}"
        )
    return V2_FOLDS[key]

#: Supervised development securities. Broad sector coverage, manageable compute.
V2_DEV_TICKERS: tuple[str, ...] = (
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC", "LT", "SUNPHARMA", "TATASTEEL",
)


# ---------------------------------------------------------------------------
# feature matrices
# ---------------------------------------------------------------------------

@dataclass
class TickerMatrices:
    """Per-security date-indexed feature matrices for fast window slicing."""

    ticker: str
    dates: np.ndarray                      # [n_dates] datetime64[ns]
    stock: np.ndarray                      # [n_dates, F_stock] float32
    context: np.ndarray                    # [n_dates, F_context] float32
    regime: np.ndarray                     # [n_dates, F_regime] float32
    position_of: dict[str, int] = field(default_factory=dict)

    @property
    def n_dates(self) -> int:
        return len(self.dates)


@dataclass
class FeatureArrays:
    """Every per-security matrix, plus the ID vocabularies."""

    matrices: dict[str, TickerMatrices]
    ticker_vocab: list[str]
    sector_vocab: list[str]
    sector_of: dict[str, str]
    stock_features: list[str]
    context_features: list[str]
    percentile_features: list[str]
    regime_features: list[str]

    @property
    def tickers(self) -> list[str]:
        return sorted(self.matrices)

    @property
    def n_stock_features(self) -> int:
        return len(self.stock_features)

    @property
    def n_context_features(self) -> int:
        return len(self.context_features)

    @property
    def n_regime_features(self) -> int:
        return len(self.regime_features)

    def ticker_id(self, ticker: str) -> int:
        return vocabulary_index(self.ticker_vocab, ticker)

    def sector_id(self, ticker: str) -> int:
        return vocabulary_index(self.sector_vocab, self.sector_of.get(ticker, UNKNOWN_LABEL))


def build_feature_arrays(stock_features: pd.DataFrame, context: pd.DataFrame,
                         sector_map: SectorMap, *, tickers: Sequence[str] | None = None,
                         use_context: bool = True) -> FeatureArrays:
    """Slice the cached store into per-security matrices with fixed vocabularies.

    ``context`` must carry the market, sector, relative, rank and regime columns
    for every (security, date) of the universe.  Only the securities in
    ``tickers`` receive supervised samples; the context columns still describe
    every available security, which is what makes the leave-one-out statistics
    and the cross-sectional ranks meaningful.
    """
    wanted = list(tickers) if tickers is not None else sorted(stock_features["ticker"].unique())
    context_columns = list(ctx.CONTEXT_FEATURES) + list(ctx.RANK_FEATURES) if use_context else []
    if context_columns:
        missing = [c for c in context_columns if c not in context.columns]
        if missing:
            raise KeyError(f"context frame is missing column(s) {missing}")

    stock = stock_features.loc[stock_features["ticker"].isin(wanted)]
    # Only the context columns are taken from ``context``: a full-frame merge
    # would collide with the identically named raw stock columns (log_return_1,
    # realized_vol_20, ...) and silently produce _x/_y suffixed columns.
    context_subset = context.loc[:, ["ticker", "date", *context_columns]]
    merged = stock.merge(context_subset, on=["ticker", "date"], how="left",
                         validate="one_to_one")
    sector_of = sector_map.sector_series()

    matrices: dict[str, TickerMatrices] = {}
    for ticker, group in merged.groupby("ticker", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        if group.empty:
            continue
        dates = pd.to_datetime(group["date"]).to_numpy(dtype="datetime64[ns]")
        stock_matrix = group.loc[:, list(feat.STOCK_FEATURE_NAMES)].to_numpy(dtype=np.float32)
        if context_columns:
            context_matrix = group.loc[:, context_columns].to_numpy(dtype=np.float32)
            regime_matrix = group.loc[:, list(ctx.REGIME_FEATURES)].to_numpy(dtype=np.float32)
        else:
            context_matrix = np.zeros((len(group), 0), dtype=np.float32)
            regime_matrix = np.zeros((len(group), 0), dtype=np.float32)
        matrices[str(ticker)] = TickerMatrices(
            ticker=str(ticker),
            dates=dates,
            stock=stock_matrix,
            context=context_matrix,
            regime=regime_matrix,
            position_of={str(pd.Timestamp(d).date()): i for i, d in enumerate(dates)},
        )

    ticker_vocab = build_vocabulary(matrices)
    sector_vocab = build_vocabulary({sector_of.get(t, UNKNOWN_LABEL) for t in matrices})
    return FeatureArrays(
        matrices=matrices,
        ticker_vocab=ticker_vocab,
        sector_vocab=sector_vocab,
        sector_of=sector_of,
        stock_features=list(feat.STOCK_FEATURE_NAMES),
        context_features=context_columns,
        percentile_features=list(ctx.RANK_FEATURES) if use_context else [],
        regime_features=list(ctx.REGIME_FEATURES) if use_context else [],
    )


# ---------------------------------------------------------------------------
# samples
# ---------------------------------------------------------------------------

@dataclass
class SampleTable:
    """The supervised sample index for one split (or for all of them)."""

    frame: pd.DataFrame
    arrays: FeatureArrays
    sequence_length: int
    diagnostics: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def tickers(self) -> list[str]:
        return sorted(self.frame["ticker"].unique())

    def subset(self, mask: np.ndarray) -> SampleTable:
        return SampleTable(
            frame=self.frame.loc[mask].reset_index(drop=True),
            arrays=self.arrays,
            sequence_length=self.sequence_length,
            diagnostics=dict(self.diagnostics),
        )


def _window_is_usable(matrices: TickerMatrices, row: int, sequence_length: int,
                      n_context: int) -> bool:
    """True when every input the sample needs exists for the whole window.

    A sample is dropped, never repaired: no forward fill, no interpolation, no
    shortened window.  This is what keeps a late listing from being backfilled.
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


def build_sample_table(arrays: FeatureArrays, targets: pd.DataFrame, *,
                       sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
                       require_targets: Sequence[str] = ("y_direction",),
                       max_date: str | None = None,
                       final_allowed_date: str | None = None) -> SampleTable:
    """Enumerate every usable supervised sample.

    ``targets`` is the store's target frame.  A sample is created only when the
    security has a full window of finite inputs at the origin, the regime vector
    is finite, and every requested target is present.

    ``final_allowed_date`` activates the PRE-COVID regime guard: a candidate
    sample whose origin OR target falls after that boundary is REJECTED with
    :class:`PostCovidDataAccessError` rather than silently dropped, so a
    2019-12-31 origin paired with a 2020-01-01 target cannot exist even by
    accident.  ``None`` keeps the ordinary V2 behaviour unchanged.
    """
    merged = targets.merge(
        pd.DataFrame({"ticker": list(arrays.matrices)}), on="ticker", how="inner")
    merged = merged.sort_values(["ticker", "origin_date"]).reset_index(drop=True)
    if final_allowed_date is not None:
        # Guard the WHOLE incoming frame BEFORE any filtering: in the PRE-COVID
        # regime a post-boundary row is an error to surface, not a row to hide
        # behind a filter.
        assert_pre_covid_dates(origin_dates=merged["origin_date"],
                               target_dates=merged["target_date"],
                               final_allowed_date=final_allowed_date,
                               where="precovid target frame")
    if max_date is not None:
        merged = merged.loc[merged["target_date"] <= pd.Timestamp(max_date)]

    records: list[dict] = []
    dropped = {
        "no_full_window": 0,
        "non_finite_inputs": 0,
        "missing_target": 0,
        "rejected_post_regime": 0,
    }
    boundary = pd.Timestamp(final_allowed_date) if final_allowed_date else None
    for ticker, group in merged.groupby("ticker", sort=True):
        matrices = arrays.matrices[str(ticker)]
        positions = matrices.position_of
        n_context = matrices.context.shape[1]
        for origin, target_row in zip(group["origin_date"], group.itertuples(),
                                      strict=True):
            if boundary is not None:
                assert_pre_covid_dates(
                    origin_dates=[origin], target_dates=[target_row.target_date],
                    feature_dates=[matrices.dates[
                        positions.get(str(pd.Timestamp(origin).date()), 0)]],
                    final_allowed_date=boundary,
                    where=f"precovid sample {ticker}")
            row = positions.get(str(pd.Timestamp(origin).date()))
            if row is None or row < sequence_length - 1:
                dropped["no_full_window"] += 1
                continue
            if not _window_is_usable(matrices, row, sequence_length, n_context):
                dropped["non_finite_inputs"] += 1
                continue
            if any(pd.isna(getattr(target_row, name)) for name in require_targets):
                dropped["missing_target"] += 1
                continue
            records.append({
                "ticker": str(ticker),
                "origin_date": pd.Timestamp(origin),
                "target_date": pd.Timestamp(target_row.target_date),
                "row": int(row),
                "ticker_id": arrays.ticker_id(str(ticker)),
                "sector_id": arrays.sector_id(str(ticker)),
                "y_direction": int(target_row.y_direction),
                "y_return": (float(target_row.y_return)
                             if hasattr(target_row, "y_return") else np.nan),
                "y_rank": (float(target_row.y_rank)
                           if hasattr(target_row, "y_rank") else np.nan),
            })

    frame = pd.DataFrame(records, columns=[
        "ticker", "origin_date", "target_date", "row", "ticker_id", "sector_id",
        "y_direction", "y_return", "y_rank",
    ])
    frame = frame.sort_values(["ticker", "origin_date"]).reset_index(drop=True)
    diagnostics = {
        "n_samples": len(frame),
        "n_tickers": int(frame["ticker"].nunique()) if len(frame) else 0,
        "sequence_length": int(sequence_length),
        "dropped": dropped,
        "first_origin_date": str(frame["origin_date"].min().date()) if len(frame) else None,
        "last_origin_date": str(frame["origin_date"].max().date()) if len(frame) else None,
        "last_target_date": str(frame["target_date"].max().date()) if len(frame) else None,
        "direction_positive_rate": float(frame["y_direction"].mean()) if len(frame) else None,
        "final_allowed_date": (None if boundary is None else str(boundary.date())),
        "max_origin_date_consumed": (str(pd.Timestamp(frame["origin_date"].max()).date())
                                     if len(frame) else None),
        "max_target_date_consumed": (str(pd.Timestamp(frame["target_date"].max()).date())
                                     if len(frame) else None),
    }
    return SampleTable(frame=frame, arrays=arrays, sequence_length=int(sequence_length),
                       diagnostics=diagnostics)


def split_masks(samples: SampleTable, window: SplitWindow) -> dict[str, np.ndarray]:
    """Train/validation masks under the origin AND target split rule."""
    frame = samples.frame
    origin = pd.to_datetime(frame["origin_date"])
    target = pd.to_datetime(frame["target_date"])
    train = ((origin >= pd.Timestamp(window.train_start))
             & (origin <= pd.Timestamp(window.train_end))
             & (target >= pd.Timestamp(window.train_start))
             & (target <= pd.Timestamp(window.train_end))).to_numpy()
    val = ((origin >= pd.Timestamp(window.val_start))
           & (origin <= pd.Timestamp(window.val_end))
           & (target >= pd.Timestamp(window.val_start))
           & (target <= pd.Timestamp(window.val_end))).to_numpy()
    if bool((train & val).any()):
        raise AssertionError("a sample cannot be in both the train and validation split")
    return {"train": train, "val": val}


def assert_split_targets_legal(masks: dict[str, np.ndarray], samples: SampleTable,
                               *, where: str, unlocked: bool | None = None,
                               final_allowed_date: str | None = None) -> None:
    """Apply the V2 firewalls to the target dates of every split.

    When ``final_allowed_date`` is given (the PRE-COVID regime) the post-2019
    rejection runs as well, so the split itself is proved clean rather than
    assumed clean.
    """
    frame = samples.frame
    for name, mask in masks.items():
        dates = frame.loc[mask, "target_date"]
        assert_no_paper_test_targets(dates, where=f"{where}/{name}")
        assert_no_lockbox_targets(dates, where=f"{where}/{name}", unlocked=unlocked)
        if final_allowed_date is not None:
            assert_pre_covid_dates(
                origin_dates=frame.loc[mask, "origin_date"], target_dates=dates,
                final_allowed_date=final_allowed_date, where=f"{where}/{name}")


# ---------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------

class GroupFeatureScaler:
    """Standardise one feature group, fitted on TRAIN samples only.

    ``passthrough`` columns (the percentile ranks) keep their native scale: they
    are already bounded in [0, 1], comparable across dates, and standardising
    them would only distort that property.
    """

    def __init__(self, columns: Sequence[str], passthrough: Sequence[str] = ()):
        self.columns = list(columns)
        self.passthrough = list(passthrough)
        self.scaler: StandardScaler | None = None
        self.fit_provenance: dict = {}

    @property
    def scaled_columns(self) -> list[str]:
        return [c for c in self.columns if c not in set(self.passthrough)]

    def fit(self, matrix: np.ndarray, provenance: dict | None = None) -> GroupFeatureScaler:
        """Fit on the matrix assembled from TRAIN samples only."""
        if matrix.ndim != 2:
            raise ValueError(f"expected a 2-D fit matrix, got shape {matrix.shape}")
        if matrix.shape[0] == 0:
            raise ValueError("refusing to fit a scaler on zero rows")
        self.fit_provenance = {
            "n_rows": int(matrix.shape[0]),
            "n_columns": int(matrix.shape[1]),
            "passthrough": list(self.passthrough),
            **(provenance or {}),
        }
        if matrix.shape[1] == 0:
            # A variant with no context features has nothing to standardise.
            self.scaler = None
            return self
        self.scaler = StandardScaler().fit(matrix)
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            if matrix.shape[1] == 0:
                return matrix.astype(np.float32)
            raise RuntimeError("GroupFeatureScaler.transform called before fit")
        return self.scaler.transform(matrix).astype(np.float32)

    def transform_passthrough(self, matrix: np.ndarray) -> np.ndarray:
        return matrix.astype(np.float32)

    def state(self) -> dict:
        return {
            "columns": self.columns,
            "passthrough": self.passthrough,
            "fit_provenance": self.fit_provenance,
        }


@dataclass
class V2ScalerBundle:
    """The two scalers V2 persists in its checkpoint."""

    stock: GroupFeatureScaler
    context: GroupFeatureScaler

    def state(self) -> dict:
        return {"stock": self.stock.state(), "context": self.context.state()}


def _assemble_origin_rows(samples: SampleTable, mask: np.ndarray, group: str,
                          columns: Sequence[str],
                          percentile: Sequence[str] = ()) -> np.ndarray:
    """Stack the ORIGIN row of every selected sample for one feature group."""
    frame = samples.frame.loc[mask]
    percentile = set(percentile)
    if not columns:
        return np.zeros((len(frame), 0), dtype=np.float32)
    blocks = []
    for ticker, row in zip(frame["ticker"], frame["row"], strict=True):
        matrices = samples.arrays.matrices[str(ticker)]
        source = matrices.stock if group == "stock" else matrices.context
        blocks.append(source[int(row), :])
    matrix = np.vstack(blocks).astype(np.float64) if blocks else np.zeros((0, 0))
    if percentile:
        keep = np.array([c not in percentile for c in columns])
        matrix = matrix[:, keep]
    return matrix


def fit_scalers_on_train(samples: SampleTable, train_mask: np.ndarray, *,
                          sequence_length: int | None = None) -> V2ScalerBundle:
    """Fit the global scalers on TRAIN samples and record their provenance.

    The provenance records the newest origin/target date that was allowed to
    influence the fit, which is what the causality tests assert on.
    """
    del sequence_length  # the origin row does not depend on the window length
    frame = samples.frame.loc[train_mask]
    provenance = {
        "n_train_samples": len(frame),
        "max_origin_date": str(pd.Timestamp(frame["origin_date"].max()).date())
        if len(frame) else None,
        "max_target_date": str(pd.Timestamp(frame["target_date"].max()).date())
        if len(frame) else None,
        "tickers": sorted(frame["ticker"].unique().tolist()),
        "fitted_on": "TRAIN_SAMPLES_ONLY",
    }

    stock_matrix = _assemble_origin_rows(
        samples, train_mask, "stock", samples.arrays.stock_features)
    stock_scaler = GroupFeatureScaler(samples.arrays.stock_features).fit(
        stock_matrix, provenance | {"group": "stock"})

    context_columns = samples.arrays.context_features
    percentile = samples.arrays.percentile_features
    if context_columns:
        context_matrix = _assemble_origin_rows(
            samples, train_mask, "context", context_columns, percentile=percentile)
        context_scaler = GroupFeatureScaler(context_columns, passthrough=percentile).fit(
            context_matrix, provenance | {"group": "context"})
    else:
        context_scaler = GroupFeatureScaler([], passthrough=[]).fit(
            np.zeros((max(len(frame), 1), 0)), provenance | {"group": "context"})
    return V2ScalerBundle(stock=stock_scaler, context=context_scaler)


# ---------------------------------------------------------------------------
# torch dataset
# ---------------------------------------------------------------------------

class V2SequenceDataset(Dataset):
    """Materialises windows on demand from the per-security matrices.

    Windows are sliced from the same matrices the scaler was fitted on, so a
    training row and a validation row go through exactly the same
    transformation.
    """

    def __init__(self, samples: SampleTable, frame: pd.DataFrame, *,
                 scalers: V2ScalerBundle | None = None,
                 sequence_length: int | None = None,
                 include_targets: bool = True) -> None:
        self.samples = samples
        self.frame = frame.reset_index(drop=True)
        self.scalers = scalers
        self.sequence_length = int(sequence_length or samples.sequence_length)
        self.include_targets = include_targets

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        ticker = str(row["ticker"])
        matrices = self.samples.arrays.matrices[ticker]
        end = int(row["row"]) + 1
        start = end - self.sequence_length

        stock = matrices.stock[start:end]
        context = matrices.context[start:end]
        regime = matrices.regime[int(row["row"])]

        if self.scalers is not None and self.scalers.stock.columns:
            stock = self.scalers.stock.transform(stock)
        if self.scalers is not None and self.scalers.context.columns:
            columns = self.scalers.context.columns
            percentile = set(self.scalers.context.passthrough)
            scaled_idx = [i for i, c in enumerate(columns) if c not in percentile]
            transformed = np.array(context, dtype=np.float32, copy=True)
            if scaled_idx:
                block = self.scalers.context.scaler.transform(context[:, scaled_idx])
                transformed[:, scaled_idx] = block.astype(np.float32)
        else:
            transformed = np.array(context, dtype=np.float32, copy=True)

        item: dict[str, object] = {
            "stock_sequence": torch.from_numpy(np.ascontiguousarray(stock, dtype=np.float32)),
            "context_sequence": torch.from_numpy(np.ascontiguousarray(transformed,
                                                                     dtype=np.float32)),
            "regime_vector": torch.from_numpy(np.ascontiguousarray(regime, dtype=np.float32)),
            "ticker_id": torch.tensor(int(row["ticker_id"]), dtype=torch.long),
            "sector_id": torch.tensor(int(row["sector_id"]), dtype=torch.long),
            "index": torch.tensor(int(index), dtype=torch.long),
        }
        if self.include_targets:
            item["y_direction"] = torch.tensor(float(row["y_direction"]), dtype=torch.float32)
            item["y_return"] = torch.tensor(float(row["y_return"]), dtype=torch.float32)
            item["y_rank"] = torch.tensor(float(row["y_rank"]), dtype=torch.float32)
        return item


def ticker_balanced_weights(frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Inverse-frequency sample weights so no security dominates the shared model.

    The eight development securities have similar histories, but the later
    full-50 run contains late listings, so the mechanism is implemented now and
    enabled by configuration.
    """
    counts = frame["ticker"].value_counts()
    inverse = 1.0 / counts.reindex(frame["ticker"]).to_numpy(dtype=float)
    weights = inverse / inverse.sum()
    return weights.astype(np.float64), {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "min_weight": float(weights.min()),
        "max_weight": float(weights.max()),
        "n_tickers": len(counts),
    }


def sample_metadata(frame: pd.DataFrame) -> dict:
    """Small provenance block recorded in manifests and the ledger."""
    if frame.empty:
        return {"n_samples": 0}
    return {
        "n_samples": len(frame),
        "n_tickers": int(frame["ticker"].nunique()),
        "tickers": sorted(frame["ticker"].unique().tolist()),
        "first_origin_date": str(pd.Timestamp(frame["origin_date"].min()).date()),
        "last_origin_date": str(pd.Timestamp(frame["origin_date"].max()).date()),
        "first_target_date": str(pd.Timestamp(frame["target_date"].min()).date()),
        "last_target_date": str(pd.Timestamp(frame["target_date"].max()).date()),
        "direction_positive_rate": float(frame["y_direction"].mean()),
    }