"""V3 sample construction: one sample table per FEATURE FAMILY, on shared targets.

THE COMPARISON THIS MODULE EXISTS TO MAKE FAIR
----------------------------------------------
An exogenous feature can be missing on a date the stock has data, so different
families naturally end up with different sample sets.  Comparing a family on its
own NATURAL sample set against X0 on the full set would attribute a difference in
the observation subset to a difference in the information.

So every family table is built on the SAME supervised universe, the SAME
multi-horizon targets and the SAME folds as the multi-horizon programme, and two
views are always reported:

NATURAL
    every sample for which that family's exogenous block is finite.
COMMON
    only the (ticker, origin) pairs whose exogenous block is finite for EVERY
    family X0..X4, so all five are scored on identical observations.

The horizon semantics, the split boundary rule and the 2019 seal are the
already-verified multi-horizon implementation, reused unchanged: this module
delegates to :mod:`agentic_forecaster.v2.horizon_dataset` rather than rebuilding
target semantics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agentic_forecaster.v2 import horizons as HZ
from agentic_forecaster.v2.dataset import SampleTable
from agentic_forecaster.v2.horizon_dataset import horizon_split
from agentic_forecaster.v2.metrics import train_majority_baseline

from . import CONTROL_FAMILY, FEATURE_FAMILIES
from .features import finite_row_mask, relative_to_market_features, sector_relative_features
from .sources import SourceRegistry
from .store import ExogenousStore, family_feature_columns

logger = logging.getLogger("agentic_forecaster.v3.dataset")

#: Relative-context columns added per stock, on top of the stock schema.
RELATIVE_COLUMNS: tuple[str, ...] = (
    "stock_minus_NIFTY50_return_1", "stock_minus_NIFTY50_return_5",
    "stock_minus_NIFTY50_return_20",
)

SECTOR_RELATIVE_COLUMNS: tuple[str, ...] = (
    "stock_minus_sector_return_1", "stock_minus_sector_return_5",
    "stock_minus_sector_return_20",
)

#: Index whose returns define the India-market relative context.
MARKET_INDEX = "NIFTY50"

#: Prefixes of SECURITY-SPECIFIC relative columns.  Everything else in a family's
#: block is source-derived, which is what the CORE-COMMON sample set is built from.
_RELATIVE_PREFIXES = ("stock_minus_",)


def _is_source_derived(column: str) -> bool:
    """True for a column produced by an accepted external source."""
    return not column.startswith(_RELATIVE_PREFIXES)


@dataclass
class FamilySampleTable:
    """One feature family's view of the shared supervised samples."""

    family: str
    horizon: int
    samples: SampleTable
    exogenous_columns: tuple[str, ...]
    exogenous: pd.DataFrame
    #: Source-DERIVED columns only.  These define the CORE-COMMON sample set; the
    #: security-specific relative blocks are excluded from it because a security
    #: without a verified sector index legitimately has none, and section 20 forbids
    #: dropping that security from the study.
    core_columns: tuple[str, ...] = ()
    diagnostics: dict = field(default_factory=dict)

    @property
    def objective_id(self) -> str:
        return HZ.objective_id(self.horizon)

    def exogenous_matrix(self, mask: np.ndarray) -> np.ndarray:
        if not self.exogenous_columns:
            return np.zeros((int(mask.sum()), 0), dtype=np.float64)
        block = self.exogenous.loc[mask, list(self.exogenous_columns)]
        return block.to_numpy(dtype=np.float64)

    def stock_matrix(self, mask: np.ndarray) -> np.ndarray:
        """The ORIGIN-row stationary stock features: the CONTROL information."""
        return _origin_stock_matrix(self.samples, mask)


def _origin_stock_matrix(samples: SampleTable, mask: np.ndarray) -> np.ndarray:
    frame = samples.frame.loc[mask]
    if frame.empty:
        return np.zeros((0, samples.arrays.n_stock_features), dtype=np.float64)
    blocks = [samples.arrays.matrices[str(t)].stock[int(r), :]
              for t, r in zip(frame["ticker"], frame["row"], strict=True)]
    return np.vstack(blocks).astype(np.float64)


def build_source_panel(store: ExogenousStore, registry: SourceRegistry, *, family: str,
                       dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """The date-keyed block: one row per stock trading date.

    These features do not vary by security, so a single row per date is correct and
    is broadcast to every security on that date.
    """
    own = family_feature_columns(store, family, registry=registry)
    if not own:
        return pd.DataFrame({"date": dates}), ()
    panel = pd.DataFrame({"date": dates}).merge(
        store.features.loc[:, ["date", *own]], on="date", how="left",
        validate="one_to_one")
    return panel, own


def build_relative_panel(store: ExogenousStore, stock_features: pd.DataFrame
                         ) -> pd.DataFrame:
    """``stock_return_k - NIFTY_return_k``, keyed by (ticker, origin_date).

    This block genuinely varies by security, so it is keyed by the PAIR and joined
    per sample rather than broadcast per date.
    """
    needed = [f"{MARKET_INDEX}_return_{window}" for window in (1, 5, 20)]
    if any(column not in store.features.columns for column in needed):
        return pd.DataFrame(columns=["ticker", "date", *RELATIVE_COLUMNS])
    stock = stock_features.loc[:, ["ticker", "date", "log_return_1", "log_return_5",
                                  "log_return_20"]].copy()
    stock["date"] = pd.to_datetime(stock["date"])
    merged = stock.merge(store.features.loc[:, ["date", *needed]], on="date",
                         how="inner", validate="many_to_one")
    relative = relative_to_market_features(merged, merged, index_prefix=MARKET_INDEX)
    relative.insert(0, "ticker", merged["ticker"].to_numpy())
    relative.insert(1, "origin_date", pd.DatetimeIndex(merged["date"]))
    return relative.loc[:, ["ticker", "origin_date", *RELATIVE_COLUMNS]]


#: Minimum share of supervised securities a sector-index mapping must cover before
#: the sector-relative block is used as a MODEL FEATURE.  Below it the block is
#: excluded and the reason recorded, because a shared model cannot be fitted on a
#: feature that is undefined for most securities, and a security must never be
#: dropped from the study merely because its sector index is missing.
MIN_SECTOR_COVERAGE = 0.5


def sector_index_coverage(sector_mapping: pd.DataFrame, tickers) -> dict:
    """How many supervised securities have a verified external sector index."""
    universe = sorted({str(t) for t in tickers})
    mapped = {str(t) for t in sector_mapping.loc[
        sector_mapping["external_sector_index_id"].astype(str).str.len() > 0, "ticker"]}
    covered = [t for t in universe if t in mapped]
    return {"n_supervised": len(universe), "n_with_sector_index": len(covered),
            "coverage_fraction": (len(covered) / len(universe) if universe else 0.0),
            "min_required_fraction": MIN_SECTOR_COVERAGE,
            "used_as_model_feature": bool(universe and len(covered) / len(universe)
                                          >= MIN_SECTOR_COVERAGE),
            "mapped_tickers": sorted(covered)}


def build_sector_panel(store: ExogenousStore, stock_features: pd.DataFrame,
                        sector_mapping: pd.DataFrame) -> pd.DataFrame:
    """``stock_return_k - <sector index>_return_k``, keyed by (ticker, origin_date).

    A security without a VERIFIED external sector index contributes nothing here.  It
    keeps its stock features and is NOT dropped from the study; only its
    sector-index block is unavailable.
    """
    empty = pd.DataFrame(columns=["ticker", "origin_date", *SECTOR_RELATIVE_COLUMNS])
    coverage = sector_index_coverage(sector_mapping, stock_features["ticker"].unique())
    if not coverage["used_as_model_feature"]:
        logger.info(
            "sector-index features EXCLUDED as a model feature: only %d of %d "
            "supervised securities have a verified external sector index "
            "(%.1f%% < %.0f%% required). The mapping is still recorded and audited.",
            coverage["n_with_sector_index"], coverage["n_supervised"],
            100 * coverage["coverage_fraction"], 100 * MIN_SECTOR_COVERAGE)
        return empty
    mapped = sector_mapping.loc[
        sector_mapping["external_sector_index_id"].astype(str).str.len() > 0]
    if mapped.empty:
        return empty
    per_ticker_source = dict(zip(mapped["ticker"], mapped["external_sector_index_id"],
                                 strict=True))
    stock = stock_features.loc[:, ["ticker", "date", "log_return_1", "log_return_5",
                                  "log_return_20"]].copy()
    stock["date"] = pd.to_datetime(stock["date"])
    blocks = []
    for ticker, source_id in per_ticker_source.items():
        needed = [f"{source_id}_return_{window}" for window in (1, 5, 20)]
        if any(column not in store.features.columns for column in needed):
            continue
        block = (stock.loc[stock["ticker"] == ticker]
                 .merge(store.features.loc[:, ["date", *needed]], on="date",
                        how="inner", validate="many_to_one"))
        if block.empty:
            continue
        relative = sector_relative_features(block, block, index_prefix=source_id)
        relative.insert(0, "ticker", ticker)
        relative.insert(1, "origin_date", pd.DatetimeIndex(block["date"]))
        blocks.append(relative.loc[:, ["ticker", "origin_date",
                                        *SECTOR_RELATIVE_COLUMNS]])
    if not blocks:
        return empty
    combined = pd.concat(blocks, ignore_index=True)
    # collapse on an explicit KEY subset: dropping duplicates across every column
    # would also hash the float features, which is slower and backend-dependent
    return combined.drop_duplicates(subset=["ticker", "origin_date"], keep="first")


def build_family_tables(samples_by_horizon: dict[int, SampleTable],
                        store: ExogenousStore, registry: SourceRegistry, *,
                        stock_features: pd.DataFrame) -> dict[tuple[int, str],
                                                               FamilySampleTable]:
    """Build every (horizon, family) table on the shared supervised samples.

    A family table is the SAME stock sample table with an exogenous block attached,
    so the target, the origin, the input window and the split rule are identical
    across families by construction.  The exogenous block is aligned ROW BY ROW with
    ``samples.frame``, which is what lets security-specific relative features
    (stock minus NIFTY, stock minus sector index) live beside date-keyed ones.
    """
    relative = build_relative_panel(store, stock_features)
    sector = build_sector_panel(store, stock_features, store.sector_mapping)
    coverage = sector_index_coverage(store.sector_mapping,
                                     stock_features["ticker"].unique())
    out: dict[tuple[int, str], FamilySampleTable] = {}

    for horizon, samples in samples_by_horizon.items():
        frame = samples.frame
        keys = pd.DataFrame({"ticker": frame["ticker"].to_numpy(),
                             "origin_date": pd.DatetimeIndex(
                                 pd.to_datetime(frame["origin_date"]))})
        for family in FEATURE_FAMILIES:
            if family == CONTROL_FAMILY:
                # X0 has NO exogenous block at all: that is what makes it the control
                exogenous = pd.DataFrame(index=range(len(frame)))
                columns: tuple[str, ...] = ()
            else:
                dates = pd.DatetimeIndex(sorted(keys["origin_date"].unique()))
                panel, own = build_source_panel(store, registry, family=family,
                                                dates=dates)
                block = keys.merge(panel, left_on="origin_date", right_on="date",
                                   how="left", validate="many_to_one")
                columns = list(own)
                if family in ("X1_INDIA_MARKET", "X4_ALL_EXOGENOUS"):
                    # the left side repeats a date across securities, so the right
                    # side must be the unique one
                    block = block.merge(relative, on=["ticker", "origin_date"],
                                        how="left", validate="many_to_one")
                    columns += list(RELATIVE_COLUMNS)
                    if coverage["used_as_model_feature"]:
                        block = block.merge(sector, on=["ticker", "origin_date"],
                                            how="left", validate="many_to_one")
                        columns += list(SECTOR_RELATIVE_COLUMNS)
                exogenous = block.loc[:, columns].reset_index(drop=True)
            mask = (finite_row_mask(exogenous, columns) if columns
                    else np.ones(len(frame), dtype=bool))
            core = tuple(c for c in columns if _is_source_derived(c))
            out[(horizon, family)] = FamilySampleTable(
                family=family, horizon=horizon, samples=samples,
                exogenous_columns=tuple(columns), exogenous=exogenous,
                core_columns=core,
                diagnostics={
                    "family": family,
                    "objective_id": HZ.objective_id(horizon),
                    "horizon_phrase": HZ.horizon_phrase(horizon),
                    "n_samples_total": len(frame),
                    "n_samples_with_family": int(mask.sum()),
                    "n_exogenous_columns": len(columns),
                    "excluded_for_missing_exogenous": int((~mask).sum()),
                    "coverage_fraction": float(mask.mean()) if len(mask)
                    else float("nan"),
                    "exogenous_columns": list(columns),
                    "core_columns": list(core),
                    "core_definition": ("source-derived features only; the "
                                        "security-specific relative blocks are "
                                        "excluded so an unmapped sector cannot empty "
                                        "the common sample set"),
                    "sector_index_coverage": coverage,
                })
    return out


def family_split_mask(table: FamilySampleTable, window, *, fold: str = "",
                      locked: bool = False) -> dict[str, np.ndarray]:
    """Train/validation masks for one family, restricted to its finite rows.

    The split RULE is the verified multi-horizon one (origin AND target inside the
    window); the only extra restriction is that the exogenous block is finite, which
    is what makes the family's own sample set explicit rather than accidental.
    """
    split = horizon_split(table.samples, window, fold=fold or window.name,
                          locked=locked, where=f"v3/{table.family}")
    finite = finite_row_mask(table.exogenous, table.exogenous_columns) \
        if table.exogenous_columns else np.ones(len(table.samples.frame), dtype=bool)
    return {"train": split.train & finite, "val": split.val & finite}


def common_family_mask(tables: dict[tuple[int, str], FamilySampleTable], window, *,
                       families=FEATURE_FAMILIES, horizon: int, fold: str = "",
                       locked: bool = False) -> dict[str, dict[str, np.ndarray]]:
    """Masks restricted to samples finite for EVERY family simultaneously.

    This is the COMMON-SAMPLE view: the five families are then scored on identical
    (ticker, origin) pairs, so an apparent improvement can only come from the
    information and not from a different observation subset.
    """
    per_family = {family: family_split_mask(tables[(horizon, family)], window,
                                           fold=fold, locked=locked)
                  for family in families if (horizon, family) in tables}

    def _finite(table: FamilySampleTable, columns) -> np.ndarray:
        if not columns:
            return np.ones(len(table.samples.frame), dtype=bool)
        return np.isfinite(table.exogenous[list(columns)].to_numpy(float)).all(axis=1)

    # CORE-COMMON: every family's SOURCE-DERIVED features are finite.  A family is
    # then additionally restricted to its OWN full column set, so its common-view
    # score is still computed only on rows where its own inputs exist.
    core = np.logical_and.reduce(
        [_finite(tables[(horizon, family)], tables[(horizon, family)].core_columns)
         for family in per_family])
    out = {}
    for family, masks in per_family.items():
        own = _finite(tables[(horizon, family)], tables[(horizon, family)].exogenous_columns)
        allowed = core & own
        out[family] = {"train": masks["train"] & allowed, "val": masks["val"] & allowed,
                       "n_train": int((masks["train"] & allowed).sum()),
                       "n_val": int((masks["val"] & allowed).sum())}
    out["__core__"] = {"train": core, "val": core}
    return out


def design_matrix(table: FamilySampleTable, mask: np.ndarray, *, use_context: bool = True
                  ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """``(stock_design, exogenous_design, stock_columns, exogenous_columns)``.

    ``stock_design`` is the ORIGIN row of the 27 stationary stock features.  For X0
    the exogenous block is empty, which makes X0 literally the same control the
    multi-horizon programme already measured.
    """
    stock = table.stock_matrix(mask)
    exogenous = table.exogenous_matrix(mask)
    stock_columns = list(table.samples.arrays.stock_features)
    exogenous_columns = (list(table.exogenous_columns) if use_context else [])
    if not use_context:
        exogenous = np.zeros((stock.shape[0], 0), dtype=np.float64)
    return stock, exogenous, stock_columns, exogenous_columns


def baseline_for(table: FamilySampleTable, train_mask: np.ndarray) -> dict:
    """TRAIN-majority baseline, globally and per ticker.

    A validation-oracle majority class would leak the answer into the baseline, so
    the majority class is always taken from TRAIN.
    """
    frame = table.samples.frame.loc[train_mask]
    if frame.empty:
        return {"global": float("nan"), "per_ticker": {}}
    labels = frame["y_direction"].to_numpy(int)
    per_ticker = {str(ticker): train_majority_baseline(group["y_direction"].to_numpy())
                  for ticker, group in frame.groupby("ticker", sort=True)}
    return {"global": train_majority_baseline(labels), "per_ticker": per_ticker,
            "n_train": len(frame)}


def ticker_breadth(predictions: pd.DataFrame, baselines: dict[str, float]) -> dict:
    """Fraction of securities beating their OWN train-majority predictor."""
    if predictions.empty or not baselines:
        return {"fraction": float("nan"), "n_tickers": 0, "beating": 0}
    per_ticker = predictions.groupby("ticker", sort=True).apply(
        lambda g: float(((g["p_up"].to_numpy(float) >= 0.5).astype(int)
                         == g["y_true"].to_numpy(int)).mean()),
        include_groups=False)
    beating = [ticker for ticker, accuracy in per_ticker.items()
               if ticker in baselines and accuracy > baselines[ticker]]
    return {"fraction": float(len(beating) / len(per_ticker)) if len(per_ticker)
            else float("nan"),
            "n_tickers": len(per_ticker),
            "beating": len(beating),
            "per_ticker_accuracy": {str(k): float(v) for k, v in per_ticker.items()}}