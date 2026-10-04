"""Causal stationary exogenous features.

Raw index levels are never fed to a model: a level is not stationary across
instruments, epochs or regimes.  Every accepted price/index series becomes
scale-free, backward-looking quantities computed from that series' own history.

FOR EVERY PRICE / INDEX SERIES
    return_1, return_3, return_5, return_10, return_20
    realized_vol_5, realized_vol_10, realized_vol_20
    drawdown_20, drawdown_60
    distance_from_ma20, distance_from_ma60

FOR INDIA VIX (a volatility INDEX, where the level itself is the signal)
    level, log_level, change_1, change_5, zscore_20

FOR YIELDS (quoted in percent, so no log transform on the level)
    level, change_1, change_5, change_20

CAUSALITY
---------
Every window is BACKWARD looking and includes ``t``.  The first ``H`` rows of a
windowed statistic are NaN and are never filled: a sample whose required history
does not exist is DROPPED, not repaired.  ``assert_causal`` re-proves the
property by construction after the fact, which is what the feature-construction
tests assert.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .firewall import assert_no_post_2019
from .sources import ASSET_RATES, ASSET_VOLATILITY, SourceSpec

logger = logging.getLogger("agentic_forecaster.v3.features")

#: Feature templates for a price / index series.
PRICE_TEMPLATES: tuple[str, ...] = (
    "return_1", "return_3", "return_5", "return_10", "return_20",
    "realized_vol_5", "realized_vol_10", "realized_vol_20",
    "drawdown_20", "drawdown_60",
    "distance_from_ma20", "distance_from_ma60",
)

#: Feature templates for a volatility index such as India VIX.
VOLATILITY_TEMPLATES: tuple[str, ...] = (
    "level", "log_level", "change_1", "change_5", "zscore_20",
)

#: Feature templates for a yield series quoted in percent.
YIELD_TEMPLATES: tuple[str, ...] = ("level", "change_1", "change_5", "change_20")

RETURN_WINDOWS: tuple[int, ...] = (1, 3, 5, 10, 20)
VOL_WINDOWS: tuple[int, ...] = (5, 10, 20)
DRAWDOWN_WINDOWS: tuple[int, ...] = (20, 60)
MA_WINDOWS: tuple[int, ...] = (20, 60)


@dataclass(frozen=True)
class ExogenousFeatureSpec:
    """The feature schema one accepted source contributes."""

    source_id: str
    asset_class: str
    availability_policy: str
    templates: tuple[str, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(f"{self.source_id}_{name}" for name in self.templates)

    def to_dict(self) -> dict:
        return {"source_id": self.source_id, "asset_class": self.asset_class,
                "availability_policy": self.availability_policy,
                "templates": list(self.templates), "n_features": len(self.templates),
                "columns": list(self.columns)}


def templates_for(spec: SourceSpec) -> tuple[str, ...]:
    """Which template family a source gets, from its ASSET CLASS."""
    if spec.asset_class == ASSET_VOLATILITY:
        return VOLATILITY_TEMPLATES
    if spec.asset_class == ASSET_RATES:
        return YIELD_TEMPLATES
    return PRICE_TEMPLATES


def build_price_features(close: pd.Series, *, prefix: str) -> pd.DataFrame:
    """Causal stationary features of ONE price/index series."""
    level = pd.to_numeric(pd.Series(close), errors="coerce").astype(float)
    out = pd.DataFrame(index=level.index)
    log_level = np.log(level.where(level > 0))

    for window in RETURN_WINDOWS:
        out[f"{prefix}_return_{window}"] = log_level - log_level.shift(window)
    for window in VOL_WINDOWS:
        # realised volatility of the last `window` daily log returns, annualised
        # per cent so that an Indian and a US series are on one comparable scale
        returns = log_level.diff()
        out[f"{prefix}_realized_vol_{window}"] = (
            returns.rolling(window, min_periods=window).std(ddof=1)
            * np.sqrt(252.0) * 100.0)
    for window in DRAWDOWN_WINDOWS:
        rolling_max = level.rolling(window, min_periods=window).max()
        out[f"{prefix}_drawdown_{window}"] = np.where(
            rolling_max > 0, level / rolling_max - 1.0, np.nan)
    for window in MA_WINDOWS:
        ma = level.rolling(window, min_periods=window).mean()
        out[f"{prefix}_distance_from_ma{window}"] = np.where(
            ma > 0, level / ma - 1.0, np.nan)
    return out


def build_volatility_features(level: pd.Series, *, prefix: str) -> pd.DataFrame:
    """India VIX style features: the LEVEL matters, so it is kept."""
    values = pd.to_numeric(pd.Series(level), errors="coerce").astype(float)
    out = pd.DataFrame(index=values.index)
    out[f"{prefix}_level"] = values
    out[f"{prefix}_log_level"] = np.log(values.where(values > 0))
    out[f"{prefix}_change_1"] = values - values.shift(1)
    out[f"{prefix}_change_5"] = values - values.shift(5)
    mean = values.rolling(20, min_periods=20).mean()
    std = values.rolling(20, min_periods=20).std(ddof=1)
    out[f"{prefix}_zscore_20"] = (values - mean) / std.where(std > 0)
    return out


def build_yield_features(level: pd.Series, *, prefix: str) -> pd.DataFrame:
    """Yield features.  The level is already a percentage, so no log transform."""
    values = pd.to_numeric(pd.Series(level), errors="coerce").astype(float)
    out = pd.DataFrame(index=values.index)
    out[f"{prefix}_level"] = values
    out[f"{prefix}_change_1"] = values - values.shift(1)
    out[f"{prefix}_change_5"] = values - values.shift(5)
    out[f"{prefix}_change_20"] = values - values.shift(20)
    return out


def build_source_features(source_frame: pd.DataFrame, spec: SourceSpec, *,
                          date_col: str = "source_date",
                          final_allowed_date: str = "2019-12-31") -> pd.DataFrame:
    """Causal features for one accepted source, keyed by ``source_date``.

    The features are computed on the SOURCE's own calendar, before any alignment,
    so an aligned value always carries the information that was available on that
    source observation date.
    """
    frame = source_frame.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values(date_col).drop_duplicates(date_col, keep="last")
    assert_no_post_2019(exogenous_dates=frame[date_col],
                        where=f"features/{spec.source_id}",
                        final_allowed_date=final_allowed_date)
    close = frame["close"]
    if spec.asset_class == ASSET_VOLATILITY:
        built = build_volatility_features(close, prefix=spec.source_id)
    elif spec.asset_class == ASSET_RATES:
        built = build_yield_features(close, prefix=spec.source_id)
    else:
        built = build_price_features(close, prefix=spec.source_id)
    built.index = frame[date_col].to_numpy()
    built = built.reset_index().rename(columns={"index": date_col})
    built.insert(0, "source_observation_date", built[date_col])
    return built.loc[:, ["source_date", "source_observation_date", *built.columns[2:]]]


def assert_causal(features: pd.DataFrame, *, source_observation_col: str,
                  feature_date_col: str = "source_date",
                  where: str = "exogenous features") -> dict:
    """Re-prove that no feature carries information from after its own date.

    Every exogenous feature is a backward-looking function of ONE source
    observation, so the feature date and the source observation date are equal by
    construction.  What must be checked is the ALIGNMENT, which happens later.
    """
    if features.empty:
        return {"checked": 0}
    dates = pd.to_datetime(features[feature_date_col])
    observations = pd.to_datetime(features[source_observation_col])
    violations = int((observations > dates).sum())
    if violations:
        raise AssertionError(
            f"{where}: {violations} feature row(s) carry a source observation AFTER "
            "their own feature date, which is a causality violation.")
    return {"checked": len(features), "violations": 0}


def sector_relative_features(stock_frame: pd.DataFrame, sector_frame: pd.DataFrame, *,
                             index_prefix: str,
                             prefix: str = "stock") -> pd.DataFrame:
    """``stock_return_k - <sector index>_return_k`` for k in 1, 5, 20.

    Both inputs are FRAMES: the stock columns and the sector-index columns live in
    one joined frame, and selecting them by name is what keeps the two apart.
    """
    out = pd.DataFrame(index=stock_frame.index)
    for window in (1, 5, 20):
        stock_column = f"log_return_{window}"
        sector_column = f"{index_prefix}_return_{window}"
        if stock_column not in stock_frame.columns or sector_column not in sector_frame:
            continue
        out[f"{prefix}_minus_sector_return_{window}"] = (
            stock_frame[stock_column] - sector_frame[sector_column])
    return out


def relative_to_market_features(stock_frame: pd.DataFrame, index_frame: pd.DataFrame,
                                *, index_prefix: str) -> pd.DataFrame:
    """``stock_return_k - <index>_return_k`` for k in 1, 5, 20.

    ``stock_frame`` must carry ``log_return_1/5/20`` from the existing stationary
    stock schema; ``index_frame`` carries the aligned index return columns.
    """
    out = pd.DataFrame(index=stock_frame.index)
    for window in (1, 5, 20):
        stock_column = f"log_return_{window}"
        index_column = f"{index_prefix}_return_{window}"
        if stock_column not in stock_frame.columns or index_column not in index_frame:
            continue
        out[f"stock_minus_{index_prefix}_return_{window}"] = (
            stock_frame[stock_column] - index_frame[index_column])
    return out


def drop_degenerate_columns(frame: pd.DataFrame, *, min_finite_fraction: float = 0.5,
                            where: str = "exogenous features") -> tuple[pd.DataFrame, list]:
    """Drop a feature that carries no usable variation.

    Three kinds of column are removed, each explicitly and each RECORDED rather
    than silently substituted:

    * entirely missing (undefined, so a sample's vector cannot be formed);
    * present on fewer than ``min_finite_fraction`` of rows (too sparse to be a
      genuine feature, and must not be invented);
    * constant (zero variance, so it cannot inform any model).
    """
    numeric = frame.select_dtypes(include=[np.number])
    rows = max(len(numeric), 1)
    floor = max(int(min_finite_fraction * rows), 1)
    all_nan = [c for c in numeric.columns if int(numeric[c].notna().sum()) == 0]
    too_sparse = [c for c in numeric.columns
                  if c not in all_nan and int(numeric[c].notna().sum()) < floor]
    constant = [c for c in numeric.columns
                if c not in all_nan and c not in too_sparse
                and float(numeric[c].std(ddof=1) or 0.0) == 0.0]
    dropped = sorted(set(all_nan) | set(too_sparse) | set(constant))
    if dropped:
        logger.info("%s: dropping %d degenerate column(s): %s", where, len(dropped),
                    dropped[:6])
    return frame.drop(columns=dropped), dropped


def finite_row_mask(frame: pd.DataFrame, columns: tuple[str, ...] | None = None
                    ) -> np.ndarray:
    """Rows where EVERY exogenous feature is finite.

    A missing external observation must not be filled with an invented value, so a
    sample with any undefined exogenous feature is excluded from the exogenous
    families rather than imputed.
    """
    if not columns:
        columns = tuple(frame.columns)
    if not columns:
        return np.ones(len(frame), dtype=bool)
    matrix = frame.loc[:, list(columns)].to_numpy(dtype=float)
    return np.isfinite(matrix).all(axis=1)