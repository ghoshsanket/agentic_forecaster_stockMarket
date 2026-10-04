"""V2 cross-sectional CONTEXT: leave-one-out market and sector statistics,
relative-to-market/sector features, cross-sectional ranks and the regime vector.

Everything in this module is built from the SAME 50-stock Yahoo universe that
supplies the price data.  No news, sentiment, options, FII/DII, VIX, macro or
alternative data is used, so the first V2 experiment is attributable to
architecture plus representation rather than to new information sources.

Prediction time
---------------
All features at date ``t`` use information available by the CLOSE of ``t``.
The next-day rank is a LABEL (built in ``store.py`` from ``t+1`` returns) and is
never an input.

Leave-one-out (LOO) discipline
------------------------------
A stock's own return is part of every market aggregate, so a market feature fed
back to that stock partly restates the stock's own input.  Every market and
sector statistic supplied to stock ``i`` therefore EXCLUDES ``i``:

* same-date statistics use the sum/count and sum/sumsq identities
  ``(S - r_i) / (C - 1)`` and
  ``sqrt(((SS - r_i^2) - (S - r_i)^2 / (C - 1)) / (C - 2))``;
* rolling statistics first build each stock's OWN LOO market/sector return
  proxy series, then roll that proxy causally over time.

When the remaining group size is too small to define the statistic the value is
NaN (``group size <= 1`` is marked unavailable) rather than invented.

Missing history is never fabricated: a security that had no bar on date ``t``
simply does not contribute to that date's aggregates, and
``market_available_count`` records how many did.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("agentic_forecaster.v2.context")

#: Raw quantity used for every cross-sectional aggregate.
RETURN_COLUMN = "log_return_1"

#: Rolling windows for momentum / volatility / drawdown aggregates.
SHORT_WINDOW = 5
LONG_WINDOW = 20
DRAWDOWN_WINDOW = 60

#: Market-level aggregates over ALL available stocks (the market proxy itself).
MARKET_FEATURES: tuple[str, ...] = (
    "market_return_mean",
    "market_return_median",
    "market_breadth",
    "market_dispersion",
    "market_momentum_5",
    "market_momentum_20",
    "market_realized_vol_20",
    "market_drawdown_60",
    "market_mean_volume_surprise",
    "market_available_count",
)

#: Leave-one-out market features, i.e. what stock ``i`` is told about the rest
#: of the universe on date ``t``.
MARKET_LOO_FEATURES: tuple[str, ...] = (
    "market_return_mean_loo",
    "market_breadth_loo",
    "market_dispersion_loo",
    "market_momentum_5_loo",
    "market_momentum_20_loo",
    "market_realized_vol_20_loo",
    "market_drawdown_60_loo",
    "market_loo_available_count",
)

#: Sector aggregates: the sector itself and the same quantities with the stock
#: itself removed.
SECTOR_FEATURES: tuple[str, ...] = (
    "sector_return_mean",
    "sector_breadth",
    "sector_available_count",
    "sector_return_mean_loo",
    "sector_breadth_loo",
    "sector_dispersion_loo",
    "sector_momentum_5_loo",
    "sector_momentum_20_loo",
    "sector_realized_vol_20_loo",
    "sector_loo_available_count",
)

#: Relative-to-market / relative-to-sector momentum features.
RELATIVE_FEATURES: tuple[str, ...] = (
    "stock_minus_market_return",
    "stock_minus_sector_return",
    "momentum_5_minus_market",
    "momentum_20_minus_market",
    "momentum_5_minus_sector",
    "momentum_20_minus_sector",
)

#: Quantities ranked across the available universe on each date.
RANK_SOURCE_COLUMNS: tuple[str, ...] = (
    "log_return_1",
    "log_return_5",
    "log_return_20",
    "rsi_14_centered",
    "volume_z_20",
    "realized_vol_20",
    "stock_minus_market_return",
    "stock_minus_sector_return",
)

#: Input feature names of the percentile ranks, in a fixed order.
RANK_FEATURES: tuple[str, ...] = tuple(f"rank_{c}" for c in RANK_SOURCE_COLUMNS)

#: Regime vector fed to the regime encoder, taken from the ORIGIN date.
REGIME_FEATURES: tuple[str, ...] = (
    "market_return_mean_loo",
    "market_momentum_5_loo",
    "market_momentum_20_loo",
    "market_realized_vol_20_loo",
    "market_breadth_loo",
    "market_dispersion_loo",
    "market_drawdown_60_loo",
)

#: Every continuous context column handed to the shared model.
CONTEXT_FEATURES: tuple[str, ...] = (
    *MARKET_FEATURES,
    *MARKET_LOO_FEATURES,
    *SECTOR_FEATURES,
    *RELATIVE_FEATURES,
)

#: Percentile columns stay in [0, 1] and are NOT standardised.
PERCENTILE_FEATURES: tuple[str, ...] = RANK_FEATURES

#: Raw columns the cross-sectional ranks are computed from.
CONTEXT_PANEL_COLUMNS: tuple[str, ...] = (
    "log_return_1",
    "log_return_5",
    "log_return_20",
    "rsi_14_centered",
    "volume_z_20",
    "realized_vol_20",
)

_MARKET_LOO_PROXY = "_market_loo_proxy"
_SECTOR_LOO_PROXY = "_sector_loo_proxy"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _rolling_sum(series: pd.Series, window: int, min_periods: int | None = None
                 ) -> pd.Series:
    return series.rolling(window, min_periods=min_periods or window).sum()


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).std(ddof=1)


def _rolling_drawdown(series: pd.Series, window: int) -> pd.Series:
    peak = series.rolling(window, min_periods=window).max()
    return series / peak - 1.0


def _loo_moments(frame: pd.DataFrame, *, count_col: str, sum_col: str,
                 sumsq_col: str, positive_col: str, count: pd.Series) -> pd.DataFrame:
    """Leave-one-out mean, breadth and dispersion on each date.

    ``frame`` is sorted by date and carries the per-date sum/count identities.
    Formulas:

    * mean    ``(S - r) / (C - 1)``
    * breadth ``(P - [r > 0]) / (C - 1)``
    * std     ``sqrt(((SS - r^2) - (S - r)^2 / (C - 1)) / (C - 2))``
    """
    n = count - 1
    out = pd.DataFrame(index=frame.index)
    s, ss, p = frame[sum_col], frame[sumsq_col], frame[positive_col]

    mean_loo = (s - frame["_value"]) / n.replace(0, np.nan)
    breadth_loo = (p - (frame["_value"] > 0).astype(float)) / n.replace(0, np.nan)
    variance_num = (ss - frame["_value"] ** 2) - (s - frame["_value"]) ** 2 / n.replace(0, np.nan)
    std_loo = np.sqrt(variance_num / (count - 2).replace(0, np.nan))
    remaining = n.to_numpy()

    out["loo_mean"] = mean_loo.where(n > 0)
    out["loo_breadth"] = breadth_loo.where(n > 0)
    out["loo_dispersion"] = std_loo.where(count > 2)
    out["loo_count"] = np.where(remaining > 0, remaining, np.nan)
    return out


# ---------------------------------------------------------------------------
# market context
# ---------------------------------------------------------------------------

def build_market_context(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build date-level market aggregates and per-stock LOO market features.

    Parameters
    ----------
    panel:
        Long frame with ``ticker``, ``date`` and the columns in
        :data:`CONTEXT_PANEL_COLUMNS`.

    Returns
    -------
    ``(market, loo)`` where ``market`` is one row per date (the market proxy)
    and ``loo`` is one row per (ticker, date) carrying the leave-one-out view
    that stock sees.
    """
    if panel.empty:
        raise ValueError("empty panel: cannot build market context")

    values = panel.loc[:, ["ticker", "date", RETURN_COLUMN]].copy()
    values = values.loc[values[RETURN_COLUMN].notna()]
    values = values.sort_values(["date", "ticker"]).reset_index(drop=True)
    values["_value"] = values[RETURN_COLUMN].astype(float)
    values["_positive"] = (values["_value"] > 0).astype(float)
    values["_value_sq"] = values["_value"] ** 2

    grouped = values.groupby("date", sort=True)
    moments = grouped.agg(
        _count=(RETURN_COLUMN, "size"),
        _sum=("_value", "sum"),
        _sumsq=("_value_sq", "sum"),
        _positive_count=("_positive", "sum"),
    )

    market = pd.DataFrame(index=moments.index)
    count = moments["_count"].astype(float)
    market["market_return_mean"] = moments["_sum"] / count
    market["market_breadth"] = moments["_positive_count"] / count
    market["market_dispersion"] = grouped[RETURN_COLUMN].std(ddof=1)
    market["market_return_median"] = grouped[RETURN_COLUMN].median()
    market["market_available_count"] = count
    surprise = panel.loc[panel["volume_z_20"].notna(), ["date", "volume_z_20"]]
    market["market_mean_volume_surprise"] = surprise.groupby("date")["volume_z_20"].mean()

    proxy = market["market_return_mean"]
    market["market_momentum_5"] = _rolling_sum(proxy, SHORT_WINDOW)
    market["market_momentum_20"] = _rolling_sum(proxy, LONG_WINDOW)
    market["market_realized_vol_20"] = _rolling_std(proxy, LONG_WINDOW)
    market["market_drawdown_60"] = _rolling_drawdown(proxy, DRAWDOWN_WINDOW)

    loo = values.merge(moments, left_on="date", right_index=True, how="left")
    loo_stats = _loo_moments(
        loo, count_col="_count", sum_col="_sum", sumsq_col="_sumsq",
        positive_col="_positive_count", count=loo["_count"].astype(float),
    )
    loo_out = pd.DataFrame({
        "ticker": loo["ticker"],
        "date": loo["date"],
        "market_return_mean_loo": loo_stats["loo_mean"].to_numpy(),
        "market_breadth_loo": loo_stats["loo_breadth"].to_numpy(),
        "market_dispersion_loo": loo_stats["loo_dispersion"].to_numpy(),
        "market_loo_available_count": loo_stats["loo_count"].to_numpy(),
    })

    # Rolling LOO statistics: build the stock's OWN leave-one-out market-return
    # proxy series first, then roll it causally over that stock's history.
    proxy_by_stock = loo_out.sort_values(["ticker", "date"]).copy()
    proxy_by_stock[_MARKET_LOO_PROXY] = proxy_by_stock["market_return_mean_loo"]
    rolled = proxy_by_stock.groupby("ticker", sort=False)[_MARKET_LOO_PROXY]
    proxy_by_stock["market_momentum_5_loo"] = rolled.transform(
        lambda s: _rolling_sum(s, SHORT_WINDOW))
    proxy_by_stock["market_momentum_20_loo"] = rolled.transform(
        lambda s: _rolling_sum(s, LONG_WINDOW))
    proxy_by_stock["market_realized_vol_20_loo"] = rolled.transform(
        lambda s: _rolling_std(s, LONG_WINDOW))
    proxy_by_stock["market_drawdown_60_loo"] = rolled.transform(
        lambda s: _rolling_drawdown(s, DRAWDOWN_WINDOW))
    loo_out = proxy_by_stock.loc[:, [
        "ticker", "date", "market_return_mean_loo", "market_breadth_loo",
        "market_dispersion_loo", "market_momentum_5_loo", "market_momentum_20_loo",
        "market_realized_vol_20_loo", "market_drawdown_60_loo",
        "market_loo_available_count",
    ]].reset_index(drop=True)

    market = market.reset_index()
    market = market.loc[:, ["date", *MARKET_FEATURES]]
    return market, loo_out


# ---------------------------------------------------------------------------
# sector context
# ---------------------------------------------------------------------------

def build_sector_context(panel: pd.DataFrame, sector_of: dict[str, str]
                         ) -> pd.DataFrame:
    """Per-(ticker, date) leave-one-out sector aggregates.

    Sector membership is STATIC metadata (see ``sectors.py``); the statistics
    are computed from the same cross-section, so a sector with a single
    available member yields NaN rather than the member's own return.
    """
    if panel.empty:
        raise ValueError("empty panel: cannot build sector context")

    frame = panel.loc[:, ["ticker", "date", RETURN_COLUMN]].copy()
    frame["sector"] = frame["ticker"].map(lambda t: sector_of.get(str(t), "UNKNOWN"))
    frame = frame.loc[frame[RETURN_COLUMN].notna()].copy()
    frame["_value"] = frame[RETURN_COLUMN].astype(float)
    frame["_positive"] = (frame["_value"] > 0).astype(float)
    frame["_value_sq"] = frame["_value"] ** 2
    frame = frame.sort_values(["sector", "ticker", "date"]).reset_index(drop=True)

    grouped = frame.groupby(["sector", "date"], sort=True)
    moments = grouped.agg(
        _count=(RETURN_COLUMN, "size"),
        _sum=("_value", "sum"),
        _sumsq=("_value_sq", "sum"),
        _positive_count=("_positive", "sum"),
    ).reset_index()

    frame = frame.merge(moments, on=["sector", "date"], how="left")
    stats = _loo_moments(
        frame, count_col="_count", sum_col="_sum", sumsq_col="_sumsq",
        positive_col="_positive_count", count=frame["_count"].astype(float),
    )
    count = frame["_count"].astype(float)
    sector = pd.DataFrame({
        "ticker": frame["ticker"].to_numpy(),
        "date": frame["date"].to_numpy(),
        "sector": frame["sector"].to_numpy(),
        "sector_return_mean": (frame["_sum"] / count).to_numpy(),
        "sector_breadth": (frame["_positive"] / count).to_numpy(),
        "sector_available_count": count.to_numpy(),
        "sector_return_mean_loo": stats["loo_mean"].to_numpy(),
        "sector_breadth_loo": stats["loo_breadth"].to_numpy(),
        "sector_dispersion_loo": stats["loo_dispersion"].to_numpy(),
        "sector_loo_available_count": stats["loo_count"].to_numpy(),
    })
    sector[_SECTOR_LOO_PROXY] = sector["sector_return_mean_loo"]
    # Sort BEFORE the rolling transform: ``groupby(...).transform`` returns values
    # aligned to the frame it was called on, so assigning them positionally to an
    # unsorted frame would attach each security's rolling momentum to the wrong
    # dates.
    sector = sector.sort_values(["ticker", "date"]).reset_index(drop=True)
    rolled = sector.groupby("ticker", sort=False)[_SECTOR_LOO_PROXY]
    sector["sector_momentum_5_loo"] = rolled.transform(
        lambda s: _rolling_sum(s, SHORT_WINDOW))
    sector["sector_momentum_20_loo"] = rolled.transform(
        lambda s: _rolling_sum(s, LONG_WINDOW))
    sector["sector_realized_vol_20_loo"] = rolled.transform(
        lambda s: _rolling_std(s, LONG_WINDOW))
    sector = sector.drop(columns=[_SECTOR_LOO_PROXY])
    return sector.sort_values(["ticker", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# relative features and cross-sectional ranks
# ---------------------------------------------------------------------------

def build_relative_features(panel: pd.DataFrame, market_loo: pd.DataFrame,
                            sector: pd.DataFrame) -> pd.DataFrame:
    """Stock-minus-market and stock-minus-sector momentum features."""
    frame = panel.merge(market_loo, on=["ticker", "date"], how="left", validate="one_to_one")
    frame = frame.merge(
        sector.loc[:, ["ticker", "date", *SECTOR_FEATURES]], on=["ticker", "date"],
        how="left", validate="one_to_one")

    relative = pd.DataFrame({
        "ticker": frame["ticker"],
        "date": frame["date"],
        "stock_minus_market_return": frame["log_return_1"] - frame["market_return_mean_loo"],
        "stock_minus_sector_return": frame["log_return_1"] - frame["sector_return_mean_loo"],
        "momentum_5_minus_market": frame["log_return_5"] - frame["market_momentum_5_loo"],
        "momentum_20_minus_market": frame["log_return_20"] - frame["market_momentum_20_loo"],
        "momentum_5_minus_sector": frame["log_return_5"] - frame["sector_momentum_5_loo"],
        "momentum_20_minus_sector": frame["log_return_20"] - frame["sector_momentum_20_loo"],
    })
    return relative


def build_cross_sectional_ranks(context_frame: pd.DataFrame) -> pd.DataFrame:
    """Percentile ranks in [0, 1] of each source column WITHIN a date.

    Only stocks with a non-NaN value on that date take part, so a late listing
    cannot dilute the ranking of the stocks that existed.  These are date-``t``
    INPUT features: no ``t+1`` information is involved anywhere.
    """
    out = context_frame.loc[:, ["ticker", "date"]].copy()
    for column in RANK_SOURCE_COLUMNS:
        if column not in context_frame.columns:
            raise KeyError(
                f"cross-sectional rank source '{column}' is not in the context frame"
            )
        out[f"rank_{column}"] = (
            context_frame.groupby("date", sort=False)[column]
            .rank(pct=True, method="average")
            .to_numpy()
        )
    return out


def build_regime_frame(context_frame: pd.DataFrame) -> pd.DataFrame:
    """The regime vector, taken from the ORIGIN date's market context.

    No bull/bear labels are assigned: the regime encoder receives the raw latent
    market state and learns its own representation.
    """
    missing = [c for c in REGIME_FEATURES if c not in context_frame.columns]
    if missing:
        raise KeyError(f"regime vector needs context column(s) {missing}")
    return context_frame.loc[:, ["ticker", "date", *REGIME_FEATURES]].copy()


def assemble_context_features(stock_features: pd.DataFrame, sector_of: dict[str, str],
                              ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build every context column and the rank panel in one call.

    Returns ``(context_frame, market_frame, diagnostics)``.  ``context_frame`` has
    one row per (ticker, date) of the input universe and columns
    ``[ticker, date, *CONTEXT_FEATURES, *RANK_FEATURES, *REGIME_FEATURES]``.
    ``market_frame`` holds the date-level market proxy itself, one row per date.
    """
    panel_columns = ["ticker", "date", *CONTEXT_PANEL_COLUMNS]
    panel = stock_features.loc[:, panel_columns].copy()

    market, market_loo = build_market_context(panel)
    sector = build_sector_context(panel, sector_of)
    relative = build_relative_features(panel, market_loo, sector)

    context = panel.merge(market, on="date", how="left", validate="many_to_one")
    context = context.merge(market_loo, on=["ticker", "date"], how="left",
                            validate="one_to_one")
    context = context.merge(
        sector.loc[:, ["ticker", "date", *SECTOR_FEATURES]], on=["ticker", "date"],
        how="left", validate="one_to_one")
    context = context.merge(relative, on=["ticker", "date"], how="left",
                            validate="one_to_one")
    ranks = build_cross_sectional_ranks(context)
    context = context.merge(ranks, on=["ticker", "date"], how="left", validate="one_to_one")
    context = context.sort_values(["ticker", "date"]).reset_index(drop=True)

    missing = [c for c in (*CONTEXT_FEATURES, *RANK_FEATURES, *REGIME_FEATURES)
               if c not in context.columns]
    if missing:
        raise AssertionError(f"context schema drift: missing {missing}")

    diagnostics = {
        "n_rows": len(context),
        "n_tickers": int(context["ticker"].nunique()),
        "first_date": str(context["date"].min().date()),
        "last_date": str(context["date"].max().date()),
        "market_available_count_min": int(context["market_available_count"].min()),
        "market_available_count_max": int(context["market_available_count"].max()),
        "market_available_count_mean": float(context["market_available_count"].mean()),
        "sector_sizes": sector.groupby("sector", sort=True)["ticker"].nunique().to_dict(),
        "loo_unavailable_rows": {
            "market_return_mean_loo": int(context["market_return_mean_loo"].isna().sum()),
            "sector_return_mean_loo": int(context["sector_return_mean_loo"].isna().sum()),
        },
    }
    return context, market.reset_index(), diagnostics