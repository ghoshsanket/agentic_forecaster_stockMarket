"""Deterministic synthetic fixtures for the multi-horizon tests.

Every fixture is SYNTHETIC: the multi-horizon unit tests never touch the downloaded
market dataset.  The calendar deliberately contains weekends AND a fabricated
five-day exchange holiday, because the whole point of the horizon definition is
that a horizon counts TRADING OBSERVATIONS and never calendar days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agentic_forecaster.v2 import features as feat
from agentic_forecaster.v2.dataset import FeatureArrays, build_feature_arrays
from agentic_forecaster.v2.sectors import SECTOR_MAP_COLUMNS, SectorMap

#: Fixture securities.
MH_TICKERS: tuple[str, ...] = ("AAA", "BBB", "CCC")

#: The fabricated exchange holiday: a full week inside the fixture calendar.
HOLIDAY_WEEK = pd.bdate_range("2018-03-05", periods=5)

#: Per-security log drift, so the three futures are not identical.
DRIFTS = {"AAA": 0.0008, "BBB": -0.0004, "CCC": 0.0}


def holiday_aware_business_days(start: str, n: int) -> pd.DatetimeIndex:
    """Business days with the fabricated holiday week removed."""
    return pd.bdate_range(start, periods=n + 20).difference(HOLIDAY_WEEK)[:n]


def synthetic_closes(dates: pd.DatetimeIndex, ticker: str, *, seed: int | None = None
                     ) -> np.ndarray:
    rng = np.random.default_rng(seed if seed is not None
                                else abs(hash(ticker)) % 9973)
    return 100 * np.exp(np.cumsum(DRIFTS[ticker] + rng.normal(0, 0.01, len(dates))))


def bars_frame(dates: pd.DatetimeIndex, closes: np.ndarray,
               *, seed: int = 3) -> pd.DataFrame:
    """One security's OHLCV bars.

    Volume VARIES on purpose: a constant volume series makes ``volume_z_20``
    degenerate (zero standard deviation), and every sample would then be dropped
    for a non-finite input -- which would make the horizon tests vacuous.
    """
    rng = np.random.default_rng(seed)
    volume = 1000.0 * (1.0 + 0.3 * rng.standard_normal(len(dates)).cumsum() * 0.05)
    return pd.DataFrame({"Date": dates, "Open": closes,
                         "High": closes * (1.0 + 0.01 * rng.random(len(dates))),
                         "Low": closes * (1.0 - 0.01 * rng.random(len(dates))),
                         "Close": closes, "Volume": np.abs(volume)})


def write_source_tree(root, tickers=MH_TICKERS, *, start: str = "2018-01-01",
                      n_days: int = 400, variant: str = "adjusted") -> dict:
    """Write a parquet source tree and return the calendar it used."""
    dates = holiday_aware_business_days(start, n_days)
    directory = root / variant / "parquet"
    directory.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        bars_frame(dates, synthetic_closes(dates, ticker)).to_parquet(
            directory / f"{ticker}.parquet", index=False)
    return {"root": root, "dates": dates, "tickers": list(tickers), "variant": variant}


def make_sector_map(tickers=MH_TICKERS) -> SectorMap:
    rows = [{
        "ticker": ticker,
        "company_name": f"{ticker} Ltd",
        "industry": f"Industry {index}",
        "broad_sector": f"S{index}",
        "source": "synthetic-multi-horizon-fixture",
        "source_sha256": "0" * 64,
        "retrieved_at": "2020-01-01T00:00:00+00:00",
    } for index, ticker in enumerate(tickers, start=1)]
    return SectorMap(
        frame=pd.DataFrame(rows, columns=list(SECTOR_MAP_COLUMNS)),
        source="synthetic-multi-horizon-fixture",
        source_sha256="0" * 64,
        retrieved_at="2020-01-01T00:00:00+00:00",
    )


def multi_year_dates(start: str = "2015-01-01", end: str = "2019-12-31") -> pd.DatetimeIndex:
    return pd.bdate_range(start, end).difference(HOLIDAY_WEEK)


def make_horizon_arrays(dates: pd.DatetimeIndex | None = None,
                        tickers=MH_TICKERS) -> FeatureArrays:
    """Stock-only feature arrays (no context) over the fixture calendar."""
    dates = multi_year_dates() if dates is None else dates
    frames = []
    for ticker in tickers:
        built = feat.build_stock_features(
            bars_frame(dates, synthetic_closes(dates, ticker)))
        built.insert(0, "ticker", ticker)
        frames.append(built)
    stock = pd.concat(frames, ignore_index=True)
    return build_feature_arrays(stock, stock.assign(), make_sector_map(tickers),
                                tickers=list(tickers), use_context=False)


def make_horizon_targets(dates: pd.DatetimeIndex | None = None, horizons=(1, 3, 5, 10),
                         tickers=MH_TICKERS, *,
                         final_allowed_date: str = "2019-12-31") -> pd.DataFrame:
    """The multi-horizon target frame for the fixture calendar.

    Built exactly the way :func:`agentic_forecaster.v2.horizons.
    build_horizon_target_frame` builds it, so the tests exercise the same
    semantics: a positional shift along each security's own trading rows, then a
    physical cap at ``final_allowed_date``.
    """
    dates = multi_year_dates() if dates is None else dates
    wide = pd.concat([
        pd.DataFrame({"ticker": ticker, "origin_date": dates,
                      "close": synthetic_closes(dates, ticker)})
        for ticker in tickers
    ], ignore_index=True)
    boundary = pd.Timestamp(final_allowed_date)
    blocks = []
    for horizon in horizons:
        frame = wide.copy()
        frame["target_end_date"] = frame.groupby("ticker")["origin_date"].shift(-horizon)
        frame["close_t_plus_h"] = frame.groupby("ticker")["close"].shift(-horizon)
        frame = frame.dropna(subset=["target_end_date", "close_t_plus_h"])
        frame = frame.loc[(frame["target_end_date"] <= boundary)
                          & (frame["origin_date"] <= boundary)]
        blocks.append(pd.DataFrame({
            "ticker": frame["ticker"],
            "origin_date": frame["origin_date"],
            "target_end_date": frame["target_end_date"],
            "horizon": int(horizon),
            "future_log_return": np.log(frame["close_t_plus_h"] / frame["close"]),
            "y": (frame["close_t_plus_h"] > frame["close"]).astype(int),
            "close_t": frame["close"],
            "close_t_plus_h": frame["close_t_plus_h"],
        }))
    return pd.concat(blocks, ignore_index=True).reset_index(drop=True)


def make_store_style_one_day_target(dates: pd.DatetimeIndex | None = None,
                                    tickers=MH_TICKERS) -> pd.DataFrame:
    """Recompute the EXISTING store one-day target frame with the store's logic.

    Mirrors ``v2.store.build_target_frame``: a wide close frame, ``shift(-1)`` along
    the union calendar, a drop of every incomplete row, then
    ``y_direction = int(close[t+1] > close[t])``.
    """
    dates = multi_year_dates() if dates is None else dates
    closes = {ticker: pd.Series(synthetic_closes(dates, ticker), index=dates)
              for ticker in tickers}
    wide = pd.DataFrame(closes).sort_index()
    blocks = []
    for ticker, series in wide.items():
        following = series.shift(-1)
        following_date = pd.Series(series.index, index=series.index).shift(-1)
        frame = pd.DataFrame({"ticker": ticker, "origin_date": series.index,
                              "target_date": following_date.to_numpy(),
                              "close_t": series.to_numpy(),
                              "close_t_plus_1": following.to_numpy()})
        blocks.append(frame.dropna(subset=["close_t", "close_t_plus_1", "target_date"]))
    targets = pd.concat(blocks, ignore_index=True)
    targets["y_direction"] = (targets["close_t_plus_1"] > targets["close_t"]).astype(int)
    targets["raw_next_return"] = np.log(targets["close_t_plus_1"] / targets["close_t"])
    return targets


def make_predictions(n: int = 400, *, seed: int = 11, tickers=MH_TICKERS,
                    positive_rate: float = 0.5, dates: pd.DatetimeIndex | None = None,
                    future_scale: float = 0.02) -> pd.DataFrame:
    """A validation prediction frame with the columns the metrics expect."""
    rng = np.random.default_rng(seed)
    dates = multi_year_dates("2018-01-01") if dates is None else dates
    ticker_values = np.array(list(tickers) * (n // len(tickers) + 1))[:n]
    return pd.DataFrame({
        "ticker": ticker_values,
        "origin_date": np.resize(dates.to_numpy(), n),
        "target_date": np.resize(dates.to_numpy(), n),
        "y_true": (rng.random(n) < positive_rate).astype(int),
        "p_up": np.clip(rng.beta(2.0, 2.0, n), 0.01, 0.99),
        "future_log_return": rng.normal(0.0, future_scale, n),
    })