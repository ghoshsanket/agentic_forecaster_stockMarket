"""Pytest fixtures for the V2 unit tests.

Every V2 fixture is SYNTHETIC.  The V2 test suite never touches the downloaded
market dataset: causality, model, meta-learning, scaling, checkpoint and firewall
behaviour are properties of the code, and proving them on small deterministic
inputs keeps them fast and independent of the data.

The fixture bodies live here (rather than being imported into each test module)
so that pytest discovers them by name and so that test signatures do not shadow
an imported symbol.
"""

from __future__ import annotations

import pandas as pd
import pytest

from agentic_forecaster.v2 import context as ctx
from agentic_forecaster.v2 import features as feat
from agentic_forecaster.v2.dataset import FeatureArrays, SampleTable, build_sample_table
from agentic_forecaster.v2.sectors import SECTOR_MAP_COLUMNS, SectorMap

from .multi_horizon_fixtures import (
    MH_TICKERS,
    make_horizon_arrays,
    make_horizon_targets,
    make_predictions,
    make_sector_map,
    multi_year_dates,
    write_source_tree,
)
from .v2_fixtures import (
    LATE_OFFSET,
    LATE_TICKER,
    SECTORS,
    SEQUENCE_LENGTH,
    TICKERS,
    make_arrays,
    make_model_config,
    make_targets,
    synthetic_bars,
)


@pytest.fixture
def bars() -> pd.DataFrame:
    """One security's daily OHLCV."""
    return synthetic_bars()


@pytest.fixture
def stock_features(bars: pd.DataFrame) -> pd.DataFrame:
    """A single security's causal stock feature frame."""
    frame = feat.build_stock_features(bars)
    frame.insert(0, "ticker", "AAA")
    return frame


@pytest.fixture
def universe_stock_features() -> pd.DataFrame:
    """Six securities, one of which lists late."""
    frames = []
    for index, ticker in enumerate(TICKERS):
        # the late listing simply has no early history at all
        offset = LATE_OFFSET if ticker == LATE_TICKER else 0
        built = feat.build_stock_features(
            synthetic_bars(400 - offset, seed=10 + index))
        built = built.iloc[offset:].reset_index(drop=True)
        built.insert(0, "ticker", ticker)
        frames.append(built)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def sector_map() -> SectorMap:
    """A static sector map built from the official-industry style records."""
    rows = []
    for ticker in TICKERS:
        industry = ("Information Technology" if SECTORS[ticker] == "TECH"
                    else "Financial Services")
        rows.append({
            "ticker": ticker,
            "company_name": f"{ticker} Ltd",
            "industry": industry,
            "broad_sector": SECTORS[ticker],
            "source": "synthetic-test-fixture",
            "source_sha256": "0" * 64,
            "retrieved_at": "2020-01-01T00:00:00+00:00",
        })
    return SectorMap(
        frame=pd.DataFrame(rows, columns=list(SECTOR_MAP_COLUMNS)),
        source="synthetic-test-fixture",
        source_sha256="0" * 64,
        retrieved_at="2020-01-01T00:00:00+00:00",
    )


@pytest.fixture
def context_frame(universe_stock_features: pd.DataFrame,
                  sector_map: SectorMap) -> pd.DataFrame:
    """Per-(security, date) market, sector, relative, rank and regime context."""
    frame, _, _ = ctx.assemble_context_features(universe_stock_features,
                                                sector_map.sector_series())
    return frame


@pytest.fixture
def model_config():
    """A small but structurally complete V2 model configuration."""
    return make_model_config()


@pytest.fixture
def arrays(universe_stock_features: pd.DataFrame, context_frame: pd.DataFrame,
           sector_map: SectorMap) -> FeatureArrays:
    return make_arrays(universe_stock_features, context_frame, sector_map)


#: alias, so a test can take both ``arrays`` and a renamed parameter
@pytest.fixture
def arrays_fixture(arrays: FeatureArrays) -> FeatureArrays:
    return arrays


@pytest.fixture
def targets(context_frame: pd.DataFrame, arrays: FeatureArrays) -> pd.DataFrame:
    return make_targets(context_frame, context_frame, arrays)


@pytest.fixture
def targets_fixture(targets: pd.DataFrame) -> pd.DataFrame:
    return targets


@pytest.fixture
def samples(arrays: FeatureArrays, targets: pd.DataFrame) -> SampleTable:
    return build_sample_table(arrays, targets, sequence_length=SEQUENCE_LENGTH,
                             require_targets=["y_direction", "y_return", "y_rank"])


@pytest.fixture
def mh_source(tmp_path):
    """A synthetic parquet source tree with weekends AND a holiday week."""
    return write_source_tree(tmp_path)


@pytest.fixture
def mh_dates():
    return multi_year_dates()


@pytest.fixture
def mh_sector_map():
    return make_sector_map()


@pytest.fixture
def mh_arrays():
    return make_horizon_arrays()


@pytest.fixture
def mh_targets():
    return make_horizon_targets()


@pytest.fixture
def mh_predictions():
    return make_predictions()


@pytest.fixture
def mh_tickers() -> tuple[str, ...]:
    return MH_TICKERS
