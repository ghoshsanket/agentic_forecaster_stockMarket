"""Multi-horizon target semantics: trading-observation horizons, the exact 1D control
identity, and the absolute 2019-12-31 boundary.

Covers sections 41A-41J of the specification:

A. ``ABS_DIR_1D_CONTROL`` equals the existing one-day target EXACTLY
B. H=3 uses the third FUTURE TRADING observation
C. H=5 uses the fifth future trading observation
D. H=10 uses the tenth future trading observation
E. weekends / exchange holidays do not count as horizon steps
F. no future row appears inside an input sequence
G. a training target cannot cross into the validation year
H. a validation target cannot cross into the next year
I. a December-2019 origin whose target ends in 2020 is rejected
J. no 2020 row can enter any horizon target
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agentic_forecaster.v2 import features as feat
from agentic_forecaster.v2 import horizons as HZ
from agentic_forecaster.v2.dataset import SplitWindow, V2SequenceDataset
from agentic_forecaster.v2.firewall import PostCovidDataAccessError
from agentic_forecaster.v2.horizon_dataset import (
    SAMPLE_COLUMNS,
    build_horizon_sample_table,
    horizon_split,
)
from agentic_forecaster.v2.sectors import SectorMap

from .multi_horizon_fixtures import (
    HOLIDAY_WEEK,
    bars_frame,
    holiday_aware_business_days,
    make_horizon_arrays,
    make_predictions,
    make_store_style_one_day_target,
    multi_year_dates,
    synthetic_closes,
)

HORIZONS = HZ.HORIZONS


def _build(source, horizons=HORIZONS, final_allowed_date="2019-12-31",
           tickers_override=None):
    return HZ.build_horizon_target_frame(
        source["root"], variant=source["variant"],
        tickers=tickers_override or source["tickers"],
        horizons=horizons, final_allowed_date=final_allowed_date)


# ---------------------------------------------------------------------------
# B/C/D/E. the horizon is a trading-observation offset
# ---------------------------------------------------------------------------

def test_horizon_targets_use_future_trading_observations(mh_source):
    targets = _build(mh_source)
    dates = mh_source["dates"]
    closes = {}
    for ticker in mh_source["tickers"]:
        normalised = feat.normalise_ohlcv_frame(pd.read_parquet(
            mh_source["root"] / mh_source["variant"] / "parquet" / f"{ticker}.parquet"))
        closes[ticker] = pd.Series(normalised["close"].to_numpy(),
                                   index=pd.DatetimeIndex(normalised["date"]))
    assert not targets.empty
    for horizon in HORIZONS:
        block = targets.loc[targets["horizon"] == horizon]
        assert len(block) > 0
        assert set(block["ticker"]) == set(mh_source["tickers"])
        sample = block.sample(min(40, len(block)), random_state=1)
        for row in sample.itertuples():
            series = closes[row.ticker]
            origin = pd.Timestamp(row.origin_date)
            expected_date = dates[dates.get_loc(origin) + horizon]
            expected_return = np.log(series.loc[expected_date] / series.loc[origin])
            assert pd.Timestamp(row.target_end_date) == expected_date
            assert row.future_log_return == pytest.approx(expected_return, abs=1e-12)
            assert row.y == int(expected_return > 0)
            assert row.horizon == horizon


@pytest.mark.parametrize("horizon", [3, 5, 10])
def test_specific_horizon_offsets(mh_source, horizon):
    """B, C, D: the H-th FUTURE observation, counted in trading rows."""
    targets = _build(mh_source, horizons=(horizon,), tickers_override=["AAA"])
    dates = mh_source["dates"]
    block = targets.loc[targets["horizon"] == horizon].reset_index(drop=True)
    for offset in (0, 1, 17, 55, 120):
        origin = dates[offset]
        expected = dates[offset + horizon]
        row = block.loc[block["origin_date"] == origin]
        assert len(row) == 1
        assert pd.Timestamp(row.iloc[0]["target_end_date"]) == expected
        assert expected > origin


def test_weekends_and_holidays_are_not_horizon_steps(mh_source):
    """E: a weekend or a holiday never counts as one trading step."""
    targets = _build(mh_source, horizons=(3, 5))
    dates = mh_source["dates"]
    for horizon in (3, 5):
        block = targets.loc[targets["horizon"] == horizon]
        for row in block.itertuples():
            start = dates.get_loc(pd.Timestamp(row.origin_date))
            assert pd.Timestamp(row.target_end_date) == dates[start + horizon]
    # the fabricated holiday week really is absent from the calendar
    assert not set(HOLIDAY_WEEK) & set(dates)
    # EXACTLY H-1 trading observations lie strictly between origin and target
    for horizon in (3, 5):
        block = targets.loc[targets["horizon"] == horizon]
        between = [
            dates.get_loc(pd.Timestamp(row.target_end_date))
            - dates.get_loc(pd.Timestamp(row.origin_date)) - 1
            for row in block.itertuples()
        ]
        assert set(between) == {horizon - 1}
    # a Monday origin reaches its 3-day target on the THURSDAY: three calendar days
    # for three trading steps, which a timedelta(days=3) reading would misread
    monday = next(d for d in dates if d.dayofweek == 0)
    row = targets.loc[(targets["horizon"] == 3)
                      & (targets["origin_date"] == monday)].iloc[0]
    assert (pd.Timestamp(row.target_end_date) - monday).days == 3
    assert pd.Timestamp(row.target_end_date).dayofweek == 3


def test_calendar_timedelta_is_never_used(mh_source):
    """The definition is positional: +timedelta(days=H) would disagree somewhere."""
    targets = _build(mh_source, horizons=(5,))
    dates = mh_source["dates"]
    differing = 0
    for row in targets.loc[targets["horizon"] == 5].itertuples():
        origin = pd.Timestamp(row.origin_date)
        positional = dates[dates.get_loc(origin) + 5]
        if positional != origin + pd.Timedelta(days=5):
            differing += 1
    assert differing > 0, ("the fixture must contain a case where the positional and "
                           "calendar definitions disagree, or this proves nothing")


# ---------------------------------------------------------------------------
# A. the 1D control reproduces the existing one-day target
# ---------------------------------------------------------------------------

def test_control_horizon_reproduces_existing_one_day_target(mh_source):
    """A: ABS_DIR_1D_CONTROL == y_direction on every shared key."""
    existing = make_store_style_one_day_target(mh_source["dates"],
                                               mh_source["tickers"])
    result = HZ.assert_control_matches_store(_build(mh_source), existing, horizon=1)
    assert result["identical"] is True
    assert result["label_mismatches"] == 0
    assert result["target_date_mismatches"] == 0
    assert result["return_mismatches"] == 0
    assert result["n_compared_keys"] > 100


def test_control_horizon_identity_is_enforced_not_assumed(mh_source):
    """A: the check must FAIL on a perturbed label or return."""
    existing = make_store_style_one_day_target(mh_source["dates"],
                                               mh_source["tickers"])
    targets = _build(mh_source)

    flipped = existing.copy()
    flipped.loc[flipped.index[7], "y_direction"] = 1 - int(
        flipped.loc[flipped.index[7], "y_direction"])
    with pytest.raises(AssertionError, match="does not reproduce"):
        HZ.assert_control_matches_store(targets, flipped, horizon=1)

    shifted = existing.copy()
    shifted.loc[shifted.index[9], "target_date"] = (
        pd.Timestamp(shifted.loc[shifted.index[9], "target_date"])
        + pd.Timedelta(days=1))
    with pytest.raises(AssertionError, match="does not reproduce"):
        HZ.assert_control_matches_store(targets, shifted, horizon=1)


def test_control_objective_id_is_explicit():
    assert HZ.objective_id(1) == "ABS_DIR_1D_CONTROL"
    assert HZ.objective_id(3) == "ABS_DIR_3D"
    assert HZ.objective_id(5) == "ABS_DIR_5D"
    assert HZ.objective_id(10) == "ABS_DIR_10D"
    with pytest.raises(ValueError, match="not part of this experiment"):
        HZ.objective_id(7)


# ---------------------------------------------------------------------------
# I/J. the absolute 2019 boundary
# ---------------------------------------------------------------------------

def test_december_2019_origin_whose_target_reaches_2020_is_dropped(tmp_path):
    """I: no 2020 close may finish a 2019 label."""
    from .multi_horizon_fixtures import write_source_tree

    dates = holiday_aware_business_days("2019-11-01", 60)
    source = write_source_tree(tmp_path, ("AAA",), start="2019-11-01", n_days=60)
    assert dates[-1] > pd.Timestamp("2019-12-31")
    targets = _build(source, horizons=(10,), final_allowed_date="2019-12-31")
    assert targets["target_end_date"].max() <= pd.Timestamp("2019-12-31")
    assert targets.loc[targets["origin_date"] >= pd.Timestamp("2019-12-20")].empty


def test_a_post_boundary_target_row_is_rejected_at_access_time():
    """I/J: a frame carrying a 2020 target is REFUSED, not quietly filtered."""
    arrays = make_horizon_arrays(pd.DatetimeIndex(["2019-01-02"]))
    targets = pd.DataFrame([{
        "ticker": "AAA", "origin_date": pd.Timestamp("2019-12-24"),
        "target_end_date": pd.Timestamp("2020-01-03"), "horizon": 10,
        "future_log_return": 0.02, "y": 1, "close_t": 100.0, "close_t_plus_h": 102.0,
    }])
    with pytest.raises(PostCovidDataAccessError):
        build_horizon_sample_table(arrays, targets, horizon=10, sequence_length=4,
                                   final_allowed_date="2019-12-31")


def test_no_2020_row_can_enter_a_horizon_target(mh_source):
    """J: the cache can never hold a 2020 target, at any cap."""
    targets = _build(mh_source)
    assert targets["target_end_date"].max() < pd.Timestamp("2020-01-01")
    # a tighter cap removes the offending rows rather than keeping them
    capped = _build(mh_source, horizons=(3,), final_allowed_date="2018-06-30")
    assert capped["target_end_date"].max() <= pd.Timestamp("2018-06-30")
    assert len(capped) < len(targets.loc[targets["horizon"] == 3])
    # and a horizon of ZERO or negative is refused outright
    with pytest.raises(ValueError, match="horizons must be >= 1"):
        _build(mh_source, horizons=(0,))


def test_loading_a_polluted_target_cache_is_refused(tmp_path):
    """J: even a hand-polluted cache is refused on load."""
    root = tmp_path / "horizon_targets"
    root.mkdir(parents=True)
    polluted = pd.DataFrame([{
        "ticker": "AAA", "origin_date": pd.Timestamp("2019-12-30"),
        "target_end_date": pd.Timestamp("2020-01-10"), "horizon": 5,
        "future_log_return": 0.01, "y": 1, "close_t": 1.0, "close_t_plus_h": 1.01,
    }])
    polluted.to_parquet(root / "targets.parquet", index=False)
    (root / "metadata.json").write_text('{"target_schema_sha256": "x"}')
    with pytest.raises(PostCovidDataAccessError):
        HZ.load_target_cache(root, final_allowed_date="2019-12-31")


# ---------------------------------------------------------------------------
# F. no future row inside an input sequence
# ---------------------------------------------------------------------------

def test_input_sequence_ends_at_the_origin(mh_arrays, mh_targets):
    """F: the input window contains t-(T-1) .. t and never t+1."""
    samples = build_horizon_sample_table(mh_arrays, mh_targets, horizon=10,
                                         sequence_length=20,
                                         final_allowed_date="2019-12-31")
    frame = samples.frame.iloc[len(samples.frame) // 2:].reset_index(drop=True)
    dataset = V2SequenceDataset(samples, frame)
    sequence = dataset[0]["stock_sequence"].numpy()
    assert sequence.shape == (20, mh_arrays.n_stock_features)

    row = frame.iloc[0]
    matrices = mh_arrays.matrices[str(row["ticker"])]
    index = int(row["row"])
    window_dates = pd.DatetimeIndex(matrices.dates[index - 19: index + 1])
    assert window_dates.max() == pd.Timestamp(matrices.dates[index])
    assert window_dates.max() == pd.Timestamp(row["origin_date"])
    assert window_dates.max() < pd.Timestamp(row["target_date"])
    # the whole window is inside the sample's own security and strictly historical
    assert len(window_dates) == 20
    assert window_dates.is_monotonic_increasing


def test_sequence_length_is_the_existing_v2_value(mh_arrays, mh_targets):
    samples = build_horizon_sample_table(mh_arrays, mh_targets, horizon=5,
                                         sequence_length=60,
                                         final_allowed_date="2019-12-31")
    assert samples.sequence_length == 60
    frame = samples.frame.iloc[10:11].reset_index(drop=True)
    dataset = V2SequenceDataset(samples, frame)
    assert dataset[0]["stock_sequence"].shape[0] == 60


# ---------------------------------------------------------------------------
# G/H. split boundary safety for multi-day targets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("horizon", [1, 3, 5, 10])
def test_validation_target_cannot_cross_into_the_next_year(mh_arrays, mh_targets,
                                                           horizon):
    """H: origin AND target_end must both lie inside the validation year."""
    samples = build_horizon_sample_table(mh_arrays, mh_targets, horizon=horizon,
                                         sequence_length=20,
                                         final_allowed_date="2019-12-31")
    window = SplitWindow("MH_DEV_2018", "2016-01-01", "2017-12-31",
                         "2018-01-01", "2018-12-31")
    split = horizon_split(samples, window, fold=window.name)
    block = samples.frame.loc[split.val]
    assert not block.empty
    origin = pd.to_datetime(block["origin_date"])
    target = pd.to_datetime(block["target_date"])
    assert (origin >= pd.Timestamp("2018-01-01")).all()
    assert (origin <= pd.Timestamp("2018-12-31")).all()
    assert (target <= pd.Timestamp("2018-12-31")).all()
    assert (target >= pd.Timestamp("2018-01-01")).all()


@pytest.mark.parametrize("horizon", [1, 3, 5, 10])
def test_training_target_cannot_cross_into_the_validation_year(mh_arrays, mh_targets,
                                                               horizon):
    """G: a December training origin whose target enters validation goes."""
    samples = build_horizon_sample_table(mh_arrays, mh_targets, horizon=horizon,
                                         sequence_length=20,
                                         final_allowed_date="2019-12-31")
    window = SplitWindow("MH_DEV_2018", "2015-01-01", "2017-12-31",
                         "2018-01-01", "2018-12-31")
    split = horizon_split(samples, window, fold=window.name)
    train = samples.frame.loc[split.train]
    assert not train.empty
    assert (pd.to_datetime(train["target_date"]) <= pd.Timestamp("2017-12-31")).all()
    # a December-2017 training origin whose target would land in 2018 is GONE
    crossing = samples.frame.loc[
        (pd.to_datetime(samples.frame["origin_date"]) >= pd.Timestamp("2017-12-15"))
        & (pd.to_datetime(samples.frame["origin_date"]) <= pd.Timestamp("2017-12-31"))
        & (pd.to_datetime(samples.frame["target_date"]) > pd.Timestamp("2017-12-31"))]
    # such rows may exist in the table (they are legitimate samples for a LATER
    # window), but NONE of them may be a training row of this fold
    assert not train.index.isin(crossing.index).any(), (
        "a training origin whose target falls into the validation year must be "
        "dropped from training")
    if horizon > 1:
        december = train.loc[pd.to_datetime(train["origin_date"])
                             >= pd.Timestamp("2017-12-01")]
        assert (pd.to_datetime(december["target_date"])
                <= pd.Timestamp("2017-12-31")).all()


def test_the_specified_december_2018_example_is_excluded(mh_arrays, mh_targets):
    """H: an origin of 2018-12-24 with a 10-day target in 2019 is not 2018."""
    samples = build_horizon_sample_table(mh_arrays, mh_targets, horizon=10,
                                         sequence_length=20,
                                         final_allowed_date="2019-12-31")
    window = SplitWindow("MH_DEV_2018", "2005-01-01", "2017-12-31",
                         "2018-01-01", "2018-12-31")
    split = horizon_split(samples, window, fold=window.name)
    validation = samples.frame.loc[split.val]
    assert (pd.to_datetime(validation["target_date"])
            <= pd.Timestamp("2018-12-31")).all()
    crossing = validation.loc[pd.to_datetime(validation["origin_date"])
                              .ge(pd.Timestamp("2018-12-20"))]
    assert crossing.empty or (pd.to_datetime(crossing["target_date"])
                               <= pd.Timestamp("2018-12-31")).all()


def test_boundary_report_proves_both_dates_per_split(mh_arrays, mh_targets):
    from agentic_forecaster.v2.horizon_dataset import boundary_report

    window = SplitWindow("MH_DEV_2018", "2015-01-01", "2017-12-31",
                         "2018-01-01", "2018-12-31")
    for horizon in (1, 10):
        samples = build_horizon_sample_table(mh_arrays, mh_targets, horizon=horizon,
                                             sequence_length=20,
                                             final_allowed_date="2019-12-31")
        report = boundary_report(samples, window, horizon=horizon)
        assert report["train_samples"] > 0
        assert report["validation_samples"] > 0
        assert report["train_max_target_end"] <= "2018-12-31"
        assert report["validation_max_target_end"] <= "2018-12-31"


# ---------------------------------------------------------------------------
# semantic wording
# ---------------------------------------------------------------------------

def test_multi_day_results_are_never_called_next_day():
    assert HZ.horizon_phrase(1) == "1-trading-day directional accuracy"
    for horizon in (3, 5, 10):
        phrase = HZ.horizon_phrase(horizon)
        assert phrase == f"{horizon}-trading-day directional accuracy"
        assert "next-day" not in phrase
        assert "day" not in phrase.replace(f"{horizon}-trading-day", "")


def test_sample_diagnostics_declare_the_objective(mh_arrays, mh_targets):
    samples = build_horizon_sample_table(mh_arrays, mh_targets, horizon=5,
                                         sequence_length=20,
                                         final_allowed_date="2019-12-31")
    assert samples.diagnostics["objective_id"] == "ABS_DIR_5D"
    assert samples.diagnostics["horizon_phrase"] == HZ.horizon_phrase(5)
    assert samples.diagnostics["max_target_date_consumed"] <= "2019-12-31"


def test_sample_frame_declares_the_expected_columns():
    assert SAMPLE_COLUMNS[:7] == ("ticker", "origin_date", "target_date", "row",
                                 "ticker_id", "sector_id", "y_direction")
    assert "future_log_return" in SAMPLE_COLUMNS
    assert "horizon" in SAMPLE_COLUMNS


def test_predictions_fixture_shape_is_usable():
    predictions = make_predictions()
    assert set(predictions.columns) >= {"ticker", "origin_date", "y_true", "p_up"}
    assert len(predictions) == 400
    assert isinstance(SectorMap, type)
    assert multi_year_dates().max() <= pd.Timestamp("2019-12-31")
    assert bars_frame(multi_year_dates()[:3], np.array([1.0, 2.0, 3.0])) is not None
    assert synthetic_closes(multi_year_dates()[:5], "AAA").shape == (5,)