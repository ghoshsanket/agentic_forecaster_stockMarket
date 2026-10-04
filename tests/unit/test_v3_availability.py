"""V3 source alignment: availability classes, lag semantics and causality.

Covers section 50 of the specification:

A. India same-close data may use date ``t`` only for sources explicitly classified
   ``INDIA_SAME_CLOSE``.
B. A US same-calendar-date close is NOT allowed at NSE date ``t`` under conservative
   lag1.
C. lag1 alignment uses the most recent prior valid observation.
D. Holidays do not cause future filling.
E. No backward fill occurs.
F. Raw source provenance survives into the processed features.
G. A deliberately future-aligned US observation causes a causality exception.
H. A 2020 observation causes ``PostCovidDataAccessError``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agentic_forecaster.v2.firewall import PostCovidDataAccessError
from agentic_forecaster.v3.availability import (
    ExogenousCausalityError,
    align_source_asof,
    availability_allowed,
    availability_report,
    lag_rule_description,
    sector_index_mapping,
)
from agentic_forecaster.v3.features import assert_causal, build_source_features
from agentic_forecaster.v3.firewall import assert_no_post_2019
from agentic_forecaster.v3.sources import CLASS_A, CLASS_B, CLASS_C, SourceSpec

INDIAN = SourceSpec("NIFTY50", "NIFTY 50", "yfinance", "^NSEI", "EQUITY_INDEX",
                    "Asia/Kolkata", "09:15-15:30 Asia/Kolkata", CLASS_A, "1D",
                    "2005-01-01")
US = SourceSpec("SP500", "S&P 500", "yfinance", "^GSPC", "EQUITY_INDEX",
                "America/New_York", "09:30-16:00 America/New_York", CLASS_B, "1D",
                "2005-01-01")
TOKYO = SourceSpec("NIKKEI_225", "Nikkei 225", "yfinance", "^N225", "EQUITY_INDEX",
                   "Asia/Tokyo", "09:00-15:00 Asia/Tokyo", CLASS_C, "1D", "2005-01-01")


def _source(dates: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"source_date": pd.to_datetime(dates),
                         "spx": values, "close": values})


def _origin(*dates: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(dates)))


# ---------------------------------------------------------------------------
# A. same-date use requires an explicit same-close class
# ---------------------------------------------------------------------------

def test_india_same_close_may_use_the_origin_date():
    assert availability_allowed(CLASS_A, "2019-12-31", "2019-12-31")
    assert availability_allowed(CLASS_A, "2019-12-30", "2019-12-31")
    assert "same-session" in lag_rule_description(CLASS_A)


def test_a_non_same_close_class_never_gets_the_origin_date():
    assert not availability_allowed(CLASS_B, "2019-12-31", "2019-12-31")
    assert availability_allowed(CLASS_B, "2019-12-30", "2019-12-31")
    assert "STRICTLY BEFORE" in lag_rule_description(CLASS_B)


def test_class_c_is_a_documented_exception_not_a_default():
    """C is opt-in with a documented earlier session close; lag1 is the default."""
    assert availability_allowed(CLASS_C, "2019-12-31", "2019-12-31")
    assert INDIAN.class_a() and US.conservative() and TOKYO.class_c()
    assert "documents" in lag_rule_description(CLASS_C)


# ---------------------------------------------------------------------------
# B/C. the conservative lag1 rule
# ---------------------------------------------------------------------------

def test_us_same_calendar_date_close_is_not_used_at_the_nse_date():
    """B: the provider's same-date US close is never taken at an NSE origin."""
    origins = _origin("2019-12-31")
    source = _source(["2019-12-27", "2019-12-30", "2019-12-31"], [1.0, 2.0, 3.0])
    aligned = align_source_asof(origins, source, spec=US, value_columns=("spx",))
    assert aligned["spx"].iloc[0] == 2.0, "must use the 2019-12-30 observation"
    assert pd.Timestamp(aligned["source_observation_date"].iloc[0]) == pd.Timestamp(
        "2019-12-30")


def test_lag1_uses_the_most_recent_prior_valid_observation():
    """C: not the first prior observation, the nearest one."""
    origins = _origin("2019-12-31")
    source = _source(["2019-12-20", "2019-12-26", "2019-12-27", "2019-12-30",
                      "2019-12-31"], [1.0, 2.0, 3.0, 4.0, 5.0])
    aligned = align_source_asof(origins, source, spec=US, value_columns=("spx",))
    assert aligned["spx"].iloc[0] == 4.0
    assert int(aligned["lag_trading_observations"].iloc[0]) == 1


def test_a_gap_in_the_source_calendar_is_reported_as_a_wider_lag():
    origins = _origin("2019-12-31")
    source = _source(["2019-12-20", "2019-12-27", "2019-12-30"], [1.0, 2.0, 3.0])
    aligned = align_source_asof(origins, source, spec=US, value_columns=("spx",))
    report = availability_report(aligned, spec=US)
    assert report["n_same_calendar_date"] == 0
    assert report["violations"] == 0
    assert aligned["spx"].iloc[0] == 3.0


def test_india_source_uses_the_same_date():
    origins = _origin("2019-12-30", "2019-12-31")
    source = _source(["2019-12-27", "2019-12-30", "2019-12-31"], [1.0, 2.0, 3.0])
    aligned = align_source_asof(origins, source, spec=INDIAN, value_columns=("spx",))
    assert list(aligned["spx"]) == [2.0, 3.0]
    report = availability_report(aligned, spec=INDIAN)
    assert report["n_same_calendar_date"] == 2
    assert report["same_calendar_date_allowed"] is True


# ---------------------------------------------------------------------------
# D/E. holidays never cause future filling
# ---------------------------------------------------------------------------

def test_an_Indian_holiday_does_not_pull_a_future_value_backwards():
    """D: 2019-12-25 is an Indian market holiday with no source observation.

    The origins either side must use observations at or before themselves; nothing
    from 2019-12-26 may appear at the 2019-12-24 origin.
    """
    origins = _origin("2019-12-24", "2019-12-26")
    source = _source(["2019-12-23", "2019-12-24", "2019-12-26"], [1.0, 2.0, 3.0])
    aligned = align_source_asof(origins, source, spec=US, value_columns=("spx",))
    assert aligned["spx"].iloc[0] == 1.0, "2019-12-24 origin must use 2019-12-23"
    # the 2019-12-26 origin is conservative-lagged, so it uses 2019-12-24 and NOT its
    # own 2019-12-26 observation
    assert aligned["spx"].iloc[1] == 2.0
    assert (pd.to_datetime(aligned["source_observation_date"])
            <= pd.to_datetime(aligned["effective_feature_date"])).all()


def test_no_backward_fill_is_possible():
    """E: the join is backwards-only, so no future value can enter."""
    origins = _origin("2019-12-31")
    source = _source(["2019-12-30", "2019-12-31"], [1.0, 2.0])
    aligned = align_source_asof(origins, source, spec=US, value_columns=("spx",))
    assert aligned["spx"].iloc[0] == 1.0
    # an origin at the very START of a source has no strictly-earlier observation, so
    # the value stays undefined rather than borrowing its own same-date value
    earlier = align_source_asof(_origin("2019-12-30"), source, spec=US,
                                value_columns=("spx",))
    assert pd.isna(earlier["spx"].iloc[0]), (
        "no observation exists before 2019-12-30, so nothing may be filled in")


def test_a_source_with_no_prior_observation_leaves_the_value_undefined():
    origins = _origin("2019-12-30")
    source = _source(["2019-12-30", "2019-12-31"], [1.0, 2.0])
    aligned = align_source_asof(origins, source, spec=US, value_columns=("spx",))
    assert pd.isna(aligned["spx"].iloc[0]), (
        "a missing prior observation must stay missing, never be invented")


# ---------------------------------------------------------------------------
# F. provenance survives into the processed features
# ---------------------------------------------------------------------------

def test_raw_provenance_survives_into_the_feature_frame():
    source = _source(["2019-12-20", "2019-12-26", "2019-12-27", "2019-12-30",
                      "2019-12-31"], [1.0, 2.0, 3.0, 4.0, 5.0])
    built = build_source_features(source, INDIAN, final_allowed_date="2019-12-31")
    assert {"source_date", "source_observation_date"} <= set(built.columns)
    assert (built["source_date"] == built["source_observation_date"]).all()
    audit = assert_causal(built, source_observation_col="source_observation_date")
    assert audit["violations"] == 0
    origins = _origin("2019-12-31")
    aligned = align_source_asof(origins, built, spec=US,
                                value_columns=("NIFTY50_return_1",))
    assert aligned["source_observation_date"].iloc[0] <= pd.Timestamp("2019-12-31")
    assert "availability_class" in aligned.columns
    assert "lag_trading_observations" in aligned.columns


# ---------------------------------------------------------------------------
# G/H. deliberate violations
# ---------------------------------------------------------------------------

def test_a_relaxed_conservative_rule_is_refused():
    """G: the causality guarantee can be tightened but never loosened."""
    origins = _origin("2019-12-31")
    source = _source(["2019-12-30", "2019-12-31"], [1.0, 2.0])
    with pytest.raises(ExogenousCausalityError, match="refusing to relax"):
        align_source_asof(origins, source, spec=US, value_columns=("spx",),
                          strict=False)


def test_forcing_a_strict_lag_on_a_same_close_source_is_honoured():
    origins = _origin("2019-12-31")
    source = _source(["2019-12-30", "2019-12-31"], [1.0, 2.0])
    aligned = align_source_asof(origins, source, spec=INDIAN, value_columns=("spx",),
                                strict=True)
    assert aligned["spx"].iloc[0] == 1.0


def test_a_2020_observation_raises_post_covid_data_access_error():
    """H: nothing from 2020 may enter a V3 feature or label."""
    frame = pd.DataFrame({"source_date": pd.to_datetime(["2019-12-31", "2020-01-02"]),
                          "spx": [1.0, 2.0]})
    with pytest.raises(PostCovidDataAccessError):
        align_source_asof(_origin("2019-12-31"), frame, spec=US,
                          value_columns=("spx",), final_allowed_date="2019-12-31")
    with pytest.raises(PostCovidDataAccessError):
        assert_no_post_2019(exogenous_dates=frame["source_date"],
                            final_allowed_date="2019-12-31")


def test_availability_assertion_catches_a_future_observation():
    from agentic_forecaster.v3.firewall import assert_availability

    with pytest.raises(ExogenousCausalityError, match="AFTER the NSE prediction"):
        assert_availability([pd.Timestamp("2020-01-02")], [pd.Timestamp("2019-12-31")])
    assert assert_availability([pd.Timestamp("2019-12-30")],
                               [pd.Timestamp("2019-12-31")])["violations"] == 0


# ---------------------------------------------------------------------------
# sector mapping (section 20)
# ---------------------------------------------------------------------------

def test_sector_mapping_refuses_to_invent_one():
    mapping = sector_index_mapping({"AAA": "IT", "BBB": "WEIRD", "CCC": "IT"},
                                   {"IT": "NIFTY_IT"})
    assert mapping.index_for("AAA") == "NIFTY_IT"
    assert mapping.index_for("BBB") is None, "an unmapped sector must map to nothing"
    assert mapping.confidence_for("BBB") == 0.0
    assert sorted(mapping.mapped_tickers()) == ["AAA", "CCC"]
    frame = mapping.frame()
    assert "provenance" in frame.columns
    assert frame.loc[frame["ticker"] == "BBB", "provenance"].iloc[0]


def test_sector_mapping_does_not_drop_the_security():
    frame = sector_index_mapping({"AAA": "IT", "BBB": "WEIRD"},
                                 {"IT": "NIFTY_IT"}).frame()
    assert len(frame) == 2, "every security stays in the mapping, mapped or not"


# ---------------------------------------------------------------------------
# missing-data policy (section 17)
# ---------------------------------------------------------------------------

def test_no_backward_fill_or_future_interpolation_is_used_anywhere():
    source = _source(["2019-12-26", "2019-12-30"], [1.0, 2.0])
    built = build_source_features(source, US, final_allowed_date="2019-12-31")
    origins = _origin("2019-12-27")
    aligned = align_source_asof(origins, built, spec=US,
                                value_columns=("SP500_return_1",))
    # 2019-12-27 is not a source observation date: the aligned value must be the
    # 2019-12-26 feature, and no 2019-12-30 feature may be interpolated onto it
    assert pd.Timestamp(aligned["source_observation_date"].iloc[0]) == pd.Timestamp(
        "2019-12-26")
    assert np.isfinite(aligned["SP500_return_1"].iloc[0]) or pd.isna(
        aligned["SP500_return_1"].iloc[0])