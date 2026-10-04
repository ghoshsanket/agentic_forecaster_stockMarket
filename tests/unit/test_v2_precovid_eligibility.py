"""PRE-COVID eligible-supervised-universe derivation, on synthetic histories.

Covers each exclusion reason with a purpose-built synthetic ticker:

* sufficient history                      -> included
* late listing                            -> excluded
* insufficient 2017 observations          -> excluded
* insufficient 2018 observations          -> excluded
* insufficient 2019 observations          -> excluded
* no causal features at all              -> excluded

and proves that the eligibility calculation never consults a date after
2019-12-31, that the frozen list is deterministic, and that the persisted CSV and
YAML carry exactly the columns the specification requires.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_forecaster.v2 import precovid as pc
from agentic_forecaster.v2.firewall import PostCovidDataAccessError

REPO = Path(__file__).resolve().parents[2]


def _samples(ticker: str, *, start: str, periods: int, per_year: int | None = None
             ) -> pd.DataFrame:
    """A synthetic supervised sample frame for one security.

    ``per_year`` limits how many samples fall in each calendar year, which is how
    the "insufficient observations in year X" cases are constructed.
    """
    dates = pd.bdate_range(start, periods=periods)
    if per_year is None:
        per_year = 240
    rows = []
    counters: dict[int, int] = {}
    for date in dates:
        year = date.year
        if counters.get(year, 0) >= per_year:
            continue
        counters[year] = counters.get(year, 0) + 1
        rows.append({
            "ticker": ticker,
            "origin_date": date,
            "target_date": pd.bdate_range(date, periods=2)[-1],
            "row": len(rows),
            "ticker_id": 1,
            "sector_id": 1,
            "y_direction": 1,
            "y_return": 0.1,
            "y_rank": 0.5,
        })
    return pd.DataFrame(rows)


def _universe(*frames: pd.DataFrame) -> pd.DataFrame:
    samples = pd.concat(frames, ignore_index=True)
    raw = {t: str(samples.loc[samples["ticker"] == t, "origin_date"].min().date())
           for t in samples["ticker"].unique()}
    sector = {t: "SYNTH" for t in samples["ticker"].unique()}
    return pc.derive_supervised_universe(samples, raw_first_dates=raw,
                                        sector_of=sector,
                                        candidates=sorted(samples["ticker"].unique()))


# ---------------------------------------------------------------------------
# inclusion / exclusion
# ---------------------------------------------------------------------------

def test_sufficient_history_is_included():
    frame = _universe(_samples("GOOD", start="2005-01-03", periods=3900))
    row = frame.iloc[0]
    assert bool(row["eligible"]) is True
    assert row["exclusion_reason"] == ""
    assert row["train_samples_through_2016"] >= pc.MIN_TRAIN_SAMPLES
    assert row["samples_2017"] >= pc.MIN_YEAR_SAMPLES
    assert row["samples_2018"] >= pc.MIN_YEAR_SAMPLES
    assert row["samples_2019"] >= pc.MIN_YEAR_SAMPLES


def test_late_listing_is_excluded():
    frame = _universe(_samples("LATE", start="2015-01-01", periods=1200))
    row = frame.iloc[0]
    assert bool(row["eligible"]) is False
    assert "A/B" in row["exclusion_reason"]


def test_insufficient_train_samples_is_excluded():
    frame = _universe(_samples("SHORT", start="2016-06-01", periods=250))
    row = frame.iloc[0]
    assert bool(row["eligible"]) is False
    assert "usable TRAIN samples through 2016-12-31" in row["exclusion_reason"]


def test_insufficient_2017_observations_is_excluded():
    # 2017 is capped below the threshold while every other year is fine
    samples = _samples("NO2017", start="2005-01-03", periods=3000)
    keep = ~((samples["origin_date"].dt.year == 2017) &
             (samples.groupby(samples["origin_date"].dt.year).cumcount() > 100))
    frame = _universe(samples.loc[keep])
    row = frame.iloc[0]
    assert bool(row["eligible"]) is False
    assert "C:" in row["exclusion_reason"]
    assert row["samples_2017"] < pc.MIN_YEAR_SAMPLES


def test_insufficient_2018_observations_is_excluded():
    samples = _samples("NO2018", start="2005-01-03", periods=3000)
    keep = ~((samples["origin_date"].dt.year == 2018) &
             (samples.groupby(samples["origin_date"].dt.year).cumcount() > 100))
    frame = _universe(samples.loc[keep])
    row = frame.iloc[0]
    assert bool(row["eligible"]) is False
    assert "D:" in row["exclusion_reason"]
    assert row["samples_2018"] < pc.MIN_YEAR_SAMPLES


def test_insufficient_2019_observations_is_excluded():
    samples = _samples("NO2019", start="2005-01-03", periods=3900)
    keep = ~((samples["origin_date"].dt.year == 2019) &
             (samples.groupby(samples["origin_date"].dt.year).cumcount() > 100))
    frame = _universe(samples.loc[keep])
    row = frame.iloc[0]
    assert bool(row["eligible"]) is False
    assert "E:" in row["exclusion_reason"]
    assert row["samples_2019"] < pc.MIN_YEAR_SAMPLES


def test_no_usable_sample_at_all_is_reported_not_dropped():
    samples = _samples("REAL", start="2005-01-03", periods=1200)
    frame = pc.derive_supervised_universe(
        samples, raw_first_dates={"REAL": "2005-01-03", "GHOST": "2019-06-01"},
        sector_of={"REAL": "SYNTH", "GHOST": "UNKNOWN"},
        candidates=["GHOST", "REAL"])
    ghost = frame.loc[frame["ticker"] == "GHOST"].iloc[0]
    assert bool(ghost["eligible"]) is False
    assert pd.isna(ghost["first_usable_sample"])
    assert "no usable supervised sample" in ghost["exclusion_reason"]
    assert ghost["sector"] == "UNKNOWN"
    assert set(frame["ticker"]) == {"GHOST", "REAL"}


def test_thresholds_are_honoured_exactly_at_the_boundary():
    rules = pc.EligibilityRules(min_train_samples=10, min_year_samples=10)
    samples = _samples("EDGE", start="2016-12-01", periods=30)
    row = pc.evaluate_eligibility(samples, ticker="EDGE", raw_first_date="2016-12-01",
                                  sector="SYNTH", rules=rules)
    assert isinstance(row["eligible"], bool)
    assert row["train_samples_through_2016"] == rules_min_train(rules, samples)


def rules_min_train(rules: pc.EligibilityRules, samples: pd.DataFrame) -> int:
    origin = pd.to_datetime(samples["origin_date"])
    target = pd.to_datetime(samples["target_date"])
    return int(((origin <= pd.Timestamp(rules.train_end))
                & (target <= pd.Timestamp(rules.train_end))).sum())


# ---------------------------------------------------------------------------
# the calculation never looks past 2019
# ---------------------------------------------------------------------------

def test_eligibility_raises_when_a_post_2019_date_is_present():
    samples = _samples("FUTURE", start="2018-01-01", periods=900)
    samples.loc[samples.index[-1], "target_date"] = pd.Timestamp("2020-01-02")
    with pytest.raises(PostCovidDataAccessError):
        pc.evaluate_eligibility(samples, ticker="FUTURE", raw_first_date="2018-01-01",
                                sector="SYNTH")


def test_derivation_raises_when_a_post_2019_origin_is_present():
    samples = _samples("FUTURE", start="2018-01-01", periods=900)
    samples.loc[samples.index[-1], "origin_date"] = pd.Timestamp("2020-02-03")
    with pytest.raises(PostCovidDataAccessError):
        pc.derive_supervised_universe(samples, raw_first_dates={}, sector_of={})


def test_the_rules_never_mention_a_year_after_2019():
    rules = pc.EligibilityRules().to_dict()
    assert rules["lockbox_year"] == "2019"
    assert rules["final_allowed_date"] == "2019-12-31"
    assert int(rules["dev_year_b"]) <= 2018


# ---------------------------------------------------------------------------
# determinism and the persisted record
# ---------------------------------------------------------------------------

def test_the_derived_list_is_deterministic():
    samples = pd.concat([
        _samples("AAA", start="2005-01-03", periods=3900),
        _samples("BBB", start="2015-01-01", periods=900),
        _samples("CCC", start="2005-01-03", periods=3900),
    ], ignore_index=True)
    raw = {t: "2005-01-03" for t in ("AAA", "BBB", "CCC")}
    sector = {t: "SYNTH" for t in ("AAA", "BBB", "CCC")}
    first = pc.derive_supervised_universe(samples, raw_first_dates=raw, sector_of=sector,
                                         candidates=["AAA", "BBB", "CCC"])
    second = pc.derive_supervised_universe(samples[::-1].reset_index(drop=True),
                                          raw_first_dates=raw, sector_of=sector,
                                          candidates=["CCC", "BBB", "AAA"])
    assert pc.eligible_tickers(first) == pc.eligible_tickers(second)
    assert pc.universe_hash(first) == pc.universe_hash(second)


def test_hash_survives_a_csv_round_trip(tmp_path):
    frame = _universe(_samples("AAA", start="2005-01-03", periods=3900))
    path = tmp_path / "universe.csv"
    frame.to_csv(path, index=False)
    assert pc.universe_hash(pd.read_csv(path)) == pc.universe_hash(frame)


def test_the_frozen_record_has_every_required_column(tmp_path):
    frame = _universe(_samples("AAA", start="2005-01-03", periods=3900))
    csv_path = tmp_path / "pre_covid_supervised_universe.csv"
    yaml_path = tmp_path / "supervised_universe.yaml"
    payload = pc.write_universe(frame, csv_path=csv_path, yaml_path=yaml_path,
                                store_sha256="f" * 64)
    stored = pd.read_csv(csv_path)
    assert list(stored.columns) == list(pc.UNIVERSE_COLUMNS)
    for column in ("ticker", "raw_first_date", "first_usable_sample",
                   "train_samples_through_2016", "samples_2017", "samples_2018",
                   "samples_2019", "eligible", "exclusion_reason", "sector"):
        assert column in stored.columns
    reloaded = pc.load_universe(yaml_path)
    assert reloaded["experiment_regime"] == pc.EXPERIMENT_REGIME
    assert reloaded["survivorship_bias_label"] == pc.BIAS_LABEL
    assert reloaded["frozen"] is True
    assert reloaded["universe_sha256"] == payload["universe_sha256"]
    assert reloaded["eligible_tickers"] == pc.eligible_tickers(frame)


def test_cohort_summary_reports_the_bias_label():
    frame = _universe(
        _samples("AAA", start="2005-01-03", periods=3900),
        _samples("BBB", start="2016-06-01", periods=200),
    )
    summary = pc.cohort_summary(frame)
    assert summary["n_eligible"] == 1
    assert summary["n_excluded"] == 1
    assert summary["survivorship_bias_label"] == "SURVIVORSHIP_BIASED_FIXED_UNIVERSE_RESEARCH_TRACK"
    assert summary["eligible_tickers"] == ["AAA"]


def test_the_real_frozen_universe_is_consistent_with_its_own_record():
    csv_path = pc.PreCovidTrack().path("universe_csv")
    yaml_path = REPO / "configs" / "v2" / "pre_covid" / "supervised_universe.yaml"
    if not csv_path.is_file() or not yaml_path.is_file():
        pytest.skip("the PRE-COVID universe has not been derived in this environment")
    frame = pd.read_csv(csv_path)
    frozen = pc.load_universe(yaml_path)
    assert pc.universe_hash(frame) == frozen["universe_sha256"]
    assert pc.eligible_tickers(frame) == frozen["eligible_tickers"]
    assert frame["eligible"].sum() == frozen["n_eligible"]
    assert int(frame["samples_2019"].min()) >= 0
    # no raw_first_date may be after the regime boundary
    assert (pd.to_datetime(frame["raw_first_date"].dropna()) <= pd.Timestamp(
        "2019-12-31")).all()
    assert (pd.to_datetime(frame["first_usable_sample"].dropna()) <= pd.Timestamp(
        "2019-12-31")).all()
    assert json.loads((pc.PreCovidTrack().results_root /
                       "precovid_universe_summary.json").read_text())[
                           "universe_sha256"] == frozen["universe_sha256"]


def test_nan_free_helper_rejects_missing_features():
    assert pc.nan_free(np.array([[1.0, 2.0]])) is True
    assert pc.nan_free(np.array([[1.0, np.nan]])) is False