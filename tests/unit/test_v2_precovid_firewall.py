"""PRE-COVID firewall and store-isolation tests.

Proves the regime is enforced physically and structurally, not by convention:

A. the PRE-COVID store contains no date after 2019-12-31;
B. its target frame contains no target after 2019-12-31;
C. a 2019-12-31 origin paired with a 2020-01-01 target is rejected;
D. :class:`PostCovidDataAccessError` fires on a 2020 FEATURE date;
E. ... on a 2020 ORIGIN date;
F. ... on a 2020 TARGET date;
G. the ordinary V2 mode still works unchanged;
H. the old 2021 store / results are never overwritten or reused.

The physical checks are driven by a synthetic store built in a tmp directory, so
no real 2020+ row has to be constructed in the repository, while the REAL
PRE-COVID store metadata is verified when it exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_forecaster.v2.dataset import build_sample_table
from agentic_forecaster.v2.experiment import load_run_config
from agentic_forecaster.v2.firewall import (
    PRECOVID_FINAL_DATE,
    PRECOVID_FORBIDDEN_DATE,
    PRECOVID_LOCKBOX_ENV,
    V2_PAPER_TEST_FIREWALL_START,
    PostCovidDataAccessError,
    V2TestFirewallError,
    assert_no_paper_test_targets,
    assert_pre_covid_dates,
    describe_precovid_firewall,
)
from agentic_forecaster.v2.store import (
    assert_store_within_pre_covid,
    load_store,
    precovid_processed_root,
    store_fingerprints,
    store_root,
)

from .v2_fixtures import make_model_config  # noqa: F401

REPO = Path(__file__).resolve().parents[2]
PRECOVID_CONFIG = REPO / "configs" / "v2" / "pre_covid" / "v2_a_shared_lstm.yaml"
ORDINARY_CONFIG = REPO / "configs" / "v2" / "v2_a_shared_lstm.yaml"


# ---------------------------------------------------------------------------
# D/E/F. the exception fires on each kind of date
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("date", ["2020-01-01", "2020-06-15", "2024-12-31"])
def test_pre_covid_firewall_rejects_2020_feature_dates(date: str):
    with pytest.raises(PostCovidDataAccessError, match="feature"):
        assert_pre_covid_dates(feature_dates=[date], where="unit-test")


@pytest.mark.parametrize("date", ["2020-01-02", "2021-03-01"])
def test_pre_covid_firewall_rejects_2020_origin_dates(date: str):
    with pytest.raises(PostCovidDataAccessError, match="origin"):
        assert_pre_covid_dates(origin_dates=[date], where="unit-test")


@pytest.mark.parametrize("date", ["2020-01-02", "2022-05-05"])
def test_pre_covid_firewall_rejects_2020_target_dates(date: str):
    with pytest.raises(PostCovidDataAccessError, match="target"):
        assert_pre_covid_dates(target_dates=[date], where="unit-test")


def test_pre_covid_firewall_accepts_the_boundary():
    assert assert_pre_covid_dates(
        feature_dates=["2019-12-31"], origin_dates=["2019-12-31"],
        target_dates=["2019-12-31"])["rejected_post_covid_dates"] == []


def test_pre_covid_firewall_boundaries_are_the_documented_dates():
    assert str(PRECOVID_FINAL_DATE.date()) == "2019-12-31"
    assert str(PRECOVID_FORBIDDEN_DATE.date()) == "2020-01-01"
    described = describe_precovid_firewall()
    assert described["regime"] == "PRE_COVID_EXPERIMENTAL_REGIME"
    assert described["error"] == "PostCovidDataAccessError"
    assert described["lockbox_env_var"] == PRECOVID_LOCKBOX_ENV
    assert PRECOVID_LOCKBOX_ENV != "V2_LOCKBOX"


def test_pre_covid_error_is_a_v2_firewall_subclass():
    from agentic_forecaster.v2.firewall import V2FirewallError

    assert issubclass(PostCovidDataAccessError, V2FirewallError)
    assert not issubclass(PostCovidDataAccessError, V2TestFirewallError)


# ---------------------------------------------------------------------------
# C. the 2019-12-31 origin / 2020-01-01 target boundary
# ---------------------------------------------------------------------------

def test_dec_2019_origin_with_jan_2020_target_cannot_be_a_sample():
    """The exact boundary the specification calls out.

    A sample whose origin is the last trading day of 2019 and whose target is the
    first trading day of 2020 must NOT exist, and attempting to build it must
    raise rather than be silently filtered.
    """
    frame = pd.DataFrame({
        "ticker": ["AAA"],
        "origin_date": [pd.Timestamp("2019-12-31")],
        "target_date": [pd.Timestamp("2020-01-01")],
        "row": [500],
        "ticker_id": [1],
        "sector_id": [1],
        "y_direction": [1],
        "y_return": [0.1],
        "y_rank": [0.9],
    })
    with pytest.raises(PostCovidDataAccessError, match="target"):
        assert_pre_covid_dates(origin_dates=frame["origin_date"],
                               target_dates=frame["target_date"],
                               where="unit-test boundary")


def test_boundary_sample_is_rejected_by_the_sample_builder(arrays, targets):
    """A poisoned target frame with a 2020 target must abort sample building."""
    poisoned = targets.copy()
    row = poisoned.index[-1]
    poisoned.loc[row, "target_date"] = pd.Timestamp("2020-01-01")
    with pytest.raises(PostCovidDataAccessError):
        build_sample_table(arrays, poisoned, sequence_length=8,
                           require_targets=["y_direction"],
                           max_date="2019-12-31", final_allowed_date="2019-12-31")


def test_pre_covid_mode_off_keeps_ordinary_behaviour(arrays, targets):
    """Without the boundary the same poisoned frame is simply out of range."""
    samples = build_sample_table(arrays, targets, sequence_length=8,
                                 require_targets=["y_direction"],
                                 max_date="2021-12-31")
    assert len(samples) > 0
    assert samples.diagnostics["final_allowed_date"] is None


# ---------------------------------------------------------------------------
# A/B. the physical PRE-COVID store
# ---------------------------------------------------------------------------

def _real_precovid_metadata() -> dict | None:
    path = store_root(precovid_processed_root()) / "metadata.json"
    return json.loads(path.read_text()) if path.is_file() else None


def test_real_precovid_store_contains_no_post_2019_date():
    metadata = _real_precovid_metadata()
    if metadata is None:
        pytest.skip("the PRE-COVID store has not been built in this environment")
    assert metadata["experiment_regime"] == "PRE_COVID"
    assert metadata["final_allowed_date"] == "2019-12-31"
    assert metadata["last_feature_date"] <= "2019-12-31"
    assert metadata["last_target_date"] <= "2019-12-31"
    assert metadata["date_range"]["last_input_date"] <= "2019-12-31"


def test_real_precovid_target_frame_contains_no_post_2019_target():
    metadata = _real_precovid_metadata()
    if metadata is None:
        pytest.skip("the PRE-COVID store has not been built in this environment")
    frame = pd.read_parquet(store_root(precovid_processed_root()) / "target_frame.parquet",
                            columns=["origin_date", "target_date"])
    assert (pd.to_datetime(frame["target_date"]) <= PRECOVID_FINAL_DATE).all()
    assert (pd.to_datetime(frame["origin_date"]) <= PRECOVID_FINAL_DATE).all()


def test_ordinary_store_still_contains_its_2021_rows():
    """G/H: the ordinary V2 store is untouched and still holds 2020-2021."""
    metadata_path = store_root() / "metadata.json"
    if not metadata_path.is_file():
        pytest.skip("the ordinary V2 store has not been built in this environment")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["date_range"]["last_input_date"] == "2021-12-31"
    assert metadata["date_range"]["last_target_date"] == "2021-12-31"


def test_the_two_stores_are_physically_different():
    """The PRE-COVID store is a SIBLING branch, never the ordinary V2 store."""
    ordinary = store_root()
    precovid = store_root(precovid_processed_root())
    assert ordinary != precovid
    assert ordinary.parent == precovid.parent.parent        # .../v2/context_store
    assert precovid.parent.parent == ordinary.parent         # .../v2/pre_covid
    if precovid.is_dir() and ordinary.is_dir():
        assert ordinary.stat().st_ino != precovid.stat().st_ino


def test_precovid_load_refuses_a_store_with_post_2019_rows():
    """A PRE-COVID run must not silently accept the ordinary 2021 store."""
    ordinary_root = store_root().parent
    if not store_root().is_dir():
        pytest.skip("the ordinary V2 store is not present")
    with pytest.raises(PostCovidDataAccessError):
        load_store(ordinary_root, final_allowed_date="2019-12-31")


def test_store_metadata_assertion_rejects_post_2019_metadata():
    with pytest.raises(PostCovidDataAccessError):
        assert_store_within_pre_covid({"last_feature_date": "2020-01-02",
                                       "last_target_date": "2019-12-31"})
    with pytest.raises(PostCovidDataAccessError):
        assert_store_within_pre_covid({"last_feature_date": "2019-12-31",
                                       "last_target_date": "2020-01-03"})
    ok = assert_store_within_pre_covid({"last_feature_date": "2019-12-31",
                                        "last_target_date": "2019-12-31"})
    assert ok["store_max_target_date_within_regime"] is True


def test_precovid_store_hash_differs_from_the_ordinary_store():
    ordinary = store_fingerprints()
    precovid = store_fingerprints(precovid_processed_root())
    if not ordinary.get("built") or not precovid.get("built"):
        pytest.skip("both stores must exist for the hash comparison")
    assert ordinary["store_sha256"] != precovid["store_sha256"]
    assert precovid["max_target_date"] == "2019-12-31"


# ---------------------------------------------------------------------------
# G. ordinary V2 mode still works, H. the 2021 protections are intact
# ---------------------------------------------------------------------------

def test_ordinary_v2_config_still_resolves_its_own_folds():
    config = load_run_config(ORDINARY_CONFIG, fold="V2_DEV_FOLD_A")
    assert config.window.val_start == "2019-01-01"
    assert config.window.val_end == "2019-12-31"
    assert config.pre_covid_mode is False
    assert config.final_allowed_date is None
    assert config.processed_root.name == "v2"
    assert config.results_root.name == "v2"
    assert len(config.supervised_tickers) == 8


def test_ordinary_v2_lockbox_still_needs_the_v2_switch():
    from agentic_forecaster.v2.firewall import lockbox_unlocked

    assert lockbox_unlocked() is False
    with pytest.raises(AssertionError, match="V2_LOCKBOX"):
        load_run_config(ORDINARY_CONFIG, fold="V2_LOCKBOX")


def test_ordinary_v2_2021_protection_is_untouched():
    from agentic_forecaster.v2.firewall import (
        V2LockboxFirewallError,
        assert_no_lockbox_targets,
    )

    with pytest.raises(V2LockboxFirewallError):
        assert_no_lockbox_targets(["2021-06-01"])
    assert assert_no_paper_test_targets(["2021-12-31"]) == "2021-12-31"


def test_ordinary_v2_results_were_not_rewritten_by_the_precovid_track():
    report = REPO / "results" / "v2" / "V2_DEV_REPORT.md"
    summary = REPO / "results" / "v2" / "v2_dev_summary.json"
    if not report.is_file() or not summary.is_file():
        pytest.skip("the historical V2 report is not present")
    text = report.read_text()
    assert "MODEL V2 DEVELOPMENT REPORT" in text
    assert "PRE_COVID" not in text
    payload = json.loads(summary.read_text())
    assert payload.get("experiment_regime", "ORDINARY_V2") == "ORDINARY_V2"
    assert payload["development_folds"] == ["V2_DEV_FOLD_A", "V2_DEV_FOLD_B"]
    assert payload["test_2022_2023_evaluated"] is False
    # the PRE-COVID track has its own ledger and never appends to this one
    ledger = REPO / "results" / "v2" / "experiment_ledger.csv"
    precovid = REPO / "results" / "v2" / "pre_covid" / "experiment_ledger.csv"
    if ledger.is_file():
        assert "pre_covid" not in ledger.read_text()
    if precovid.is_file():
        assert "V2_DEV_FOLD_A" not in precovid.read_text()


def test_precovid_track_does_not_reuse_the_2021_lockbox_authorisation():
    """The PRE-COVID track must not depend on unlocking 2021 anywhere."""
    for name in ("run_v2_precovid_program.sh", "run_v2_precovid_lockbox.py",
                 "summarize_v2_precovid.py"):
        text = (REPO / "scripts" / name).read_text()
        assert "V2_LOCKBOX=1" not in text.replace("unset V2_LOCKBOX", "")
        assert PRECOVID_LOCKBOX_ENV in text or "LOCKBOX_FOLD" in text


def test_2022_firewall_still_applies_inside_the_precovid_track():
    with pytest.raises(V2TestFirewallError):
        assert_no_paper_test_targets(["2022-06-01"])
    with pytest.raises(PostCovidDataAccessError):
        assert_pre_covid_dates(target_dates=["2022-06-01"])


def test_numeric_boundary_helpers_are_consistent():
    assert PRECOVID_FINAL_DATE < V2_PAPER_TEST_FIREWALL_START
    assert np.datetime64(PRECOVID_FINAL_DATE.date()) < np.datetime64("2020-01-01")
    assert PRECOVID_FINAL_DATE.year == 2019
    assert PRECOVID_FINAL_DATE.month == 12