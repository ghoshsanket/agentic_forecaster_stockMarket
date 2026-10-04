"""V3 FAIR COMPARISON: the guarantees that make an increment meaningful.

Covers section 52 of the specification:

* the X0-X4 common-sample evaluation uses IDENTICAL (ticker, origin) pairs;
* the target labels are IDENTICAL between feature-family comparisons;
* the horizon implementation is the already-verified one, not a V3 re-implementation;
* 2019 is inaccessible during model selection;
* 2020+ is inaccessible everywhere;
* every earlier track stays untouched and green.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_forecaster.v2 import horizons as HZ
from agentic_forecaster.v2.dataset import SplitWindow
from agentic_forecaster.v2.firewall import PostCovidDataAccessError
from agentic_forecaster.v2.horizon_dataset import build_horizon_sample_table, horizon_split
from agentic_forecaster.v3 import CONTROL_FAMILY, FEATURE_FAMILIES, V3Track
from agentic_forecaster.v3.dataset import (
    build_family_tables,
    common_family_mask,
    family_split_mask,
)
from agentic_forecaster.v3.experiment import append_row
from agentic_forecaster.v3.firewall import V3LockboxError as MultiHorizonLockboxError
from agentic_forecaster.v3.firewall import lockbox_unlocked
from agentic_forecaster.v3.store import (
    FAMILY_MEMBERSHIP,
    availability_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "v3"
WINDOW = SplitWindow("MH_DEV_2017", "2015-01-01", "2016-12-31",
                     "2017-01-01", "2017-12-31")

from .multi_horizon_fixtures import make_horizon_arrays, make_horizon_targets


@pytest.fixture(scope="module")
def samples_by_horizon():
    arrays = make_horizon_arrays()
    targets = make_horizon_targets()
    return {horizon: build_horizon_sample_table(arrays, targets, horizon=horizon,
                                                sequence_length=20,
                                                final_allowed_date="2019-12-31")
            for horizon in HZ.HORIZONS}


class _StubExogenousStore:
    """A minimal date-keyed stand-in for the real exogenous store."""

    def __init__(self, samples_by_horizon, registry, families=FEATURE_FAMILIES):
        self.samples_by_horizon = samples_by_horizon
        self.registry = registry
        self.families = families
        dates = pd.DatetimeIndex(sorted(set(
            pd.to_datetime(samples_by_horizon[1].frame["origin_date"]))))
        columns = ["date"]
        for family in families:
            if family == CONTROL_FAMILY:
                continue
            for source in ("AAA_IDX", "BBB_IDX"):
                # a sector index needs all three relative windows, so the stub
                # provides return_1/5/20 for the mapped source
                for template in ("return_1", "return_5", "return_20"):
                    columns.append(f"{source}_{template}")
        # the market index the relative block is built from
        for template in ("return_1", "return_5", "return_20"):
            columns.append(f"NIFTY50_{template}")
        values = {column: np.arange(len(dates), dtype=float) / 1000.0
                  for column in columns if column != "date"}
        self.features = pd.DataFrame({"date": dates, **values})
        universe = sorted(set(samples_by_horizon[1].frame["ticker"]))
        self.sector_mapping = pd.DataFrame({
            "ticker": universe,
            "sector": ["IT"] * len(universe),
            "external_sector_index_id": ["AAA_IDX"] * len(universe),
            "mapping_confidence": [0.9] * len(universe),
            "provenance": ["test"] * len(universe)})
        self.metadata = {"alignment": {}, "last_date": str(dates.max().date())}
        self.schema_sha256 = "test-schema"
        self.root = Path(".")

    @property
    def feature_columns(self) -> tuple[str, ...]:
        reserved = ("date",)
        return tuple(c for c in self.features.columns
                     if c not in reserved
                     and not c.startswith(("source_observation_", "lag_")))


def _stock_features(samples_by_horizon) -> pd.DataFrame:
    """The stock frame in the shape the family builder expects: ``ticker``/``date``."""
    return pd.concat([
        samples.frame.loc[:, ["ticker", "origin_date"]]
        .rename(columns={"origin_date": "date"})
        .assign(log_return_1=0.01, log_return_5=0.02, log_return_20=0.03)
        for samples in samples_by_horizon.values()
    ]).drop_duplicates(subset=["ticker", "date"], keep="first")


def _stub_registry():
    from agentic_forecaster.v3.sources import CLASS_A, CLASS_B, SourceSpec

    specs = []
    for index, source_id in enumerate(("AAA_IDX", "BBB_IDX")):
        specs.append(SourceSpec(
            source_id, f"stub {source_id}", "test", source_id, "EQUITY_INDEX",
            "Asia/Kolkata", "session", CLASS_A if index == 0 else CLASS_B, "1D",
            "2005-01-01",
            family=("E1_INDIA_MARKET" if index == 0 else "E2_GLOBAL_RISK"),
            accepted=True, available=True, raw_sha256="0" * 64))
    from agentic_forecaster.v3.sources import SourceRegistry

    registry = SourceRegistry(sources=specs)
    for spec in specs:
        spec.accepted = True
    return registry


# ---------------------------------------------------------------------------
# identical observations on the common sample
# ---------------------------------------------------------------------------

def test_common_sample_uses_identical_ticker_origin_pairs(samples_by_horizon):
    """Every family is scored on the SAME observations in the common view."""
    registry = _stub_registry()
    store = _StubExogenousStore(samples_by_horizon, registry)
    stock_features = _stock_features(samples_by_horizon)
    tables = build_family_tables(samples_by_horizon, store, registry,
                                 stock_features=stock_features)
    masks = common_family_mask(tables, WINDOW, horizon=5, fold=WINDOW.name)

    key_sets = {}
    for family, block in masks.items():
        if family == "__core__":
            continue
        frame = tables[(5, family)].samples.frame
        selected = frame.loc[block["val"]]
        key_sets[family] = set(zip(selected["ticker"], selected["origin_date"]))
        assert block["n_val"] == len(key_sets[family])
    assert key_sets, "no family produced a common sample"
    reference = key_sets[CONTROL_FAMILY]
    for family, keys in key_sets.items():
        assert keys == reference, (
            f"{family} was scored on {len(keys)} observations while the control had "
            f"{len(reference)}; the common sample must be identical")


def test_natural_views_may_differ_but_are_reported_separately(samples_by_horizon):
    registry = _stub_registry()
    store = _StubExogenousStore(samples_by_horizon, registry)
    stock_features = _stock_features(samples_by_horizon)
    tables = build_family_tables(samples_by_horizon, store, registry,
                                 stock_features=stock_features)
    natural = family_split_mask(tables[(5, "X1_INDIA_MARKET")], WINDOW,
                                fold=WINDOW.name)
    common = common_family_mask(tables, WINDOW, horizon=5, fold=WINDOW.name)
    assert common["X1_INDIA_MARKET"]["n_val"] <= natural["val"].sum(), (
        "the common view can only be a subset of the natural view")
    assert common["X1_INDIA_MARKET"]["n_val"] > 0


def test_control_family_carries_no_exogenous_block(samples_by_horizon):
    registry = _stub_registry()
    store = _StubExogenousStore(samples_by_horizon, registry)
    stock_features = _stock_features(samples_by_horizon)
    tables = build_family_tables(samples_by_horizon, store, registry,
                                 stock_features=stock_features)
    control = tables[(5, CONTROL_FAMILY)]
    assert control.exogenous_columns == ()
    assert control.diagnostics["n_exogenous_columns"] == 0
    assert control.diagnostics["coverage_fraction"] == 1.0


def test_every_family_declares_the_same_stock_feature_schema(samples_by_horizon):
    registry = _stub_registry()
    store = _StubExogenousStore(samples_by_horizon, registry)
    stock_features = _stock_features(samples_by_horizon)
    tables = build_family_tables(samples_by_horizon, store, registry,
                                 stock_features=stock_features)
    schemas = {family: tables[(5, family)].samples.arrays.stock_features
               for family in FEATURE_FAMILIES}
    reference = schemas[CONTROL_FAMILY]
    for family, schema in schemas.items():
        assert list(schema) == list(reference), (
            f"{family} does not use the unchanged 27-feature stock schema")


# ---------------------------------------------------------------------------
# identical labels
# ---------------------------------------------------------------------------

def test_target_labels_are_identical_across_feature_families(samples_by_horizon):
    registry = _stub_registry()
    store = _StubExogenousStore(samples_by_horizon, registry)
    stock_features = _stock_features(samples_by_horizon)
    tables = build_family_tables(samples_by_horizon, store, registry,
                                 stock_features=stock_features)
    reference = tables[(5, CONTROL_FAMILY)].samples.frame
    for family in FEATURE_FAMILIES:
        frame = tables[(5, family)].samples.frame
        assert frame["ticker"].equals(reference["ticker"]), family
        assert frame["origin_date"].equals(reference["origin_date"]), family
        assert frame["target_date"].equals(reference["target_date"]), family
        assert frame["y_direction"].equals(reference["y_direction"]), family


def test_horizon_implementation_is_the_verified_v2_one():
    """V3 delegates to the multi-horizon implementation instead of rebuilding it."""
    source = (REPO_ROOT / "src/agentic_forecaster/v3/dataset.py").read_text()
    assert "from agentic_forecaster.v2.horizon_dataset import horizon_split" in source
    assert "from agentic_forecaster.v2 import horizons as HZ" in source
    # V3 must not define its own trading-day offset or its own target equation
    assert "timedelta" not in source
    assert "log(Close[t+H]" not in source


def test_objective_ids_come_from_the_shared_multi_horizon_registry():
    for horizon in (1, 3, 5, 10):
        assert HZ.objective_id(horizon).startswith("ABS_DIR_")
    assert HZ.objective_id(1) == "ABS_DIR_1D_CONTROL"


# ---------------------------------------------------------------------------
# 2019 and 2020+ are inaccessible during selection
# ---------------------------------------------------------------------------

def test_development_sample_tables_exclude_the_lockbox_year(samples_by_horizon):
    for samples in samples_by_horizon.values():
        assert samples.diagnostics["lockbox_unlocked"] is False
        assert samples.frame["target_date"].max() <= pd.Timestamp("2018-12-31")


def test_a_development_split_declaring_2019_raises(samples_by_horizon):
    lockbox_window = SplitWindow("MH_LOCKBOX_2019", "2005-01-01", "2018-12-31",
                                 "2019-01-01", "2019-12-31")
    # the shared multi-horizon splitter enforces the seal, and V3 inherits it
    with pytest.raises((MultiHorizonLockboxError, HZ.MultiHorizonLockboxError)):
        horizon_split(samples_by_horizon[5], lockbox_window,
                      fold=lockbox_window.name, locked=False)


def test_lockbox_switch_is_v3_specific(monkeypatch):
    monkeypatch.delenv("V3_PRECOVID_LOCKBOX", raising=False)
    monkeypatch.setenv("V2_LOCKBOX", "1")
    monkeypatch.setenv("PRECOVID_LOCKBOX", "1")
    monkeypatch.setenv("MULTI_HORIZON_LOCKBOX", "1")
    assert lockbox_unlocked() is False, (
        "another track's switch must never unlock the V3 lockbox")
    monkeypatch.setenv("V3_PRECOVID_LOCKBOX", "1")
    assert lockbox_unlocked() is True


def test_2020_is_inaccessible_everywhere():
    from agentic_forecaster.v3.firewall import assert_no_post_2019

    for column in ("feature_dates", "origin_dates", "exogenous_dates", "target_dates"):
        with pytest.raises(PostCovidDataAccessError):
            assert_no_post_2019(**{column: [pd.Timestamp("2020-01-02")]},
                                final_allowed_date="2019-12-31")


def test_ledger_refuses_a_false_lockbox_or_post_2019_claim(tmp_path):
    with pytest.raises(ValueError, match="2019_lockbox_evaluated=true"):
        append_row({"experiment_id": "A", "model": "LOGISTIC",
                    "feature_family": "X1_INDIA_MARKET", "horizon": 5,
                    "fold": "MH_DEV_2017", "2019_lockbox_evaluated": True},
                   path=tmp_path / "ledger.csv")
    with pytest.raises(ValueError, match="post_2019_evaluated=true"):
        append_row({"experiment_id": "B", "model": "LOGISTIC",
                    "feature_family": "X1_INDIA_MARKET", "horizon": 5,
                    "fold": "MH_DEV_2017", "post_2019_evaluated": True},
                   path=tmp_path / "ledger.csv")


def test_family_membership_covers_every_non_control_family():
    assert set(FAMILY_MEMBERSHIP) == set(FEATURE_FAMILIES) - {CONTROL_FAMILY}
    assert FAMILY_MEMBERSHIP["X4_ALL_EXOGENOUS"] == (
        FAMILY_MEMBERSHIP["X1_INDIA_MARKET"] + FAMILY_MEMBERSHIP["X2_GLOBAL_RISK"]
        + FAMILY_MEMBERSHIP["X3_MACRO_COMMODITY"])


# ---------------------------------------------------------------------------
# the earlier tracks stay untouched
# ---------------------------------------------------------------------------

def test_earlier_track_artefacts_are_untouched():
    assert (REPO_ROOT / "results/v2/experiment_ledger.csv").is_file()
    assert (REPO_ROOT / "results/v2/pre_covid/PRE_COVID_V2_REPORT.md").is_file()
    assert (REPO_ROOT / "results/v2/multi_horizon/MULTI_HORIZON_REPORT.md").is_file()
    from agentic_forecaster.v2.ledger import read_ledger

    assert len(read_ledger()) == 6
    assert len(read_ledger(results_root="results/v2/pre_covid")) == 6
    assert not list(REPO_ROOT.glob("results/v2/v3*")), (
        "V3 must never write inside results/v2")


def test_v3_writes_only_to_its_own_track():
    track = V3Track()
    assert track.results_root.as_posix().endswith("results/v3/pre_covid_exogenous")
    assert "v3/pre_covid_exogenous" in track.runtime_root.as_posix()
    assert "v3/pre_covid_exogenous" in track.processed_root.as_posix()
    assert track.ledger.parent.name == "pre_covid_exogenous"


def test_configs_declare_the_separate_track():
    from agentic_forecaster.v2.horizons import load_track_config

    for name in ("precovid_exogenous_base.yaml", "screen_stock_only.yaml",
                 "screen_india_market.yaml", "screen_global_risk.yaml",
                 "screen_macro_commodity.yaml", "screen_all_exogenous.yaml",
                 "shared_lstm.yaml", "lstm_transformer.yaml"):
        assert (CONFIG_DIR / name).is_file(), name
    base = load_track_config(CONFIG_DIR / "precovid_exogenous_base.yaml")
    assert base["results_root"] == "results/v3/pre_covid_exogenous"
    assert base["final_allowed_date"] == "2019-12-31"
    assert base["feature_families"] == list(FEATURE_FAMILIES)
    assert base["control_family"] == CONTROL_FAMILY
    assert base["data"]["source_store_root"].endswith("v2/pre_covid")
    assert "v2/multi_horizon/horizon_targets" in base["data"]["horizon_target_cache"]
    assert base["availability"]["external_same_date_allowed"] is False
    assert base["feature_family_gate"]["min_min_incremental_auc"] == 0.015
    assert base["seed_stability"]["seeds"] == [11, 42, 73]
    assert base["neural_gate"]["min_mean_roc_auc"] == 0.56
    assert base["neural"]["not_run_in_this_programme"] == ["FILM", "REPTILE", "MAML",
                                                          "MULTITASK"]


def test_master_script_has_a_lock_and_the_hard_stops():
    script = (REPO_ROOT / "scripts" / "run_v3_precovid_program.sh").read_text()
    assert "set -Eeuo pipefail" in script
    assert "flock" in script
    assert "V3_PRECOVID_LOCKBOX=1" in script
    assert script.count("STOP") >= 3
    assert "BUILD_HISTORICAL_SENTIMENT_EVENT_DATASET" in script
    for forbidden in ("FINBERT", "run_recovered_paper", "PAPER_REFERENCE"):
        assert forbidden not in script


def test_source_registry_declares_both_families_and_every_required_field():
    from agentic_forecaster.v3.sources import REGISTRY_FIELDS, load_registry

    registry = load_registry(CONFIG_DIR / "source_registry.yaml")
    assert len(registry) > 20
    for spec in registry:
        for field in REGISTRY_FIELDS:
            assert hasattr(spec, field), f"{spec.source_id} lacks {field}"
    assert registry.by_family("E1_INDIA_MARKET")
    assert registry.by_family("E2_GLOBAL_RISK")
    assert registry.by_family("E3_FX_COMMODITY_RATES")
    # FII/DII are declared but DISABLED, so the programme never depends on them
    flows = registry.by_family("FII_DII_FLOWS")
    assert flows and all(not spec.enabled for spec in flows)
    # every non-Indian source is on the conservative lag by default
    for spec in registry:
        if spec.family in ("E1_INDIA_MARKET",):
            assert spec.availability_policy == "INDIA_SAME_CLOSE"
        else:
            assert spec.availability_policy == "EXTERNAL_CONSERVATIVE_LAG1"


def test_availability_audit_catches_a_deliberate_violation(samples_by_horizon):
    store = _StubExogenousStore(samples_by_horizon, _stub_registry())
    block = store.features.head(50).copy()
    block["source_observation_SP500"] = block["date"] + pd.Timedelta(days=1)
    block["lag_SP500"] = 1.0
    store.features = block
    store.metadata["alignment"] = {
        "SP500": {"availability_class": "EXTERNAL_CONSERVATIVE_LAG1"}}
    audit = availability_audit(store, sample_size=20, seed=1)
    assert audit["passed"] is False
    assert audit["n_violations"] >= 1


def test_python_and_import_surface():
    assert sys.version_info >= (3, 11)
    assert subprocess.run(
        [sys.executable, "-c", "import agentic_forecaster.v3.neural"],
        capture_output=True, check=False).returncode == 0


def test_report_json_is_machine_readable(samples_by_horizon):
    path = V3Track().path("report")
    if path.is_file():
        text = path.read_text()
        assert "V3_EXOGENOUS_SIGNAL:" in text
        assert "SURVIVORSHIP_BIASED_FIXED_UNIVERSE_RESEARCH_TRACK" in text
        summary = V3Track().path("summary")
        if summary.is_file():
            payload = json.loads(summary.read_text())
            assert payload["track"] == "V3_EXOGENOUS_PRECOVID"
            assert "recommended_next_action" in payload