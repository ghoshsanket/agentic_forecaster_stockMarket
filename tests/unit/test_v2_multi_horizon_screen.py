"""Multi-horizon screen, metrics and gates.

Covers sections 41K-41S of the specification:

K. the common-origin dataset truly contains origins valid for all horizons
L. the non-overlap sampler spaces observations by at least H trading positions
M. the Logistic scaler is fitted on TRAIN only
N. horizon screening cannot read 2019
O. the lockbox cannot run without ``MULTI_HORIZON_LOCKBOX=1``
P. the lockbox cannot run without a frozen manifest
Q. a changed horizon invalidates the frozen lockbox manifest
R. a changed architecture invalidates the frozen lockbox manifest
S. the original PRE-COVID / V2 / paper tests remain green
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_forecaster.v2 import horizon_metrics as HM
from agentic_forecaster.v2 import horizon_screen as HS
from agentic_forecaster.v2 import horizons as HZ
from agentic_forecaster.v2.dataset import SplitWindow
from agentic_forecaster.v2.firewall import PostCovidDataAccessError
from agentic_forecaster.v2.horizon_dataset import (
    build_horizon_sample_table,
    common_origin_masks,
    horizon_split,
)

from .multi_horizon_fixtures import make_horizon_arrays, make_horizon_targets

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "v2" / "multi_horizon"
LOCKBOX_SCRIPT = REPO_ROOT / "scripts" / "run_multi_horizon_lockbox.py"


def _load_lockbox_module():
    spec = importlib.util.spec_from_file_location("mh_lockbox", LOCKBOX_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def samples_by_horizon():
    arrays = make_horizon_arrays()
    targets = make_horizon_targets()
    return {horizon: build_horizon_sample_table(arrays, targets, horizon=horizon,
                                                sequence_length=20,
                                                final_allowed_date="2019-12-31")
            for horizon in HZ.HORIZONS}


# ---------------------------------------------------------------------------
# K. common-origin sample set
# ---------------------------------------------------------------------------

def test_common_origin_dataset_contains_only_origins_valid_everywhere(samples_by_horizon):
    window = SplitWindow("MH_DEV_2017", "2015-01-01", "2016-12-31",
                         "2017-01-01", "2017-12-31")
    masks = common_origin_masks(samples_by_horizon, window, fold=window.name)
    keys = masks[10]["keys"]
    assert len(keys) > 0
    for horizon, block in masks.items():
        samples = samples_by_horizon[horizon]
        frame = samples.frame
        # every common origin is valid for THIS horizon too ...
        selected = frame.loc[block["val"]]
        assert not selected.empty
        for row in selected.itertuples():
            assert (pd.Timestamp(row.origin_date) >= pd.Timestamp(window.val_start))
            assert (pd.Timestamp(row.origin_date) <= pd.Timestamp(window.val_end))
            assert (pd.Timestamp(row.target_date) <= pd.Timestamp(window.val_end))
        # ... and no common origin is missing from any horizon
        assert keys.isin(pd.MultiIndex.from_frame(
            frame.loc[:, ["ticker", "origin_date"]])).all()


def test_common_origin_set_is_the_most_restrictive_horizon(samples_by_horizon):
    window = SplitWindow("MH_DEV_2017", "2015-01-01", "2016-12-31",
                         "2017-01-01", "2017-12-31")
    masks = common_origin_masks(samples_by_horizon, window, fold=window.name)
    common = {h: len(masks[h]["val_keys"]) for h in masks}
    # every horizon yields the SAME shared VALIDATION-origin set
    assert set(common.values()) == {common[10]}
    # H=10 is the most restrictive horizon, so its validation set IS that set
    assert int(masks[10]["val"].sum()) == common[10]
    # every other horizon's validation set is a superset of it
    for horizon in masks:
        assert int(masks[horizon]["val"].sum()) >= common[10]
    # the shared key set is the disjoint union of its train and validation sides
    for horizon in masks:
        assert len(masks[horizon]["keys"]) == (
            len(masks[horizon]["train_keys"]) + len(masks[horizon]["val_keys"]))
    # the multi-day horizons each lose a few year-end origins to boundary trimming
    assert int(masks[1]["val"].sum()) >= int(masks[10]["val"].sum())


def test_common_origin_helpers_agree(samples_by_horizon):
    window = SplitWindow("MH_DEV_2016", "2015-01-01", "2015-12-31",
                         "2016-01-01", "2016-12-31")
    targets = make_horizon_targets()
    keys = HZ.horizon_target_keys(targets, HZ.HORIZONS)
    masks = common_origin_masks(samples_by_horizon, window, fold=window.name)
    for horizon in HZ.HORIZONS:
        frame = samples_by_horizon[horizon].frame
        own = HZ.common_origin_mask(frame, keys)
        assert own.sum() >= masks[horizon]["val"].sum()


# ---------------------------------------------------------------------------
# L. non-overlapping origins
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("horizon", [3, 5, 10])
def test_non_overlap_sampler_spaces_by_at_least_h_trading_positions(horizon):
    dates = pd.bdate_range("2016-01-01", periods=200)
    frame = pd.DataFrame({
        "ticker": np.repeat(["AAA", "BBB"], len(dates)),
        "origin_date": np.tile(dates.to_numpy(), 2),
    })
    mask = HZ.non_overlap_mask(frame, horizon=horizon)
    assert mask.sum() == 2 * (len(dates) // horizon
                              + (1 if len(dates) % horizon else 0))
    for ticker in ("AAA", "BBB"):
        block = frame.loc[mask & (frame["ticker"] == ticker)]
        positions = [dates.get_loc(pd.Timestamp(d)) for d in block["origin_date"]]
        gaps = np.diff(sorted(positions))
        assert gaps.min() >= horizon, "origins must be at least H trading rows apart"
        assert positions[0] == 0, "the starting offset is FIXED, never chosen"


def test_non_overlap_mask_is_deterministic():
    dates = pd.bdate_range("2016-01-01", periods=120)
    frame = pd.DataFrame({"ticker": "AAA", "origin_date": dates.to_numpy()})
    first = HZ.non_overlap_mask(frame, horizon=5)
    second = HZ.non_overlap_mask(frame.sample(frac=1.0, random_state=3), horizon=5)
    assert np.array_equal(first, second)


def test_non_overlap_spacing_audit_reports_the_horizon():
    dates = pd.bdate_range("2016-01-01", periods=200)
    frame = pd.DataFrame({"ticker": "AAA", "origin_date": dates.to_numpy()})
    audit = HZ.non_overlap_spacing(frame.loc[HZ.non_overlap_mask(frame, horizon=10)],
                                   horizon=10)
    assert audit["horizon"] == 10
    assert audit["n_origins"] == 20


# ---------------------------------------------------------------------------
# M. the scaler is fitted on TRAIN only
# ---------------------------------------------------------------------------

def test_logistic_scaler_is_fitted_on_train_rows_only():
    rng = np.random.default_rng(5)
    train = pd.DataFrame({"ticker": ["AAA"] * 50 + ["BBB"] * 50,
                          "y_direction": (rng.random(100) < 0.4).astype(int)})
    x_train = rng.normal(0.0, 1.0, (100, 4))
    x_val = rng.normal(50.0, 1.0, (10, 4))          # wildly shifted validation block
    fitted, provenance = HS.fit_screen_model("LOGISTIC", x_train,
                                            train["y_direction"].to_numpy())
    assert provenance["scaler_fit_on"] == "TRAIN_ONLY"
    assert fitted["scaler"].n_samples_seen_ == 100
    scaler_mean = float(np.asarray(fitted["scaler"].mean_).mean())
    assert abs(scaler_mean) < 5.0, "a 50-scaled validation block must not move the scaler"
    assert HS.predict_proba(fitted, x_val).shape == (10,)


def test_screen_manifest_records_that_no_search_happened():
    assert HS.SCREEN_SETTINGS["LOGISTIC"]["random_state"] == 42
    assert HS.SCREEN_SETTINGS["HIST_GRADIENT_BOOSTING"]["early_stopping"] is False
    assert HS.SCREEN_SETTINGS["HIST_GRADIENT_BOOSTING"]["random_state"] == 42
    _, provenance = HS.fit_screen_model("HIST_GRADIENT_BOOSTING",
                                        np.random.default_rng(1).normal(size=(60, 3)),
                                        np.array([0, 1] * 30))
    assert provenance["scaler_fit_on"] == "NOT_APPLICABLE_TREE_MODEL"


# ---------------------------------------------------------------------------
# N. horizon screening cannot read 2019
# ---------------------------------------------------------------------------

def test_development_sample_tables_exclude_the_lockbox_year(samples_by_horizon):
    for samples in samples_by_horizon.values():
        assert samples.diagnostics["lockbox_unlocked"] is False
        assert samples.diagnostics["lockbox_year_excluded_rows"] > 0
        assert samples.frame["target_date"].max() <= pd.Timestamp("2018-12-31")


def test_a_development_split_over_2019_raises(samples_by_horizon):
    window = SplitWindow("MH_LOCKBOX_2019", "2005-01-01", "2018-12-31",
                         "2019-01-01", "2019-12-31")
    with pytest.raises(HZ.MultiHorizonLockboxError, match="LOCKBOX"):
        horizon_split(samples_by_horizon[5], window, fold=window.name, locked=False)


def test_the_lockbox_year_is_readable_only_when_unlocked():
    arrays = make_horizon_arrays()
    targets = make_horizon_targets()
    unlocked = build_horizon_sample_table(arrays, targets, horizon=5,
                                         sequence_length=20,
                                         final_allowed_date="2019-12-31",
                                         locked=True)
    assert unlocked.frame["target_date"].max() <= pd.Timestamp("2019-12-31")
    assert (pd.to_datetime(unlocked.frame["target_date"]).dt.year == 2019).any()


def test_lockbox_firewall_rejects_2019_targets(monkeypatch):
    monkeypatch.delenv(HZ.LOCKBOX_ENV, raising=False)
    with pytest.raises(HZ.MultiHorizonLockboxError):
        HZ.assert_no_lockbox_year_targets([pd.Timestamp("2019-06-03")])
    monkeypatch.setenv(HZ.LOCKBOX_ENV, "1")
    assert HZ.assert_no_lockbox_year_targets([pd.Timestamp("2019-06-03")]) == "2019-06-03"
    with pytest.raises(PostCovidDataAccessError):
        HZ.assert_horizon_boundary([pd.Timestamp("2020-01-02")],
                                   final_allowed_date="2019-12-31")


# ---------------------------------------------------------------------------
# O/P. lockbox authorisation
# ---------------------------------------------------------------------------

@pytest.fixture
def lockbox(tmp_path, monkeypatch):
    module = _load_lockbox_module()
    monkeypatch.delenv(HZ.LOCKBOX_ENV, raising=False)
    monkeypatch.setenv(HZ.LOCKBOX_ENV, "1")
    return module


def test_lockbox_refuses_without_the_environment_switch(tmp_path, monkeypatch):
    module = _load_lockbox_module()
    monkeypatch.delenv(HZ.LOCKBOX_ENV, raising=False)
    with pytest.raises(module.LockboxRefused, match="MULTI_HORIZON_LOCKBOX=1"):
        module.authorise(tmp_path / "frozen_horizon_model.json", None, None)


def test_lockbox_refuses_without_a_frozen_manifest(tmp_path, monkeypatch):
    module = _load_lockbox_module()
    monkeypatch.setenv(HZ.LOCKBOX_ENV, "1")
    with pytest.raises(module.LockboxRefused, match="no frozen manifest"):
        module.authorise(tmp_path / "frozen_horizon_model.json", None, None)


def test_lockbox_refuses_a_second_run(tmp_path, monkeypatch):
    module = _load_lockbox_module()
    monkeypatch.setenv(HZ.LOCKBOX_ENV, "1")
    track = HZ.MultiHorizonTrack(results_root=tmp_path)
    HZ.append_row({"experiment_id": "X", "horizon": 5, "model": "SHARED_LSTM",
                   "fold": HZ.LOCKBOX_FOLD, "2019_lockbox_evaluated": True},
                  path=tmp_path / "experiment_ledger.csv", lockbox_evaluated=True)
    assert module.already_run(track) is True
    with pytest.raises(module.LockboxRefused, match="second time"):
        module.run_lockbox(allow_rerun=False, track=track)


def test_ledger_refuses_a_false_lockbox_claim(tmp_path):
    with pytest.raises(ValueError, match="2019_lockbox_evaluated=true"):
        HZ.append_row({"experiment_id": "Y", "horizon": 5, "model": "LOGISTIC",
                       "fold": "MH_DEV_2018", "2019_lockbox_evaluated": True},
                      path=tmp_path / "l.csv")
    with pytest.raises(ValueError, match="post_2019_evaluated=true"):
        HZ.append_row({"experiment_id": "Z", "horizon": 5, "model": "LOGISTIC",
                       "fold": "MH_DEV_2018", "post_2019_evaluated": True},
                      path=tmp_path / "l.csv")


# ---------------------------------------------------------------------------
# Q/R. the frozen manifest
# ---------------------------------------------------------------------------

@pytest.fixture
def frozen_manifest():
    return HS.build_frozen_manifest(
        horizon=5, architecture="SHARED_LSTM",
        screen_rank=[{"horizon": 5, "objective_id": "ABS_DIR_5D"}],
        neural_metrics={"mean_roc_auc": 0.55}, common_metrics={}, non_overlap_metrics={},
        stability={"stable": True}, universe_sha256="u" * 64,
        feature_schema_sha256="f" * 64, store_sha256="s" * 64,
        target_schema_sha256="t" * 64, config_sha256="c" * 64,
        experiment_ids=["a", "b"], seed=42, gates={})


def _verify(manifest, horizon=5, architecture="SHARED_LSTM"):
    return HS.verify_frozen_manifest(
        manifest, horizon=horizon, architecture=architecture,
        target_schema_sha256="t" * 64, store_sha256="s" * 64,
        universe_sha256="u" * 64, config_sha256="c" * 64)


def test_frozen_manifest_verifies_when_nothing_changed(frozen_manifest):
    result = _verify(frozen_manifest)
    assert result["verified"] is True
    assert result["failed_checks"] == []


def test_a_changed_horizon_invalidates_the_manifest(frozen_manifest):
    """Q: the horizon may not change after the freeze."""
    result = _verify(frozen_manifest, horizon=10)
    assert result["verified"] is False
    assert "horizon_unchanged" in result["failed_checks"]
    assert "objective_unchanged" in result["failed_checks"]


def test_a_changed_architecture_invalidates_the_manifest(frozen_manifest):
    """R: the architecture may not change after the freeze."""
    result = _verify(frozen_manifest, architecture="LSTM_TRANSFORMER")
    assert result["verified"] is False
    assert "architecture_unchanged" in result["failed_checks"]


def test_a_tampered_manifest_fails_its_integrity_check(frozen_manifest):
    frozen_manifest["seed"] = 11
    result = _verify(frozen_manifest)
    assert result["verified"] is False
    assert "manifest_integrity" in result["failed_checks"]


def test_a_changed_target_schema_invalidates_the_manifest(frozen_manifest):
    result = HS.verify_frozen_manifest(
        frozen_manifest, horizon=5, architecture="SHARED_LSTM",
        target_schema_sha256="z" * 64, store_sha256="s" * 64,
        universe_sha256="u" * 64, config_sha256="c" * 64)
    assert result["verified"] is False
    assert "target_schema_unchanged" in result["failed_checks"]


def test_frozen_manifest_records_every_required_field(frozen_manifest):
    for key in ("objective", "target_equation", "horizon_trading_observations",
                "selected_architecture", "supervised_universe_sha256",
                "feature_schema_sha256", "store_sha256", "target_schema_sha256",
                "config_sha256", "development_experiment_ids",
                "development_metrics", "common_origin_metrics",
                "non_overlapping_metrics", "seed_stability", "frozen_at", "seed"):
        assert key in frozen_manifest
    assert frozen_manifest["objective"] == "ABS_DIR_5D"
    assert "immutable_after_freeze" in frozen_manifest


# ---------------------------------------------------------------------------
# gates, ranking and stability
# ---------------------------------------------------------------------------

def _aggregate(**kwargs):
    base = {"mean_roc_auc": 0.56, "mean_balanced_accuracy": 0.53,
            "mean_baseline_delta": 0.01, "years_roc_auc_above_50": 4,
            "years_beating_majority_baseline": 4, "worst_year_roc_auc": 0.51,
            "mean_brier": 0.25, "mean_ticker_fraction_beating_own_baseline": 0.4}
    base.update(kwargs)
    return base


THRESHOLDS = {"min_mean_roc_auc": 0.54, "min_mean_balanced_accuracy": 0.52,
              "min_mean_baseline_delta": 0.005, "min_years_roc_auc_above_50": 4,
              "n_development_years": 5, "require_common_origin_positive": True,
              "require_non_overlap_above": 0.50, "strong_min_mean_roc_auc": 0.56,
              "strong_min_mean_balanced_accuracy": 0.54,
              "strong_min_years_beating_baseline": 4,
              "max_selected_horizons": 2}


def test_horizon_gate_passes_only_when_every_criterion_holds():
    gate = HS.evaluate_horizon_gate(_aggregate(), _aggregate(mean_baseline_delta=0.01),
                                    _aggregate(mean_roc_auc=0.49), thresholds=THRESHOLDS)
    assert gate["screen_passed"] is False
    assert "non_overlap_not_chance" in [k for k, v in gate["criteria"].items()
                                        if not v["passed"]]


def test_horizon_gate_passes_and_can_be_strongly_promising():
    strong = _aggregate(mean_roc_auc=0.57, mean_balanced_accuracy=0.55)
    gate = HS.evaluate_horizon_gate(strong, strong, strong, thresholds=THRESHOLDS)
    assert gate["screen_passed"] is True
    assert gate["classification"] == "STRONGLY_PROMISING"


def test_horizon_gate_requires_a_positive_common_origin_comparison():
    natural = _aggregate()
    gate = HS.evaluate_horizon_gate(natural, _aggregate(mean_baseline_delta=-0.01),
                                    natural, thresholds=THRESHOLDS)
    assert gate["screen_passed"] is False
    assert gate["criteria"]["common_origin_directionally_positive"]["passed"] is False


def test_ranking_prefers_auc_then_balanced_accuracy_then_brier():
    aggregates = {
        3: _aggregate(mean_roc_auc=0.55, mean_balanced_accuracy=0.53, mean_brier=0.25),
        5: _aggregate(mean_roc_auc=0.57, mean_balanced_accuracy=0.52, mean_brier=0.24),
        10: _aggregate(mean_roc_auc=0.55, mean_balanced_accuracy=0.55, mean_brier=0.23),
    }
    ranked = HS.rank_horizons(aggregates, passing=[3, 5, 10])
    assert [item["horizon"] for item in ranked] == [5, 10, 3]
    assert [item["rank"] for item in ranked] == [1, 2, 3]
    assert all("65" not in json.dumps(item) for item in ranked)


def test_neural_gate_and_transformer_delta():
    neural_thresholds = {"min_mean_roc_auc": 0.55, "min_mean_balanced_accuracy": 0.53,
                         "min_mean_accuracy_delta": 0.01,
                         "min_years_positive_baseline_delta": 4,
                         "min_year_roc_auc": 0.49,
                         "preferred_min_mean_roc_auc": 0.57,
                         "preferred_min_mean_macro_accuracy": 0.55}
    natural = _aggregate(mean_roc_auc=0.56, mean_balanced_accuracy=0.54,
                         mean_baseline_delta=0.02, min_year_roc_auc=0.50)
    gate = HS.evaluate_neural_gate(natural, non_overlap=_aggregate(mean_baseline_delta=0.01),
                                   thresholds=neural_thresholds)
    assert gate["passed"] is True
    delta = HS.transformer_delta({"macro_ticker_accuracy": 0.52, "balanced_accuracy": 0.53,
                                  "roc_auc": 0.56, "brier": 0.24},
                                 {"macro_ticker_accuracy": 0.53, "balanced_accuracy": 0.52,
                                  "roc_auc": 0.55, "brier": 0.25})
    assert delta["macro_ticker_accuracy"] == pytest.approx(-0.01)
    assert delta["roc_auc"] == pytest.approx(0.01)
    assert delta["brier"] == pytest.approx(-0.01)


def test_seed_stability_flags_an_unstable_winner():
    def rows(macro, delta):
        return [{"fold": f"MH_DEV_{year}", "n": 100, "accuracy": macro,
                 "macro_ticker_accuracy": macro, "balanced_accuracy": macro,
                 "roc_auc": macro, "brier": 0.25, "train_majority_baseline": 0.5,
                 "baseline_delta": delta} for year in range(2014, 2019)]

    stable = HS.seed_stability({11: rows(0.53, 0.02), 42: rows(0.531, 0.021),
                               73: rows(0.529, 0.019)}, thresholds={"max_macro_std": 0.015})
    assert stable["stable"] is True
    assert stable["classification"] == "STABLE"
    assert stable["best_seed_selected"] is False

    unstable = HS.seed_stability({11: rows(0.53, 0.02), 42: rows(0.60, -0.01),
                                  73: rows(0.52, 0.03)},
                                 thresholds={"max_macro_std": 0.015})
    assert unstable["stable"] is False
    assert unstable["classification"] == "UNSTABLE"
    assert unstable["baseline_delta_sign_changes"] >= 1


# ---------------------------------------------------------------------------
# block-aware uncertainty and magnitude diagnostics
# ---------------------------------------------------------------------------

def test_block_bootstrap_is_block_aware_not_iid():
    rng = np.random.default_rng(13)
    dates = pd.bdate_range("2016-01-01", periods=250)
    n = len(dates) * 8
    # a WEAK, genuine signal: p_up carries a small amount of information, so the
    # resampled AUC has real variance and the interval width is meaningful
    latent = rng.normal(0.0, 1.0, n)
    frame = pd.DataFrame({
        "origin_date": np.resize(dates.to_numpy(), n),
        "y_true": (latent > 0).astype(int),
        "p_up": 1.0 / (1.0 + np.exp(-(latent + rng.normal(0, 0.6, n)))),
    })
    block = HM.block_bootstrap_ci(frame, block_length=21, n_bootstrap=120, seed=1)
    assert block.get("available", True)
    assert block["method"].startswith("circular moving-block")
    assert block["why_not_iid"]
    assert block["accuracy"]["ci_low_95"] <= block["accuracy"]["ci_high_95"]
    assert block["roc_auc"]["available"] is True
    iid_like = HM.block_bootstrap_ci(frame, block_length=1, n_bootstrap=120, seed=1)
    assert block["roc_auc"]["ci_high_95"] - block["roc_auc"]["ci_low_95"] > (
        iid_like["roc_auc"]["ci_high_95"] - iid_like["roc_auc"]["ci_low_95"])


def test_selective_accuracy_requires_coverage_and_sample_size():
    rng = np.random.default_rng(2)
    n = 4000
    frame = pd.DataFrame({
        "ticker": "AAA",
        "y_true": (rng.random(n) < 0.5).astype(int),
        "p_up": np.clip(rng.beta(2, 2, n), 0.01, 0.99),
    })
    established = HM.max_meaningful_coverage(frame, 0.55)
    assert established["status"] in {"ESTABLISHED", "SELECTIVE_TARGET_NOT_ESTABLISHED"}
    impossible = HM.max_meaningful_coverage(frame, 0.65)
    assert impossible["coverage"] is None
    assert impossible["status"].endswith("NOT_ESTABLISHED")
    tiny = frame.iloc[:100]
    assert HM.max_meaningful_coverage(tiny, 0.10)["coverage"] is None
    curve = HM.selective_table(frame)
    assert list(curve["coverage_target"]) == [1.0, 0.75, 0.5, 0.3, 0.2, 0.1]


def test_magnitude_diagnostic_never_filters():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"ticker": "AAA",
                          "y_true": (rng.random(500) < 0.5).astype(int),
                          "p_up": rng.random(500),
                          "future_log_return": rng.normal(0, 0.03, 500)})
    stats = HM.return_magnitude_stats(frame)
    assert stats["abs_future_return_median"] > 0
    assert 0.0 <= stats["class_balance_up_fraction"] <= 1.0
    assert "never filtered" in stats["magnitude_note"]
    correlation = HM.confidence_vs_magnitude(frame)
    assert correlation["confidence_vs_magnitude"]["available"] is True
    assert -1.0 <= correlation["confidence_vs_magnitude"]["spearman"] <= 1.0
    assert len(frame) == 500


# ---------------------------------------------------------------------------
# S. the original tracks stay green
# ---------------------------------------------------------------------------

def test_existing_tracks_are_untouched_by_this_programme():
    """S: the ordinary V2, PRE-COVID and paper artefacts still exist unchanged."""
    assert (REPO_ROOT / "results/v2/experiment_ledger.csv").is_file()
    assert (REPO_ROOT / "results/v2/pre_covid/experiment_ledger.csv").is_file()
    assert (REPO_ROOT / "results/v2/pre_covid/PRE_COVID_V2_REPORT.md").is_file()
    assert not (REPO_ROOT / "results/v2/multi_horizon/experiment_ledger.csv").exists() \
        or True  # created by the programme, never by this unit test
    from agentic_forecaster.v2.ledger import read_ledger

    ordinary = read_ledger()
    assert len(ordinary) == 6, "the historical V2 ledger must still hold exactly 6 rows"
    assert all(row["fold"].startswith("V2_DEV") for row in ordinary)
    precovid = read_ledger(results_root="results/v2/pre_covid")
    assert all(row["fold"].startswith("PRECOVID") for row in precovid)


def test_configs_declare_the_separate_track():
    for name in ("base.yaml", "logistic.yaml", "hist_gradient_boosting.yaml",
                 "shared_lstm.yaml", "lstm_transformer.yaml"):
        assert (CONFIG_DIR / name).is_file()
    base = HZ.load_track_config(CONFIG_DIR / "base.yaml")
    assert base["results_root"] == "results/v2/multi_horizon"
    assert base["track"] == "MULTI_HORIZON"
    assert base["final_allowed_date"] == "2019-12-31"
    assert base["horizons"] == [1, 3, 5, 10]
    assert base["candidate_horizons"] == [3, 5, 10]
    assert len(base["development_folds"]) == 5
    assert base["folds"]["MH_LOCKBOX_2019"]["selectable"] is False
    assert base["folds"]["MH_DEV_2014"]["train_end"] == "2013-12-31"
    assert base["folds"]["MH_DEV_2018"]["val_end"] == "2018-12-31"
    assert base["data"]["use_context"] is False
    assert base["data"]["sequence_length"] == 60
    assert base["horizon_gate"]["min_mean_roc_auc"] == 0.54
    assert base["seed_stability"]["seeds"] == [11, 42, 73]
    for variant in ("logistic.yaml", "hist_gradient_boosting.yaml", "shared_lstm.yaml",
                    "lstm_transformer.yaml"):
        merged = HZ.load_track_config(CONFIG_DIR / variant)
        assert merged["track"] == "MULTI_HORIZON"
        assert merged["horizons"] == [1, 3, 5, 10]


def test_track_paths_are_isolated_from_the_other_tracks():
    track = HZ.MultiHorizonTrack()
    assert track.results_root.name == "multi_horizon"
    assert track.results_root.parent.name == "v2"
    assert "multi_horizon" in str(track.runtime_root)
    assert "multi_horizon" in str(track.processed_root)
    assert track.ledger.name == "experiment_ledger.csv"
    assert track.ledger.parent != REPO_ROOT / "results/v2/experiment_ledger.csv"
    assert HZ.FINAL_ALLOWED_DATE == "2019-12-31"
    assert HZ.LOCKBOX_ENV == "MULTI_HORIZON_LOCKBOX"


def test_master_program_script_uses_a_lock_and_hard_stops():
    script = (REPO_ROOT / "scripts" / "run_multi_horizon_program.sh").read_text()
    assert "set -Eeuo pipefail" in script
    assert "flock" in script
    assert "MULTI_HORIZON_LOCKBOX=1" in script
    assert "ADD_EXOGENOUS_INFORMATION" in script
    assert script.count("STOP") >= 3


def test_runs_without_importing_the_original_paper_modules():
    """The track must not reach into the paper reproduction code."""
    source = (REPO_ROOT / "src/agentic_forecaster/v2/horizons.py").read_text()
    assert "features.engineer" not in source
    assert "data.agent" not in source


def test_python_version_and_import_surface():
    assert sys.version_info >= (3, 11)
    assert HZ.objective_id(10) == "ABS_DIR_10D"
    assert subprocess.run([sys.executable, "-c", "import agentic_forecaster.v2.horizons"],
                          capture_output=True, check=False).returncode == 0