"""Tests for the performance-recovery framework.

Covers the safety-critical behaviours:

* the test-set firewall blocks 2022/2023 scoring and allows pre-2022
* the search folds are all strictly pre-2022
* the experiment ledger is append-only and refuses a false test claim
* the frozen-config mechanism refuses an edited config
* the final test refuses to run without FINAL_TEST=1
* the Brier definition is mean((p_up - y)^2)
* the CLI train-all --tickers regression

STAGE 0 is exercised end-to-end on SYNTHETIC data so the harness is proven
without touching a real dataset or the network.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RECOVERY = REPO_ROOT / "src" / "agentic_forecaster" / "recovery"
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.recovery import folds, variants
from agentic_forecaster.recovery.calibrators import (
    brier_score,
    fit_calibrator,
)
from agentic_forecaster.recovery.firewall import (
    TEST_FIREWALL_START,
    TestSetFirewallError,
    assert_config_windows_safe,
    assert_labels_not_firewalled,
    assert_pre_test_dates,
    date_in_firewalled_window,
    describe_firewall,
    firewall_guard,
)
from agentic_forecaster.recovery.freeze import (
    FrozenConfigError,
    assert_final_test_allowed,
    final_test_authorised,
    freeze_config,
    verify_frozen_config,
)
from agentic_forecaster.recovery.harness import (
    run_shuffled_label_control,
    run_tiny_overfit,
    save_stage0,
)
from agentic_forecaster.recovery.ledger import (
    LEDGER_COLUMNS,
    append_experiment,
    read_ledger,
)


def FIREWALL():
    """A FRESH search-mode context manager per use.

    A module-level firewall_guard(True) instance is a generator and can only be
    entered once; reusing it silently no-ops and would make every firewall test
    vacuous. Always build a new one.
    """
    return firewall_guard(True)


# ------------------------------------------------------------------ firewall

def test_firewall_start_is_2022():
    assert TEST_FIREWALL_START == pd.Timestamp("2022-01-01")


@pytest.mark.parametrize("date", [
    "2022-01-01", "2022-01-03", "2022-06-30", "2022-12-31",
    "2023-01-02", "2023-12-29", "2024-05-01", "2030-01-01",
])
def test_firewall_blocks_2022_and_later(date):
    assert date_in_firewalled_window(date) is True
    with FIREWALL(), pytest.raises(TestSetFirewallError):
        assert_pre_test_dates([date])


@pytest.mark.parametrize("date", [
    "2000-01-03", "2015-12-31", "2016-01-01", "2019-12-31",
    "2020-01-01", "2021-12-31",
])
def test_firewall_allows_pre_2022(date):
    assert date_in_firewalled_window(date) is False
    with FIREWALL():
        assert_pre_test_dates([date])  # must not raise


def test_firewall_allows_when_search_mode_off():
    assert_pre_test_dates(["2022-06-01"], search=False)  # not search mode


def test_firewall_rejects_paper_test_windows_in_config():
    with FIREWALL():
        with pytest.raises(TestSetFirewallError):
            assert_config_windows_safe({"test_start": "2022-01-01"})
        with pytest.raises(TestSetFirewallError):
            assert_config_windows_safe({"val_start": "2022-01-01"})
        assert_config_windows_safe({"val_start": "2019-01-01", "val_end": "2019-12-31"})


def test_firewall_refuses_labels_without_dates():
    """Without dates we cannot prove labels are pre-2022, so refuse."""
    with FIREWALL(), pytest.raises(TestSetFirewallError, match="without dates"):
        assert_labels_not_firewalled(np.array([0, 1, 1, 0]))


def test_firewall_allows_labels_with_pre2022_dates():
    with FIREWALL():
        assert_labels_not_firewalled(
            np.array([0, 1, 1, 0]), dates=pd.to_datetime(
                ["2019-01-01", "2019-01-02", "2019-01-03", "2019-01-04"]))


def test_firewall_context_manager_restores_environment():
    import os
    before = os.environ.get("RECOVERY_SEARCH_MODE")
    with firewall_guard(True):
        assert os.environ["RECOVERY_SEARCH_MODE"] == "1"
    after = os.environ.get("RECOVERY_SEARCH_MODE")
    assert after == before


def test_firewall_description_lists_protected_years():
    d = describe_firewall()
    assert d["protected_test_years"] == [2022, 2023]


# --------------------------------------------------------------- search folds

def test_three_search_folds_exist_with_the_specified_windows():
    assert set(folds.SEARCH_FOLDS) == {"SEARCH_FOLD_A", "SEARCH_FOLD_B", "SEARCH_FOLD_C"}
    assert folds.SEARCH_FOLDS["SEARCH_FOLD_A"].as_data_cfg() == {
        "train_start": "2016-01-01", "train_end": "2018-12-31",
        "val_start": "2019-01-01", "val_end": "2019-12-31"}
    assert folds.SEARCH_FOLDS["SEARCH_FOLD_B"].as_data_cfg() == {
        "train_start": "2016-01-01", "train_end": "2019-12-31",
        "val_start": "2020-01-01", "val_end": "2020-12-31"}
    assert folds.SEARCH_FOLDS["SEARCH_FOLD_C"].as_data_cfg() == {
        "train_start": "2016-01-01", "train_end": "2020-12-31",
        "val_start": "2021-01-01", "val_end": "2021-12-31"}


@pytest.mark.parametrize("name", sorted(folds.SEARCH_FOLDS))
def test_every_search_fold_is_strictly_pre_2022(name):
    f = folds.get_search_fold(name)
    assert pd.Timestamp(f.val_end) < TEST_FIREWALL_START
    assert pd.Timestamp(f.train_end) < TEST_FIREWALL_START
    folds.assert_fold_is_pre_test(name)


def test_paper_folds_are_the_two_documented_folds():
    names = [f["fold"] for f in folds.PAPER_FOLDS]
    assert names == ["fold_0", "fold_1"]
    assert folds.PAPER_FOLDS[0]["test_start"] == "2022-01-01"
    assert folds.PAPER_FOLDS[1]["test_start"] == "2023-01-01"


def test_fold_config_for_drops_the_test_window():
    base = {"data": {"train_start": "x", "test_start": "2022-01-01",
                     "test_end": "2022-12-31"}, "experiment": {"name": "r"}}
    cfg = folds.fold_config_for("SEARCH_FOLD_C", base)
    assert cfg["data"]["val_start"] == "2021-01-01"
    assert "test_start" not in cfg["data"]
    assert "test_end" not in cfg["data"]
    # the original base config must not be mutated
    assert base["data"]["test_start"] == "2022-01-01"


# ------------------------------------------------------------------- ledger

def test_ledger_columns_match_the_specification():
    expected = [
        "experiment_id", "timestamp", "git_commit", "dataset_variant",
        "dataset_hash", "universe_id", "config_hash", "search_fold",
        "ticker_subset", "feature_set", "rsi_method", "lookback", "scaler",
        "hidden_size", "layers", "dropout", "max_epochs", "best_epoch",
        "patience", "weight_decay", "class_weighting", "calibration_method",
        "seed", "train_loss", "validation_loss", "validation_accuracy",
        "validation_f1", "validation_brier", "validation_ece",
        "test_evaluated", "notes",
    ]
    assert list(LEDGER_COLUMNS) == expected


def test_ledger_appends_and_keeps_poor_experiments(tmp_path):
    p = tmp_path / "experiment_ledger.csv"
    with firewall_guard(True):
        append_experiment({"search_fold": "SEARCH_FOLD_C", "feature_set": "F1",
                           "max_epochs": 3, "validation_accuracy": 0.50}, path=p)
        append_experiment({"search_fold": "SEARCH_FOLD_C", "feature_set": "F2",
                           "max_epochs": 100, "validation_accuracy": 0.49}, path=p)
    rows = read_ledger(p)
    assert len(rows) == 2, "poor experiments must never be dropped"
    assert rows[0]["feature_set"] == "F1" and rows[1]["feature_set"] == "F2"
    assert all(r["test_evaluated"] == "false" for r in rows)


def test_ledger_refuses_a_search_row_claiming_test_evaluation(tmp_path):
    p = tmp_path / "experiment_ledger.csv"
    with firewall_guard(True), pytest.raises(ValueError, match="test_evaluated"):
        append_experiment({"search_fold": "SEARCH_FOLD_C",
                           "test_evaluated": "true"}, path=p)
    assert read_ledger(p) == []


def test_ledger_rows_carry_an_id_and_git_commit(tmp_path):
    p = tmp_path / "experiment_ledger.csv"
    row = append_experiment({"search_fold": "SEARCH_FOLD_A"}, path=p)
    assert row["experiment_id"].startswith("EXP-")
    assert row["timestamp"]
    assert row["git_commit"]


# ------------------------------------------------------------------ variants

def test_feature_families_are_the_four_specified():
    assert set(variants.FEATURE_FAMILIES) == {"F0", "F1", "F2", "F3"}
    assert variants.FEATURE_FAMILIES["F0"]["indicators"] == ()
    f1 = set(variants.FEATURE_FAMILIES["F1"]["indicators"])
    f2 = set(variants.FEATURE_FAMILIES["F2"]["indicators"])
    assert f1 < f2
    for extra in ("sma_5", "sma_20", "bb_percent_b", "obv", "bb_upper", "bb_lower"):
        assert extra in f2, extra


def test_f2_and_f3_indicators_all_exist_in_the_engine():
    from agentic_forecaster.features import engineer
    for fam in ("F1", "F2", "F3"):
        for name in variants.build_feature_indicators(fam, "R1_wilder"):
            # authoritative check: the engine raises on an unknown indicator
            df = pd.DataFrame({
                "date": pd.date_range("2021-01-01", periods=60, freq="D"),
                "open": np.linspace(100, 110, 60), "high": np.linspace(101, 111, 60),
                "low": np.linspace(99, 109, 60), "close": np.linspace(100, 110, 60),
                "volume": np.full(60, 1000.0)})
            engineer.build_feature_frame(df, indicators=[name], use_ohlcv=True)


def test_rsi_variants_replace_rather_than_accumulate():
    for method, indicator in (("R0_rolling", "rsi_14_rolling"),
                              ("R1_wilder", "rsi_14"),
                              ("R2_ema", "rsi_14_ema")):
        names = variants.build_feature_indicators("F2", method)
        assert indicator in names, method
        assert not any(n.startswith("rsi_14_") for n in names if n != indicator)
        assert len([n for n in names if "rsi" in n]) == 1


def test_rsi_variants_differ_and_are_causal():
    from agentic_forecaster.features.engineer import rsi_ema, rsi_rolling, rsi_wilder
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    r0, r1, r2 = rsi_rolling(close), rsi_wilder(close), rsi_ema(close)
    assert r0.notna().any() and r1.notna().any() and r2.notna().any()
    assert not np.allclose(r0.dropna().to_numpy(), r1.dropna().to_numpy())
    # causality: truncating the future must not change past values
    assert np.allclose(r1.iloc[:150].to_numpy(), rsi_wilder(close.iloc[:150]).to_numpy())


def test_training_length_and_lookback_catalogues():
    assert set(variants.TRAINING_LENGTHS) == {"T3", "T30", "T50", "T100"}
    for key in ("T30", "T50", "T100"):
        assert variants.TRAINING_LENGTHS[key]["patience"] == 10
        assert variants.TRAINING_LENGTHS[key]["restore_best_checkpoint"] is True
        assert variants.TRAINING_LENGTHS[key]["max_epochs"] > 3
    assert variants.LOOKBACKS == {"L10": 10, "L20": 20, "L30": 30, "L40": 40, "L60": 60}
    assert variants.DEFAULT_LOOKBACK == 30


def test_scaler_calibration_architecture_catalogues():
    assert set(variants.SCALERS) == {"standard", "minmax", "robust"}
    assert set(variants.CALIBRATION_METHODS) == {"none", "temperature", "platt", "isotonic"}
    assert set(variants.ARCHITECTURES) == {"A0", "A1", "A2", "A3"}
    assert variants.ARCHITECTURES["A0"] == {"num_layers": 2, "hidden_size": 64,
                                             "description": "reference"}
    assert set(variants.WEIGHT_DECAYS) == {0.0, 1e-6, 1e-5, 1e-4, 1e-3}
    assert set(variants.DROPOUTS) == {0.0, 0.2, 0.5}
    assert variants.SEEDS == (11, 23, 42, 73, 101)


def test_stage_plan_is_staged_not_cartesian():
    plan = variants.stage_plan()
    assert [s["stage"] for s in plan] == [
        "STAGE_0", "STAGE_A", "STAGE_B", "STAGE_C", "STAGE_D", "STAGE_E"]
    assert all("gate" in s for s in plan)


# --------------------------------------------------------------- calibration

def test_brier_is_mean_squared_up_probability_error():
    y = [1, 0, 1, 1]
    p = [0.9, 0.2, 0.6, 0.4]
    expected = float(np.mean([(0.9 - 1) ** 2, (0.2 - 0) ** 2,
                              (0.6 - 1) ** 2, (0.4 - 1) ** 2]))
    assert brier_score(y, p) == pytest.approx(expected)


def test_brier_is_not_directional_confidence_after_flipping():
    """Scoring a flipped DOWN confidence must give a different (wrong) number."""
    y = [1, 0, 1, 0]
    p_up = [0.8, 0.3, 0.7, 0.2]
    p_down_flipped = 1.0 - np.array(p_up)   # directional confidence
    correct = brier_score(y, p_up)
    flipped = brier_score(y, p_down_flipped)
    assert correct < flipped
    assert correct != flipped


def test_brier_matches_sklearn():
    from sklearn.metrics import brier_score_loss
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    assert brier_score(y, p) == pytest.approx(float(brier_score_loss(y, p)))


def test_calibration_none_is_identity():
    cal = fit_calibrator("none", [0.2, 0.8], [0, 1])
    assert cal.predict(np.array([0.2, 0.8])).tolist() == [0.2, 0.8]


def test_calibration_refuses_firewalled_dates():
    with FIREWALL(), pytest.raises(TestSetFirewallError):
        fit_calibrator("temperature", [0.2, 0.8], [0, 1],
                       dates=pd.to_datetime(["2022-03-01", "2022-03-02"]))


@pytest.mark.parametrize("method", ["temperature", "platt", "isotonic"])
def test_all_calibration_methods_fit_and_predict(method):
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.4 + 0.3 + rng.normal(0, 0.15, 300), 0.01, 0.99)
    cal = fit_calibrator(method, p, y)
    out = cal.predict(p)
    assert out.shape == (300,)
    assert np.all((out >= 0) & (out <= 1))


# ----------------------------------------------------------- freeze / final

def test_final_test_requires_the_env_var():
    assert final_test_authorised({}) is False
    assert final_test_authorised({"FINAL_TEST": "0"}) is False
    assert final_test_authorised({"FINAL_TEST": "1"}) is True


def test_final_test_refuses_without_freeze(tmp_path):
    cfg = tmp_path / "recovered_paper.yaml"
    cfg.write_text("data: {}\n")
    with pytest.raises(FrozenConfigError, match="No frozen configuration"):
        assert_final_test_allowed(cfg, env={"FINAL_TEST": "1"}, root=tmp_path)


def test_final_test_refuses_without_final_test_1(tmp_path):
    cfg = tmp_path / "configs" / "recovered_paper.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("data: {}\n")
    freeze_config(cfg, dataset_variant="unadjusted", dataset_manifest=None,
                  universe_id="U", validation_metrics={}, supporting_experiment_ids=[],
                  root=tmp_path)
    with pytest.raises(FrozenConfigError, match="FINAL_TEST=1"):
        assert_final_test_allowed(cfg, env={}, root=tmp_path)


def test_frozen_config_hash_is_enforced(tmp_path):
    cfg = tmp_path / "configs" / "recovered_paper.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("models:\n  attention_lstm:\n    max_epochs: 100\n")
    manifest = freeze_config(cfg, dataset_variant="unadjusted", dataset_manifest=None,
                             universe_id="PAPER_CANDIDATE", validation_metrics={"acc": 0.55},
                             supporting_experiment_ids=["EXP-1"], root=tmp_path)
    assert manifest["config_sha256"]
    assert manifest["git_commit"]
    # unchanged config verifies
    assert verify_frozen_config(cfg, root=tmp_path)["config_sha256"]
    # a single-character edit is rejected
    cfg.write_text("models:\n  attention_lstm:\n    max_epochs: 101\n")
    with pytest.raises(FrozenConfigError, match="CHANGED"):
        verify_frozen_config(cfg, root=tmp_path)
    with pytest.raises(FrozenConfigError, match="CHANGED"):
        assert_final_test_allowed(cfg, env={"FINAL_TEST": "1"}, root=tmp_path)


# ---------------------------------------------------------------- STAGE 0

def _synthetic_sequences(n=600, seq=30, features=4, seed=0, strength=2.5):
    """A learnable synthetic signal: next-day up-ness depends on the last bar.

    ``strength`` controls separability; the default is strong enough that a
    correctly wired model separates real labels from shuffled labels, which is
    what the control test needs.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, seq, features)).astype(np.float32)
    logit = X[:, -1, :].sum(axis=1) * strength
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(np.float32)
    return X, y


def test_tiny_overfit_passes_on_synthetic_data():
    X, y = _synthetic_sequences(n=120, seed=1)
    with firewall_guard(True):
        res = run_tiny_overfit(X, y, X, y, epochs=250, patience=1000,
                               max_train_rows=64, hidden_size=8, num_layers=1,
                               batch_size=16, seed=42)
    assert res.passed, res.diagnostics
    assert res.best_train_accuracy >= 0.95, res.best_train_accuracy
    assert res.best_train_loss < 0.5, res.best_train_loss


def test_shuffled_label_control_passes_on_synthetic_data():
    X, y = _synthetic_sequences(n=800, seed=2)
    Xtr, ytr = X[:600], y[:600]
    Xva, yva = X[600:], y[600:]
    with firewall_guard(True):
        res = run_shuffled_label_control(Xtr, ytr, Xva, yva, epochs=25,
                                         hidden_size=8, num_layers=1,
                                         batch_size=32, seed=42)
    assert res.passed, res.diagnostics
    assert 0.3 <= res.validation_accuracy <= 0.7, res.validation_accuracy


def test_real_labels_beat_shuffled_labels():
    """The control's whole purpose: real labels must win."""
    X, y = _synthetic_sequences(n=1200, seed=3)
    Xtr, ytr, Xva, yva = X[:800], y[:800], X[800:], y[800:]
    with firewall_guard(True):
        real = run_tiny_overfit(Xtr, ytr, Xva, yva, epochs=40, patience=20,
                                hidden_size=8, num_layers=1, batch_size=32,
                                seed=42, min_train_accuracy=0.0,
                                name="real_labels")
        shuffled = run_shuffled_label_control(Xtr, ytr, Xva, yva, epochs=40,
                                              hidden_size=8, num_layers=1,
                                              batch_size=32, seed=42)
    assert real.validation_accuracy > shuffled.validation_accuracy


def test_stage0_results_are_saved(tmp_path):
    X, y = _synthetic_sequences(n=100, seed=4)
    with firewall_guard(True):
        a = run_tiny_overfit(X, y, X, y, epochs=120, patience=1000,
                             max_train_rows=48, hidden_size=8, num_layers=1,
                             batch_size=16, seed=42)
        b = run_shuffled_label_control(X, y, X, y, epochs=10, hidden_size=8,
                                       num_layers=1, batch_size=16, seed=42)
    payload = save_stage0([a, b], tmp_path)
    assert (tmp_path / "tiny_overfit.json").is_file()
    assert (tmp_path / "shuffled_label_control.json").is_file()
    assert (tmp_path / "stage0_summary.json").is_file()
    assert payload["stage"] == "STAGE_0"
    assert set(payload["results"]) == {"tiny_overfit", "shuffled_label_control"}


def test_harness_refuses_firewalled_validation_dates():
    X, y = _synthetic_sequences(n=60, seed=5)
    bad_dates = pd.to_datetime(["2022-01-03", "2022-01-04"])
    with pytest.raises(TestSetFirewallError):
        run_tiny_overfit(X, y, X, y, val_dates=bad_dates, epochs=1)
    with pytest.raises(TestSetFirewallError):
        run_shuffled_label_control(X, y, X, y, val_dates=bad_dates, epochs=1)


# ------------------------------------------------------------------- the CLI

def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "agentic_forecaster", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600, check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(REPO_ROOT / "src")})


def test_train_all_parser_defines_tickers():
    """Regression: _cmd_train_all read args.tickers, which argparse never set."""
    out = _run_cli("train-all", "--help")
    assert out.returncode == 0, out.stderr
    assert "--tickers" in out.stdout


def test_train_all_enters_the_command_body(tmp_path):
    """A parser-help check is not enough: actually reach _cmd_train_all."""
    import yaml

    from agentic_forecaster.cli import build_parser
    from agentic_forecaster.cli import main as cli_main

    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(0)
    n = 400
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="B").astype(str),
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": np.full(n, 1000.0),
    }).to_csv(raw / "TINY_minute.csv", index=False)

    cfg = {
        "experiment": {"name": "t", "seed": 42, "output_dir": str(tmp_path / "out")},
        "data": {"raw_root": str(raw), "synthetic": False,
                 "processed_root": str(tmp_path / "proc"),
                 "sequence_length": 10,
                 "train_start": "2020-01-01", "train_end": "2020-06-30",
                 "val_start": "2020-07-01", "val_end": "2020-09-30",
                 "test_start": "2020-10-01", "test_end": "2020-12-31"},
        "features": {"use_ohlcv": True, "indicators": ["log_return"], "drop_na": True},
        "models": {"attention_lstm": {"hidden_size": 4, "num_layers": 1,
                                       "dropout": 0.0, "max_epochs": 1,
                                       "patience": 1, "batch_size": 16}},
        "evaluation": {"metrics": ["accuracy"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    parser = build_parser()
    # --tickers must parse into args.tickers (the bug)
    args = parser.parse_args(["train-all", "--config", str(cfg_path),
                              "--tickers", "TINY"])
    assert args.tickers == "TINY"
    assert args.baselines is False
    # and the command body must accept it without AttributeError
    code = cli_main(["train-all", "--config", str(cfg_path), "--tickers", "TINY"])
    assert code in (0, 1), f"unexpected exit {code}"


# --------------------------------------------------- recovery scripts --help

@pytest.mark.parametrize("script", [
    "recovery_audit.py", "run_reproduction_search.py", "freeze_recovered_config.py",
    "run_recovered_paper.py", "compare_reproduction.py",
])
def test_recovery_script_help(script):
    out = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / script), "--help"],
                         cwd=REPO_ROOT, capture_output=True, text=True, timeout=600, check=False)
    assert out.returncode == 0, out.stderr


def test_run_recovered_paper_refuses_without_final_test():
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_recovered_paper.py"),
         "--config", "configs/recovered_paper.yaml"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600, check=False)
    assert out.returncode != 0
    combined = out.stdout + out.stderr
    assert "FINAL_TEST=1" in combined or "frozen" in combined.lower()


def test_search_dry_run_resolves_and_is_firewalled():
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_reproduction_search.py"),
         "--config", "configs/reproduction_search/paper_snapshot_2025_unadjusted_perf.yaml",
         "--search-fold", "SEARCH_FOLD_C", "--training-length", "T30",
         "--feature-family", "F2", "--dry-run"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600, check=False)
    assert out.returncode == 0, out.stderr
    assert '"max_epochs": 30' in out.stdout
    assert "REMOVED" in out.stdout
    assert "DRY RUN" in out.stdout
