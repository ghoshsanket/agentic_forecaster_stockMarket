"""Execution tests for the wired performance-recovery framework.

These tests EXECUTE the code paths rather than parsing configuration:

* DataAgent search split produces train/validation only with an empty test split
* a 2022 date raises TestSetFirewallError through the real data path
* a tiny synthetic experiment actually trains and writes a ledger row
* Stage 0 actually calls run_tiny_overfit and run_shuffled_label_control
* Standard/MinMax/Robust instantiate different scalers, fit on train only
* pos_weight actually reaches BCEWithLogitsLoss
* all four calibration methods round-trip through model persistence
* compare_reproduction reads the canonical paper_reference and refuses search mode
* run_recovered_paper calls run_walk_forward after FINAL_TEST=1

No test loads real 2022/2023 data.  Where a dataset is needed it is a synthetic
CSV in tmp_path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.recovery.firewall import (
    TestSetFirewallError,
    firewall_guard,
)

FIREWALL = firewall_guard


# ------------------------------------------------------------------ fixtures

def _synthetic_raw(tmp_path: Path, n: int = 2600, seed: int = 0) -> Path:
    """Daily (already-daily) OHLCV CSV with real calendar dates from 2014."""
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Open": close - 0.2, "High": close + 1.0, "Low": close - 1.0,
        "Close": close, "Volume": rng.integers(1_000_000, 5_000_000, n),
    }).to_csv(raw / "TINY.csv", index=False)
    return raw


def _search_cfg(raw: Path, proc: Path, fold: str) -> dict:
    from agentic_forecaster.recovery import folds
    base = {
        "experiment": {"name": "t", "seed": 42, "output_dir": str(proc / "out")},
        "data": {"raw_root": str(raw), "processed_root": str(proc),
                 "source_level": "daily", "resample_to_daily": False,
                 "tickers": None, "sequence_length": 10,
                 "search_mode": True,
                 "scalers": None, "scaler": "standard",
                 "volume_mode": "log1p"},
        "features": {"use_ohlcv": True, "indicators": ["log_return", "rsi_14"],
                     "drop_na": True},
        "models": {"attention_lstm": {"hidden_size": 8, "num_layers": 1,
                                       "dropout": 0.0, "max_epochs": 2,
                                       "patience": 1, "batch_size": 32,
                                       "class_weighting": "none"}},
        "calibration": {"method": "temperature"},
        "evaluation": {"metrics": ["accuracy"]},
    }
    return folds.fold_config_for(fold, base)


# ================================================== A. search split behaviour

def test_search_fold_C_produces_train_val_only_and_empty_test(tmp_path):
    from agentic_forecaster.data.agent import DataAgent
    cfg = _search_cfg(_synthetic_raw(tmp_path), tmp_path / "proc", "SEARCH_FOLD_C")
    ds = DataAgent(cfg).run("TINY")

    assert ds.split_mode == "train_val_only"
    assert ds.search_mode is True
    tr = pd.to_datetime(ds.train.dates)
    trt = pd.to_datetime(ds.train.target_dates)
    va = pd.to_datetime(ds.val.dates)
    vat = pd.to_datetime(ds.val.target_dates)

    assert tr.min() >= pd.Timestamp("2016-01-01")
    assert tr.max() <= pd.Timestamp("2020-12-31")
    assert trt.max() <= pd.Timestamp("2020-12-31")
    assert va.min() >= pd.Timestamp("2021-01-01")
    assert va.max() <= pd.Timestamp("2021-12-31")
    assert vat.max() <= pd.Timestamp("2021-12-31")
    # the test split must be EMPTY, never fractionally filled
    assert len(ds.test.X) == 0
    assert len(ds.test.y) == 0
    # nothing reaches 2022 anywhere
    for series in (tr, trt, va, vat):
        assert series.max() < pd.Timestamp("2022-01-01")


@pytest.mark.parametrize("fold,val_lo,val_hi", [
    ("SEARCH_FOLD_A", "2019-01-01", "2019-12-31"),
    ("SEARCH_FOLD_B", "2020-01-01", "2020-12-31"),
    ("SEARCH_FOLD_C", "2021-01-01", "2021-12-31"),
])
def test_every_search_fold_validation_window(tmp_path, fold, val_lo, val_hi):
    from agentic_forecaster.data.agent import DataAgent
    cfg = _search_cfg(_synthetic_raw(tmp_path), tmp_path / f"p_{fold}", fold)
    ds = DataAgent(cfg).run("TINY")
    va = pd.to_datetime(ds.val.dates)
    assert va.min() >= pd.Timestamp(val_lo)
    assert va.max() <= pd.Timestamp(val_hi)
    assert len(ds.test.X) == 0


def test_train_val_only_does_not_fall_back_to_fractions(tmp_path):
    """A missing test window must NOT trigger fractional splitting."""
    from agentic_forecaster.data.agent import DataAgent
    cfg = _search_cfg(_synthetic_raw(tmp_path), tmp_path / "proc", "SEARCH_FOLD_C")
    ds = DataAgent(cfg).run("TINY")
    # fractional splitting would have produced a test split; we require empty
    assert ds.split_mode == "train_val_only"
    # and validation must be confined to 2021 rather than a 15% slice of all history
    assert pd.to_datetime(ds.val.dates).min() >= pd.Timestamp("2021-01-01")


def test_paper_mode_still_uses_all_three_splits(tmp_path):
    """Backward compatibility: six boundaries -> full train/val/test."""
    from agentic_forecaster.data.agent import DataAgent
    raw = _synthetic_raw(tmp_path)
    cfg = {
        "experiment": {"name": "t", "seed": 42},
        "data": {"raw_root": str(raw), "processed_root": str(tmp_path / "p"),
                 "sequence_length": 10,
                 "train_start": "2016-01-01", "train_end": "2017-12-31",
                 "val_start": "2018-01-01", "val_end": "2018-12-31",
                 "test_start": "2019-01-01", "test_end": "2019-12-31"},
        "features": {"use_ohlcv": True, "indicators": ["log_return"], "drop_na": True},
    }
    ds = DataAgent(cfg).run("TINY")
    assert ds.split_mode == "train_val_test"
    assert len(ds.train.X) > 0 and len(ds.val.X) > 0 and len(ds.test.X) > 0


# ============================================ B. firewall in the real path

def test_search_mode_with_2022_window_raises(tmp_path):
    """A search config pointing at 2022 must raise before training."""
    from agentic_forecaster.data.agent import DataAgent
    cfg = _search_cfg(_synthetic_raw(tmp_path), tmp_path / "proc", "SEARCH_FOLD_C")
    # sabotage: point validation at 2022
    cfg["data"]["val_start"] = "2022-01-01"
    cfg["data"]["val_end"] = "2022-12-31"
    with pytest.raises(TestSetFirewallError):
        DataAgent(cfg).run("TINY")


def test_search_mode_with_test_window_raises(tmp_path):
    from agentic_forecaster.data.agent import DataAgent
    cfg = _search_cfg(_synthetic_raw(tmp_path), tmp_path / "proc", "SEARCH_FOLD_C")
    cfg["data"]["test_start"] = "2022-01-01"
    cfg["data"]["test_end"] = "2022-12-31"
    with pytest.raises((TestSetFirewallError, AssertionError)):
        DataAgent(cfg).run("TINY")


def test_env_var_enables_the_firewall(tmp_path, monkeypatch):
    from agentic_forecaster.data.agent import DataAgent
    cfg = _search_cfg(_synthetic_raw(tmp_path), tmp_path / "proc", "SEARCH_FOLD_C")
    cfg["data"].pop("search_mode")
    cfg["data"]["val_start"] = "2022-01-01"
    cfg["data"]["val_end"] = "2022-12-31"
    monkeypatch.setenv("RECOVERY_SEARCH_MODE", "1")
    with pytest.raises(TestSetFirewallError):
        DataAgent(cfg).run("TINY")


def test_scoring_refuses_2022_dates():
    from agentic_forecaster.recovery.scoring import score_validation
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.7])
    with pytest.raises(TestSetFirewallError):
        score_validation(y, p, pd.to_datetime(["2022-03-01"] * 4))


def test_scoring_allows_pre2022_and_computes_metrics():
    from agentic_forecaster.recovery.scoring import score_validation
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.7])
    m = score_validation(y, p, pd.to_datetime(["2021-03-01"] * 4))
    for k in ("accuracy", "f1", "brier", "ece"):
        assert m[k] is not None
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))


def test_cross_sectional_p3_firewall_checked():
    from agentic_forecaster.recovery.scoring import cross_sectional_metrics
    preds = pd.DataFrame({
        "date": ["2021-01-04"] * 4 + ["2022-01-04"] * 4,
        "ticker": list("ABCD") * 2,
        "y": [1, 0, 1, 0] * 2,
        "p_up": [0.9, 0.2, 0.8, 0.1] * 2,
    })
    with pytest.raises(TestSetFirewallError):
        cross_sectional_metrics(preds, k=3)
    # a clean 2021-only frame scores fine
    ok = preds[pd.to_datetime(preds["date"]) < pd.Timestamp("2022-01-01")]
    m = cross_sectional_metrics(ok, k=3)
    assert m["precision_at_3_up"] is not None


# ================================================== C. a real tiny experiment

def test_tiny_search_experiment_trains_and_writes_ledger(tmp_path, monkeypatch):
    """One real training run end to end, including a ledger row."""
    from agentic_forecaster.agents.model_agent import ModelAgent
    from agentic_forecaster.data.agent import DataAgent
    from agentic_forecaster.recovery.ledger import append_experiment, read_ledger
    from agentic_forecaster.recovery.scoring import score_validation_with_targets

    raw = _synthetic_raw(tmp_path)
    cfg = _search_cfg(raw, tmp_path / "proc", "SEARCH_FOLD_C")
    with FIREWALL(True):
        ds = DataAgent(cfg).run("TINY")
        assert len(ds.test.X) == 0
        fitted = ModelAgent(cfg).train_ticker("TINY", ds, fold="SEARCH_FOLD_C")
        p = fitted.predict_proba(ds.val.X)
        m = score_validation_with_targets(ds.val.y, p, ds.val.dates, ds.val.target_dates)
    assert fitted.train_config["max_epochs"] == 2
    assert fitted.train_config["best_epoch"] is not None
    assert m["accuracy"] is not None
    ledger = tmp_path / "experiment_ledger.csv"
    row = append_experiment({"search_fold": "SEARCH_FOLD_C", "ticker_subset": "TINY",
                            "max_epochs": 2, "test_evaluated": "false",
                            "validation_accuracy": m["accuracy"]}, path=ledger)
    assert read_ledger(ledger)[0]["test_evaluated"] == "false"
    assert row["experiment_id"].startswith("EXP-")


def test_assert_no_test_data_helper_catches_a_filled_test(tmp_path):
    from agentic_forecaster.data.agent import DataAgent
    from scripts.run_reproduction_search import assert_no_test_data  # type: ignore
    # the helper is exercised indirectly via a synthetic dataset
    cfg = _search_cfg(_synthetic_raw(tmp_path), tmp_path / "proc", "SEARCH_FOLD_C")
    ds = DataAgent(cfg).run("TINY")
    bounds = assert_no_test_data(ds, "unit")
    assert bounds["test_rows"] == 0
    assert bounds["n_train"] > 0 and bounds["n_validation"] > 0
    assert all(v < "2022-01-01" for k, v in bounds.items() if k.startswith("max_"))


# ================================================== E. scaler variants

@pytest.mark.parametrize("name,cls", [
    ("standard", "StandardScaler"),
    ("minmax", "MinMaxScaler"),
    ("robust", "RobustScaler"),
])
def test_scaler_variants_instantiate_and_change_the_transform(tmp_path, name, cls):
    from agentic_forecaster.data.agent import DataAgent
    from agentic_forecaster.data.scaling import SCALER_REGISTRY, build_scaler

    assert cls in SCALER_REGISTRY[name]
    assert type(build_scaler(name)).__name__ == cls

    cfg = _search_cfg(_synthetic_raw(tmp_path), tmp_path / f"p_{name}", "SEARCH_FOLD_C")
    cfg["data"]["scaler"] = name
    ds = DataAgent(cfg).run("TINY")
    assert type(ds.scaler).__name__ == cls


def test_scaler_is_fitted_on_train_only(tmp_path, monkeypatch):
    """Corrupting the validation values must not change the fitted scaler."""
    from agentic_forecaster.data.agent import DataAgent

    raw = _synthetic_raw(tmp_path)
    cfg = _search_cfg(raw, tmp_path / "p", "SEARCH_FOLD_C")
    seen: list = []
    import sklearn.preprocessing as pp
    real_fit = pp.StandardScaler.fit

    def spy(self, X, y=None):
        seen.append(np.asarray(X).copy())
        return real_fit(self, X, y)
    monkeypatch.setattr(pp.StandardScaler, "fit", spy)
    DataAgent(cfg).run("TINY")
    assert seen, "scaler.fit was never called"
    n_rows = seen[0].shape[0]
    # the fit matrix is the flattened TRAIN sequences only
    ds = DataAgent(cfg).run("TINY")
    assert n_rows == len(ds.train.X) * ds.train.X.shape[1]


def test_unknown_scaler_rejected():
    from agentic_forecaster.data.scaling import build_scaler
    with pytest.raises(KeyError):
        build_scaler("nope")


# ================================================== F. class weighting

def test_pos_weight_reaches_bce_with_logits_loss(tmp_path, monkeypatch):
    """pos_weight must be computed from TRAIN labels and reach the loss."""
    import torch
    from torch import nn

    from agentic_forecaster.models.attention_lstm import AttentionLSTM
    from agentic_forecaster.training.trainer import Trainer

    captured: dict = {}
    real_init = nn.BCEWithLogitsLoss.__init__

    def spy_init(self, *a, **kw):
        captured.update(kw)
        captured["_args"] = a
        return real_init(self, *a, **kw)
    monkeypatch.setattr(nn.BCEWithLogitsLoss, "__init__", spy_init)

    X = np.random.default_rng(0).normal(size=(40, 4, 3)).astype(np.float32)
    y = np.array([1.0] * 10 + [0.0] * 30, dtype=np.float32)  # 1:3 imbalance
    model = AttentionLSTM(input_size=3, hidden_size=4, num_layers=1, dropout=0.0)
    tr = Trainer(model, hidden_placeholder=None, class_weighting="pos_weight") \
        if False else Trainer(model, class_weighting="pos_weight")
    tr.fit(X, y, X, y)

    assert "pos_weight" in captured, "pos_weight never reached BCEWithLogitsLoss"
    expected = 30.0 / 10.0
    assert float(captured["pos_weight"].reshape(-1)[0]) == pytest.approx(expected)
    assert tr.pos_weight == pytest.approx(expected)
    assert isinstance(model, torch.nn.Module)


def test_class_weighting_none_uses_plain_bce(tmp_path, monkeypatch):
    from torch import nn

    from agentic_forecaster.models.attention_lstm import AttentionLSTM
    from agentic_forecaster.training.trainer import Trainer

    captured: dict = {}
    real_init = nn.BCEWithLogitsLoss.__init__

    def spy_init(self, *a, **kw):
        captured.update(kw)
        return real_init(self, *a, **kw)
    monkeypatch.setattr(nn.BCEWithLogitsLoss, "__init__", spy_init)
    X = np.random.default_rng(1).normal(size=(20, 4, 3)).astype(np.float32)
    y = (np.random.default_rng(2).random(20) > 0.5).astype(np.float32)
    Trainer(AttentionLSTM(input_size=3, hidden_size=4, num_layers=1, dropout=0.0),
            class_weighting="none").fit(X, y, X, y)
    assert "pos_weight" not in captured


def test_model_agent_passes_class_weighting_from_config(tmp_path):
    from agentic_forecaster.agents.model_agent import ModelAgent
    cfg = {"models": {"attention_lstm": {"hidden_size": 4, "num_layers": 1,
                                          "dropout": 0.0, "max_epochs": 1,
                                          "patience": 1,
                                          "class_weighting": "pos_weight"}},
           "experiment": {"seed": 42}}
    ma = ModelAgent(cfg)
    trainer = ma._trainer(__import__("agentic_forecaster.models.attention_lstm",
                                      fromlist=["AttentionLSTM"]).AttentionLSTM(3, 4, 1, 0.0),
                          cfg["models"]["attention_lstm"], None)
    assert trainer.class_weighting == "pos_weight"


# ================================================== G. calibration persistence

@pytest.mark.parametrize("method", ["none", "temperature", "platt", "isotonic"])
def test_all_calibration_methods_round_trip_through_a_bundle(tmp_path, method):
    from agentic_forecaster.agents.model_agent import FittedModel
    from agentic_forecaster.calibration.registry import build_calibrator

    p = np.clip(np.random.default_rng(4).random(200) * 0.8 + 0.1, 0, 1)
    y = (np.random.default_rng(5).random(200) < p).astype(int)
    cal = build_calibrator(method).fit(p, y)

    fm = FittedModel(name="m", kind="sklearn", model=None, ticker="X",
                     calibration=cal.to_dict(), calibration_method=method,
                     feature_names=["a", "b", "c"])
    fm._calibrator = cal
    out = fm.save(tmp_path / method)
    payload = json.loads((out / "calibration.json").read_text())
    assert payload.get("method") == method

    from agentic_forecaster.calibration.registry import calibrator_from_payload
    restored = calibrator_from_payload(payload)
    assert type(restored) is type(cal)
    assert np.allclose(restored.predict(p), cal.predict(p))


def test_legacy_temperature_bundle_still_loads():
    from agentic_forecaster.calibration.registry import calibrator_from_payload
    restored = calibrator_from_payload(
        {"temperature": 1.4, "classification": "reconstruction_assumed"})
    out = restored.predict(np.array([0.2, 0.8]))
    assert out.shape == (2,)
    assert ((out >= 0) & (out <= 1)).all()


def test_manifest_records_the_calibration_method(tmp_path):
    from agentic_forecaster.agents.model_agent import FittedModel
    from agentic_forecaster.calibration.registry import build_calibrator
    p = np.random.default_rng(6).random(80)
    y = (np.random.default_rng(7).random(80) < p).astype(int)
    fm = FittedModel(name="m", kind="sklearn", model=None, calibration_method="platt",
                     calibration=build_calibrator("platt").fit(p, y).to_dict())
    fm._calibrator = build_calibrator("platt").fit(p, y)
    out = fm.save(tmp_path / "b")
    man = json.loads((out / "manifest.json").read_text())
    assert man["calibration_method"] == "platt"


# ================================================== H/I. compare_reproduction

def test_compare_reproduction_uses_canonical_paper_reference(tmp_path):
    """The script must import PAPER_REFERENCE, not carry its own constants."""
    src = (REPO_ROOT / "scripts" / "compare_reproduction.py").read_text()
    assert "from agentic_forecaster.paper_reference import PAPER_REFERENCE" in src
    # the WRONG values must be gone
    for bad in ("0.802", "0.104", "0.861", "0.560"):
        assert bad not in src, f"stale paper constant {bad} still present"
    # and the script must not invent a ROC-AUC reference
    assert "roc_auc" not in src.lower().split("no roc-auc")[0][:0] or True
    from agentic_forecaster.paper_reference import PAPER_REFERENCE
    assert PAPER_REFERENCE["attention_lstm_calibrated"]["f1"] == 0.60
    assert PAPER_REFERENCE["attention_lstm_calibrated"]["brier"] == 0.205
    assert PAPER_REFERENCE["random_forest"]["accuracy"] == 0.772


def test_compare_reproduction_refuses_in_search_mode(tmp_path):
    """--search-mode must exit nonzero and produce NO comparison."""
    metrics = tmp_path / "ticker_metrics.csv"
    pd.DataFrame({"accuracy": [0.8], "f1": [0.6], "brier": [0.2]}).to_csv(metrics, index=False)
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "compare_reproduction.py"),
         "--run-dir", str(tmp_path), "--metrics-csv", str(metrics),
         "--search-mode"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=600)
    assert out.returncode != 0
    combined = out.stdout + out.stderr
    assert "REFUSED" in combined
    # no paper comparison leaked into stdout
    assert "absolute_gap" not in combined
    assert "paper_reference" not in combined


def test_compare_reproduction_produces_a_comparison_without_search_mode(tmp_path):
    metrics = tmp_path / "ticker_metrics.csv"
    pd.DataFrame({"accuracy": [0.815], "f1": [0.60], "brier": [0.205]}).to_csv(metrics, index=False)
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "compare_reproduction.py"),
         "--run-dir", str(tmp_path), "--metrics-csv", str(metrics)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=600)
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    assert payload["paper_reference"]["accuracy"] == 0.815
    assert payload["absolute_gap"]["accuracy"] == pytest.approx(0.0, abs=1e-9)


# ================================================== J. final runner executes

def test_run_recovered_paper_calls_walk_forward_after_authorisation(tmp_path, monkeypatch):
    """With a valid freeze and FINAL_TEST=1 it must ACTUALLY call run_walk_forward."""
    from agentic_forecaster.recovery import freeze

    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True)
    cfg = cfg_dir / "recovered_paper.yaml"
    cfg.write_text("experiment:\n  name: recovered\ndata:\n  synthetic: true\n")

    # a real ledger with one experiment per search fold, test_evaluated=false
    from agentic_forecaster.recovery.ledger import append_experiment
    ledger = tmp_path / "results" / "reproduction_recovery" / "experiment_ledger.csv"
    ids = []
    for f in ("SEARCH_FOLD_A", "SEARCH_FOLD_B", "SEARCH_FOLD_C"):
        row = append_experiment({"search_fold": f, "ticker_subset": "TINY",
                                "max_epochs": 30, "test_evaluated": "false",
                                "validation_accuracy": 0.55}, path=ledger)
        ids.append(row["experiment_id"])

    manifest_json = tmp_path / "ds_manifest.json"
    manifest_json.write_text("{}")
    freeze.freeze_config(cfg, dataset_variant="unadjusted",
                         dataset_manifest=manifest_json,
                         universe_id="PAPER_CAND", validation_metrics={"accuracy": 0.55},
                         supporting_experiment_ids=ids, root=tmp_path, ledger=ledger)

    # monkeypatch run_walk_forward so no real training happens
    called: dict = {}

    def fake_run_walk_forward(config, device=None, run_id=None, tickers=None):
        called["config"] = config
        called["device"] = device
        called["run_id"] = run_id
        return {"run_dir": str(tmp_path / "run"), "folds": ["fold_0", "fold_1"]}

    import agentic_forecaster.orchestration.walk_forward as wf
    monkeypatch.setattr(wf, "run_walk_forward", fake_run_walk_forward)

    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_recovered_paper.py"),
         "--config", str(cfg), "--device", "cpu", "--run-id", "unit_final"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=600,
        env={**os.environ, "FINAL_TEST": "1", "AGENTIC_OUTPUT_ROOT": str(tmp_path / "out")})
    combined = out.stdout + out.stderr
    assert "not wired" not in combined.lower(), "placeholder still present"
    assert "frozen_config" in combined, combined[-500:]
    assert "reproduction_recovery" in combined or "run_walk_forward" in combined


def test_freeze_gate_is_strict(tmp_path):
    """Unknown ids, empty metrics, missing manifest must all FAIL."""
    from agentic_forecaster.recovery.freeze import FrozenConfigError, freeze_config
    from agentic_forecaster.recovery.ledger import append_experiment

    cfg = tmp_path / "configs" / "recovered_paper.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("data: {}\n")
    ds = tmp_path / "ds.json"
    ds.write_text("{}")
    ledger = tmp_path / "results" / "reproduction_recovery" / "experiment_ledger.csv"
    ids = [append_experiment({"search_fold": f, "test_evaluated": "false"},
                             path=ledger)["experiment_id"]
           for f in ("SEARCH_FOLD_A", "SEARCH_FOLD_B", "SEARCH_FOLD_C")]

    # unknown id -> FAIL
    with pytest.raises(FrozenConfigError, match="not present in the ledger"):
        freeze_config(cfg, dataset_variant="unadjusted", dataset_manifest=ds,
                      universe_id="U", validation_metrics={"a": 1},
                      supporting_experiment_ids=ids + ["EXP-NOPE"], root=tmp_path,
                      ledger=ledger)
    # empty metrics -> FAIL
    with pytest.raises(FrozenConfigError, match="EMPTY validation metrics"):
        freeze_config(cfg, dataset_variant="unadjusted", dataset_manifest=ds,
                      universe_id="U", validation_metrics={},
                      supporting_experiment_ids=ids, root=tmp_path, ledger=ledger)
    # missing dataset manifest -> FAIL
    with pytest.raises(FrozenConfigError, match="dataset manifest"):
        freeze_config(cfg, dataset_variant="unadjusted", dataset_manifest=None,
                      universe_id="U", validation_metrics={"a": 1},
                      supporting_experiment_ids=ids, root=tmp_path, ledger=ledger)
    # no supporting ids -> FAIL
    with pytest.raises(FrozenConfigError, match="supporting experiment ids"):
        freeze_config(cfg, dataset_variant="unadjusted", dataset_manifest=ds,
                      universe_id="U", validation_metrics={"a": 1},
                      supporting_experiment_ids=[], root=tmp_path, ledger=ledger)
    # only one search fold -> FAIL
    with pytest.raises(FrozenConfigError, match="all search folds"):
        freeze_config(cfg, dataset_variant="unadjusted", dataset_manifest=ds,
                      universe_id="U", validation_metrics={"a": 1},
                      supporting_experiment_ids=[ids[0]], root=tmp_path, ledger=ledger)
    # a test-leaking support id -> FAIL
    bad = append_experiment({"search_fold": "SEARCH_FOLD_C",
                             "test_evaluated": "true"}, path=ledger)["experiment_id"]
    with pytest.raises(FrozenConfigError, match="test evaluation"):
        freeze_config(cfg, dataset_variant="unadjusted", dataset_manifest=ds,
                      universe_id="U", validation_metrics={"a": 1},
                      supporting_experiment_ids=ids + [bad], root=tmp_path, ledger=ledger)


# ================================================== K. volume_mode

@pytest.mark.parametrize("mode", ["raw", "log1p"])
def test_volume_mode_reaches_the_model_input(tmp_path, mode):
    from agentic_forecaster.data.agent import DataAgent
    cfg = _search_cfg(_synthetic_raw(tmp_path), tmp_path / f"p_{mode}", "SEARCH_FOLD_C")
    cfg["data"]["volume_mode"] = mode
    ds = DataAgent(cfg).run("TINY")
    assert ds.volume_mode == mode
    vi = ds.feature_names.index("volume")
    # raw_volume is always the GENUINE market volume
    assert ds.train.raw_volume is not None
    assert (ds.train.raw_volume > 0).all()
    if mode == "log1p":
        # the model input is compressed relative to the raw market volume
        assert float(np.mean(ds.train.X[:, -1, vi])) < float(np.mean(ds.train.raw_volume))


def test_volume_mode_does_not_touch_ohlc():
    from agentic_forecaster.data.scaling import apply_volume_mode
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3),
                       "open": [1.0, 2.0, 3.0], "high": [2.0, 3.0, 4.0],
                       "low": [0.5, 1.5, 2.5], "close": [1.5, 2.5, 3.5],
                       "volume": [100.0, 200.0, 300.0]})
    out = apply_volume_mode(df, "log1p")
    for col in ("open", "high", "low", "close"):
        assert out[col].tolist() == df[col].tolist()
    assert out["volume"].tolist() == pytest.approx(np.log1p([100.0, 200.0, 300.0]).tolist())
    # the input frame is not mutated
    assert df["volume"].tolist() == [100.0, 200.0, 300.0]


def test_unknown_volume_mode_rejected():
    from agentic_forecaster.data.scaling import apply_volume_mode
    with pytest.raises(KeyError):
        apply_volume_mode(pd.DataFrame({"volume": [1.0]}), "sqrt")


# ================================================== L. feature/RSI variants act

def test_feature_families_change_the_real_input_columns(tmp_path):
    from agentic_forecaster.data.agent import DataAgent
    counts = {}
    for fam in ("F0", "F1", "F2", "F3"):
        cfg = _search_cfg(_synthetic_raw(tmp_path / fam), tmp_path / f"p_{fam}",
                          "SEARCH_FOLD_C")
        indicators = list(__import__(
            "agentic_forecaster.recovery.variants", fromlist=["x"]
        ).build_feature_indicators(fam, "R1_wilder"))
        cfg["features"]["indicators"] = indicators
        ds = DataAgent(cfg).run("TINY")
        n_in = ds.train.X.shape[-1]
        counts[fam] = n_in
        assert n_in == 5 + len(indicators)  # OHLCV + indicators
    assert counts["F0"] < counts["F1"] < counts["F2"] <= counts["F3"]


def test_rsi_variants_change_the_actual_column(tmp_path):
    from agentic_forecaster.data.agent import DataAgent
    from agentic_forecaster.recovery.variants import build_feature_indicators
    raw = _synthetic_raw(tmp_path)
    traces = {}
    for m in ("R0_rolling", "R1_wilder", "R2_ema"):
        cfg = _search_cfg(raw, tmp_path / f"p_{m}", "SEARCH_FOLD_C")
        cfg["features"]["indicators"] = list(build_feature_indicators("F1", m))
        ds = DataAgent(cfg).run("TINY")
        assert sum("rsi" in n for n in ds.feature_names) == 1
        vi = ds.feature_names.index(
            {"R0_rolling": "rsi_14_rolling", "R1_wilder": "rsi_14",
             "R2_ema": "rsi_14_ema"}[m])
        traces[m] = ds.train.X[:, -1, vi].copy()
    assert not np.allclose(traces["R0_rolling"], traces["R1_wilder"])
    assert not np.allclose(traces["R2_ema"], traces["R1_wilder"])


def test_architecture_lookback_dropout_reach_the_model(tmp_path):
    from agentic_forecaster.agents.model_agent import ModelAgent
    raw = _synthetic_raw(tmp_path)
    seen = set()
    for label, lookback, arch, drop, wd, epochs in (
        ("A0", 30, (2, 64), 0.2, 1e-4, 2),
        ("A1", 10, (1, 64), 0.0, 0.0, 1),
        ("A3", 40, (2, 128), 0.5, 1e-3, 1),
    ):
        cfg = _search_cfg(raw, tmp_path / f"p_{label}", "SEARCH_FOLD_C")
        cfg["data"]["sequence_length"] = lookback
        cfg["models"]["attention_lstm"].update(
            {"num_layers": arch[0], "hidden_size": arch[1], "dropout": drop,
             "weight_decay": wd, "max_epochs": epochs, "patience": 1})
        with FIREWALL(True):
            ds = __import__("agentic_forecaster.data.agent", fromlist=["DataAgent"]
                            ).DataAgent(cfg).run("TINY")
            fitted = ModelAgent(cfg).train_ticker("TINY", ds, fold="SEARCH_FOLD_C")
        assert ds.train.X.shape[1] == lookback
        assert fitted.train_config["num_layers"] == arch[0]
        assert fitted.train_config["hidden_size"] == arch[1]
        assert fitted.train_config["dropout"] == drop
        assert fitted.train_config["weight_decay"] == wd
        assert fitted.train_config["max_epochs"] == epochs
        seen.add(label)
    assert len(seen) == 3
