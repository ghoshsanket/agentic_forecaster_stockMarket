"""Regression tests for the pre-Stage-A reproducibility fixes.

Covers:

* deterministic model INITIALISATION (weights depend on the seed, not on
  whatever experiment happened to run before)
* generic calibration survives a FULL model-bundle round trip for all four
  methods, verified through predict_proba() rather than by inspecting JSON
* Stage-D nested (chronological) calibration never fits and scores on the same
  labels
* Stage-0 ledger rows carry per-experiment metadata
* raw volume is the Stage-A reference
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


# ============================================ 1. deterministic INITIALISATION

def _state_dict(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _identical(a: dict, b: dict) -> bool:
    return all(torch.equal(a[k], b[k]) for k in a)


def test_seed_determines_initial_attention_lstm_weights():
    """The requested seed alone must fix the initial weights."""
    from agentic_forecaster.models.attention_lstm import AttentionLSTM
    from agentic_forecaster.utils import seed_everything

    # Burn RNG state in between, exactly as a previously-run experiment would.
    seed_everything(1234)
    a = _state_dict(AttentionLSTM(4, 8, 1, 0.0))
    _ = np.random.default_rng(0).normal(size=10_000)
    torch.rand(1_000)
    seed_everything(1234)
    b = _state_dict(AttentionLSTM(4, 8, 1, 0.0))
    assert _identical(a, b), "same seed must give identical initial weights"

    seed_everything(999)
    c = _state_dict(AttentionLSTM(4, 8, 1, 0.0))
    assert not _identical(a, c), "a different seed must give different weights"


def test_model_agent_seeds_before_constructing_the_model():
    """ModelAgent must call seed_everything() itself, not rely on Trainer."""
    import inspect

    from agentic_forecaster.agents.model_agent import ModelAgent

    src = inspect.getsource(ModelAgent.train_ticker)
    assert "seed_everything(" in src, "train_ticker must seed explicitly"
    assert src.index("seed_everything(") < src.index("model = AttentionLSTM("), (
        "seeding must happen BEFORE model construction")

    baseline_src = inspect.getsource(ModelAgent.train_baselines)
    if "PlainLSTM(" in baseline_src:
        assert baseline_src.index("seed_everything(") < baseline_src.index("PlainLSTM(")


def test_same_seed_reproduces_training_result():
    """Train the same tiny model twice with an identical config and seed."""
    from agentic_forecaster.agents.model_agent import ModelAgent

    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 6, 3)).astype(np.float32)
    y = (rng.random(60) < 0.5).astype(np.float32)

    class _Split:
        def __init__(self, X_, y_):
            self.X, self.y = X_, y_

    class _DS:
        def __init__(self):
            self.train = _Split(X, y)
            self.val = _Split(X, y)
            self.test = _Split(np.zeros((0, 6, 3), np.float32),
                               np.zeros((0,), np.float32))
            self.feature_names = ["a", "b", "c"]
            self.scaler = None

    cfg = {"experiment": {"seed": 11},
           "models": {"attention_lstm": {"hidden_size": 6, "num_layers": 1,
                                           "dropout": 0.0, "max_epochs": 3,
                                           "patience": 5, "batch_size": 16}}}
    ma = ModelAgent(cfg)
    f1 = ma.train_ticker("TINY", _DS(), fold="fold_0")
    # perturb the global RNG, then repeat with the same requested seed
    _ = np.random.default_rng(99).normal(size=5_000)
    torch.rand(500)
    f2 = ma.train_ticker("TINY", _DS(), fold="fold_0")

    assert _identical(_state_dict(f1.model), _state_dict(f2.model))
    assert np.allclose(f1.predict_proba(X), f2.predict_proba(X), atol=1e-6), (
        "predictions must be reproducible")


def test_seed_everything_is_repeatable():
    from agentic_forecaster.utils import seed_everything
    seed_everything(5)
    a = torch.rand(5)
    seed_everything(5)
    assert torch.equal(a, torch.rand(5))


# ================================== 2. generic calibration bundle round trip

@pytest.mark.parametrize("method", ["none", "temperature", "platt", "isotonic"])
def test_full_attention_lstm_bundle_round_trip(tmp_path, method):
    """Create a REAL AttentionLSTM bundle, save, reload, compare predict_proba."""
    from agentic_forecaster.agents.model_agent import FittedModel
    from agentic_forecaster.calibration.registry import build_calibrator
    from agentic_forecaster.models.attention_lstm import AttentionLSTM
    from agentic_forecaster.models.checkpoint import load_model_bundle, save_model_bundle
    from agentic_forecaster.utils import seed_everything

    seed_everything(3)
    input_dim = 4
    model = AttentionLSTM(input_dim, 8, 1, 0.0)
    X = np.random.default_rng(1).normal(size=(80, 5, input_dim)).astype(np.float32)
    with torch.no_grad():
        raw = torch.sigmoid(model(torch.tensor(X)).reshape(-1)).numpy()

    rng = np.random.default_rng(2)
    y = (rng.random(80) < raw).astype(int)
    cal = build_calibrator(method).fit(raw, y)
    fm = FittedModel(name="attention_lstm", kind="torch", model=model, ticker="TINY",
                     fold="SEARCH_FOLD_C", metrics={"accuracy": 0.5},
                     calibration=cal.to_dict(), calibration_method=method,
                     # the loader rebuilds the architecture from config.json
                     train_config={"hidden_size": 8, "num_layers": 1,
                                   "dropout": 0.0},
                     feature_names=["a", "b", "c", "d"], seed=3)
    fm._calibrator = cal
    bundle = save_model_bundle(fm, tmp_path / method)

    payload = json.loads((bundle / "calibration.json").read_text())
    assert payload.get("method") == method

    # Load on the SAME device as the original model. "auto" would move the
    # bundle to CUDA while the in-memory model stays on CPU, and the float32
    # LSTM kernels then differ by ~1e-5 (amplified by isotonic's step
    # function). Same device -> bit-identical, which is the real test of
    # serialisation fidelity.
    loaded = load_model_bundle(bundle, device="cpu")
    assert loaded.calibration_method == method
    assert loaded._calibrator is not None, "generic calibrator was not restored"
    p_original = fm.predict_proba(X)
    p_loaded = loaded.predict_proba(X)
    assert np.allclose(p_original, p_loaded, atol=1e-6), (
        f"{method}: loaded calibrated predictions must match the original")
    if method == "temperature":
        assert loaded.temperature == pytest.approx(cal.to_dict()["temperature"])
    else:
        # a non-temperature method must not be forced into a fake temperature
        assert loaded.temperature == 1.0


def test_legacy_temperature_bundle_loads_through_the_registry(tmp_path):
    """A pre-generic bundle storing only a temperature float must still work."""
    from agentic_forecaster.agents.model_agent import FittedModel
    from agentic_forecaster.calibration.temperature import TemperatureCalibrator
    from agentic_forecaster.models.attention_lstm import AttentionLSTM
    from agentic_forecaster.models.checkpoint import load_model_bundle, save_model_bundle
    from agentic_forecaster.utils import seed_everything

    seed_everything(4)
    fm = FittedModel(name="attention_lstm", kind="torch",
                     model=AttentionLSTM(3, 6, 1, 0.0), ticker="TINY",
                     temperature=1.7,
                     train_config={"hidden_size": 6, "num_layers": 1,
                                   "dropout": 0.0},
                     feature_names=["a", "b", "c"])
    bundle = save_model_bundle(fm, tmp_path / "legacy")
    # rewrite calibration.json in the OLD shape
    (bundle / "calibration.json").write_text(
        json.dumps({"temperature": 1.7, "classification": "reconstruction_assumed"}))

    loaded = load_model_bundle(bundle, device="cpu")
    assert loaded.temperature == pytest.approx(1.7)
    X = np.random.default_rng(5).normal(size=(4, 5, 3)).astype(np.float32)
    ref = TemperatureCalibrator()
    ref.temperature = 1.7
    expected = ref.calibrate_proba(loaded.predict_proba_raw(X))
    assert np.allclose(loaded.predict_proba(X), expected, atol=1e-6)


def test_checkpoint_docstring_mentions_the_selected_calibrator():
    src = (REPO_ROOT / "src" / "agentic_forecaster" / "models" / "checkpoint.py").read_text()
    assert "calibration.json   selected calibrator" in src
    assert "temperature + metadata" not in src


# ======================================= 7. Stage-D nested calibration tests

def test_temporal_split_is_chronological_and_disjoint():
    from agentic_forecaster.recovery.scoring import temporal_calibration_split
    dates = pd.bdate_range("2021-01-04", periods=250)
    fit, score = temporal_calibration_split(dates, 0.6)
    assert len(fit) == 150 and len(score) == 100
    assert fit.max() < score.min(), "fit must precede score (no shuffle)"
    assert not set(fit) & set(score)


def test_stage_d_does_not_fit_and_score_on_the_same_labels():
    from agentic_forecaster.recovery.scoring import stage_d_calibration_metrics
    rng = np.random.default_rng(0)
    n = 250
    dates = pd.bdate_range("2021-01-04", periods=n)
    p = rng.random(n)
    y = (rng.random(n) < p).astype(float)
    rows = stage_d_calibration_metrics(p, y, dates)
    for r in rows:
        # the two windows are DISJOINT, so no label is both fitted and scored
        assert r["n_fit"] + r["n_score"] == n, "every date must be used exactly once"
        assert r["n_fit"] >= 2 and r["n_score"] >= 2
        assert r["fit_end"] < r["score_start"]
        assert r["brier"] is not None
    assert {r["method"] for r in rows} == {"none", "temperature", "platt", "isotonic"}


def test_stage_d_firewall_blocks_2022():
    from agentic_forecaster.recovery.firewall import TestSetFirewallError
    from agentic_forecaster.recovery.scoring import stage_d_calibration_metrics
    n = 60
    dates = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(1)
    p = rng.random(n)
    y = (rng.random(n) < p).astype(float)
    with pytest.raises(TestSetFirewallError):
        stage_d_calibration_metrics(p, y, dates)


def test_final_paper_calibration_stays_train_val_then_test():
    """The final run differs and is valid: val fits, TEST scores."""
    from agentic_forecaster.recovery.firewall import TestSetFirewallError
    from agentic_forecaster.recovery.scoring import score_validation_with_targets
    n = 40
    val = pd.bdate_range("2021-01-04", periods=n)
    rng = np.random.default_rng(2)
    p = rng.random(n)
    y = (rng.random(n) < p).astype(float)
    m = score_validation_with_targets(y, p, val, val + pd.offsets.BDay(1))
    assert m["accuracy"] is not None
    with pytest.raises(TestSetFirewallError):
        score_validation_with_targets(
            y, p, pd.bdate_range("2022-01-03", periods=n),
            pd.bdate_range("2022-01-04", periods=n))


# ================================================= 3. Stage-0 ledger metadata

def test_stage0_ledger_rows_carry_per_experiment_metadata(tmp_path):
    """Each Stage-0 row records ITS OWN requested epochs/patience/actual run."""
    from agentic_forecaster.recovery.firewall import firewall_guard
    from agentic_forecaster.recovery.harness import (
        run_real_label_reference,
        run_shuffled_label_control,
        run_tiny_overfit,
    )
    from agentic_forecaster.recovery.ledger import append_experiment, read_ledger

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6, 3)).astype(np.float32)
    y = (rng.random(400) < 0.5).astype(np.float32)
    with firewall_guard(True):
        a = run_tiny_overfit(X, y, X, y, epochs=40, patience=40, max_train_rows=64,
                             hidden_size=8, num_layers=1, batch_size=16, seed=1)
        b = run_shuffled_label_control(X, y, X, y, epochs=5, hidden_size=8,
                                       num_layers=1, batch_size=16, seed=1)
        c = run_real_label_reference(X, y, X, y, epochs=3, hidden_size=8,
                                     num_layers=1, batch_size=16, seed=1)

    ledger = tmp_path / "experiment_ledger.csv"
    for res, req, pat in ((a, 40, 40), (b, 5, 10), (c, 3, 10)):
        with firewall_guard(True):
            append_experiment({"search_fold": "SEARCH_FOLD_C",
                               "max_epochs": req, "patience": pat,
                               "best_epoch": res.best_epoch,
                               "epochs_run": res.epochs_run,
                               "test_evaluated": "false"}, path=ledger)
    rows = read_ledger(ledger)
    assert [int(r["max_epochs"]) for r in rows] == [40, 5, 3], "requested epochs must differ"
    assert [int(r["patience"]) for r in rows] == [40, 10, 10], "patience must differ"
    assert all(r["epochs_run"] != "" for r in rows), "actual epochs run must be recorded"


def test_raw_volume_is_the_stage_a_reference():
    from agentic_forecaster.recovery.variants import DEFAULT_VOLUME_MODE, VOLUME_MODES
    assert DEFAULT_VOLUME_MODE == "raw"
    assert "log1p" in VOLUME_MODES, "log1p must remain a Stage-B option"
    assert VOLUME_MODES["raw"].startswith("raw volume")


def test_calibration_none_is_the_preferred_search_setting():
    """Stage A/B/C compare other axes, so calibration must not confound them.

    The scorer is capable of calibration, but the Stage-A pilot sets
    ``--calibration none`` so validation accuracy/F1/Brier/ECE/P@3 are computed
    from RAW p(up). Stage D evaluates calibration separately.
    """
    from agentic_forecaster.calibration.registry import SUPPORTED_METHODS
    assert "none" in SUPPORTED_METHODS
    pilot = (REPO_ROOT / "scripts" / "run_stage_a_pilot.sh")
    if pilot.is_file():
        text = pilot.read_text()
        assert "--calibration none" in text, (
            "the Stage-A pilot must score RAW probabilities")


def test_stages_a_to_c_default_to_no_calibration():
    """Stages 0/A/B/C must score RAW p(up); only Stage D defaults to a calibrator.

    A calibrator left on by default would silently confound every A-C
    comparison with a calibration effect, so the stage decides the default.
    """
    import subprocess

    from agentic_forecaster.config import _env_defaults
    roots = _env_defaults()
    for stage, expect in (("0", "none"), ("A", "none"), ("B", "none"),
                          ("C", "none"), ("D", "temperature")):
        out = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "run_reproduction_search.py"),
             "--config", "configs/reproduction_search/paper_snapshot_2025_unadjusted_perf.yaml",
             "--stage", stage, "--search-fold", "SEARCH_FOLD_C",
             "--tickers", "RELIANCE", "--dry-run"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False,
            env={**os.environ, **roots},
        )
        assert out.returncode == 0, out.stderr[-2000:]
        assert f'"calibration": "{expect}"' in out.stdout, (
            f"stage {stage} default calibration was not {expect}")


def test_ten_epochs_is_author_confirmed_primary():
    """The author's confirmed schedule is 10 epochs; T3/T100 are diagnostics.

    This is an evidence claim, not a preference, so it is pinned by test: a
    future edit that promotes T100 or reinstates a 3-epoch default would
    silently break faithfulness of the reconstruction.
    """
    from agentic_forecaster.recovery import variants

    t10 = variants.TRAINING_LENGTHS["T10_AUTHOR_CONFIRMED"]
    assert t10["max_epochs"] == 10
    assert t10["patience"] == 10
    assert t10["restore_best_checkpoint"] is True
    assert t10["evidence"] == "AUTHOR-CONFIRMED"
    assert t10["role"] == "PRIMARY"
    assert variants.DEFAULT_TRAINING_LENGTH == "T10_AUTHOR_CONFIRMED"

    # T3 must never be presented as the author's choice.
    t3 = variants.TRAINING_LENGTHS["T3_RECONSTRUCTION_SHORTCUT"]
    assert t3["max_epochs"] == 3
    assert t3["role"] == "diagnostic"
    assert t3["evidence"] == "RECONSTRUCTION-SHORTCUT"

    t100 = variants.TRAINING_LENGTHS["T100_DIAGNOSTIC"]
    assert t100["max_epochs"] == 100
    assert t100["role"] == "diagnostic"
    assert t100["evidence"] == "diagnostic"


def test_author_confirmed_pilot_script_uses_ten_epochs_and_no_calibration():
    """The pilot must run T10 x {F1,F2} with calibration off and no FINAL_TEST."""
    script = (REPO_ROOT / "scripts" / "run_stage_a_pilot.sh").read_text()
    assert "T10_AUTHOR_CONFIRMED F1" in script
    assert "T10_AUTHOR_CONFIRMED F2" in script
    assert "--calibration none" in script
    # Strip comments: the header legitimately *documents* that FINAL_TEST is
    # never set, so only executable lines are asserted on.
    code = "\n".join(line for line in script.splitlines()
                     if not line.lstrip().startswith("#"))
    assert "FINAL_TEST" not in code
    assert "run_recovered_paper" not in code
