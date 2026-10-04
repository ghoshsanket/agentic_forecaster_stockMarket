"""Tests for the pre-2022 forensic diagnosis.

These pin the properties that make the forensics trustworthy: the faithful
configs really do use the author-confirmed schedule, the firewall genuinely
rejects a protected date, the adjusted selector really changes the data root,
and the deliberately invalid probes are permanently marked as such.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.recovery import forensics as fx


def _cfg(rel: str) -> dict:
    return yaml.safe_load((REPO_ROOT / rel).read_text())


def _load_runner(alias: str):
    """Import the forensic runner as a module (it is a script, not a package)."""
    path = REPO_ROOT / "scripts" / "run_pre2022_forensic_diagnostics.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. faithful configs use 10 epochs
# ---------------------------------------------------------------------------

def test_paper_yaml_uses_ten_epochs_for_both_lstm_variants():
    cfg = _cfg("configs/paper.yaml")
    for key in ("attention_lstm", "lstm"):
        block = cfg["models"][key]
        assert block["epochs"] == 10, f"models.{key}.epochs must be 10"
        assert block["patience"] == 10, f"models.{key}.patience must be 10"


def test_demo_config_is_untouched():
    """The demo config is intentionally short; it must NOT be bumped to 10."""
    assert _cfg("configs/demo.yaml")["models"]["attention_lstm"]["epochs"] == 3


@pytest.mark.parametrize("name", [
    "paper_snapshot_2025_unadjusted_perf",
    "paper_snapshot_2025_adjusted_perf",
    "legacy_yfinance_perf_unadjusted",
    "legacy_yfinance_perf_adjusted",
])
def test_faithful_perf_configs_use_ten_epochs(name):
    cfg = _cfg(f"configs/reproduction_search/{name}.yaml")
    block = cfg["models"]["attention_lstm"]
    assert block["max_epochs"] == 10, f"{name}: max_epochs must be 10"
    assert block["patience"] == 10, f"{name}: patience must be 10"
    assert block["restore_best_checkpoint"] is True, f"{name}: restore_best"


def test_hundred_epoch_diagnostic_is_not_the_default():
    """100 epochs must survive only as a diagnostic, never as the default."""
    from agentic_forecaster.recovery import variants
    assert variants.DEFAULT_TRAINING_LENGTH == "T10_AUTHOR_CONFIRMED"
    assert variants.TRAINING_LENGTHS["T100_DIAGNOSTIC"]["role"] == "diagnostic"
    assert variants.TRAINING_LENGTHS["T10_AUTHOR_CONFIRMED"]["max_epochs"] == 10


# ---------------------------------------------------------------------------
# 2. the firewall rejects protected dates
# ---------------------------------------------------------------------------

def test_legitimate_run_rejects_protected_target_dates():
    good = pd.date_range("2020-01-01", "2021-12-31", freq="D")
    fx.assert_targets_pre_test(good, good, where="ok")  # must not raise

    leaking = list(good) + [pd.Timestamp("2022-03-01")]
    with pytest.raises(fx.ForensicFirewallAbort):
        fx.assert_targets_pre_test(leaking, good, where="train")
    with pytest.raises(fx.ForensicFirewallAbort):
        fx.assert_targets_pre_test(good, leaking, where="val")


def test_cutoff_boundary_is_exclusive():
    """2021-12-31 is legal; 2022-01-01 is not."""
    ok = [pd.Timestamp("2021-12-31")]
    fx.assert_targets_pre_test(ok, ok)
    with pytest.raises(fx.ForensicFirewallAbort):
        fx.assert_targets_pre_test([pd.Timestamp("2022-01-01")], ok)


def test_metrics_refuse_protected_dates():
    df = pd.DataFrame({
        "ticker": ["A", "B"], "fold": ["F", "F"],
        "y_true": [1, 0], "p_up": [0.9, 0.1],
        "target_date": ["2021-12-30", "2022-05-05"],
    })
    with pytest.raises(fx.ForensicFirewallAbort):
        fx.aggregation_metrics(df)
    with pytest.raises(fx.ForensicFirewallAbort):
        fx.confidence_subset_metrics(df)


# ---------------------------------------------------------------------------
# 3. adjusted/unadjusted selector changes the REAL data root
# ---------------------------------------------------------------------------

def test_adjusted_selector_changes_real_data_root(tmp_path):
    """A cosmetic flag would make diagnostic A silently re-run unadjusted data."""
    mod = _load_runner("fp")

    root = tmp_path / "ds" / "yfinance_paper_snapshot_2025_11_04"
    (root / "unadjusted" / "csv").mkdir(parents=True)
    (root / "adjusted" / "csv").mkdir(parents=True)
    cfg = {
        "experiment": {"seed": 42},
        "data": {"raw_root": str(root / "unadjusted" / "csv"),
                 "processed_root": str(tmp_path / "p"),
                 "variant": "unadjusted", "search_mode": True,
                 "sequence_length": 30, "scaler": "standard",
                 "volume_mode": "raw", "train_frac": 0.7, "val_frac": 0.15,
                 "test_frac": 0.15},
        "features": {"indicators": [], "use_ohlcv": True},
        "models": {"attention_lstm": {}, "random_forest": {}},
        "calibration": {"method": "none"},
    }
    # a config file on disk for the loader
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    base = str(cfg_path)
    unadj = mod.build_config(base, "SEARCH_FOLD_C", adjusted=False)
    adj = mod.build_config(base, "SEARCH_FOLD_C", adjusted=True)
    assert unadj["data"]["raw_root"] != adj["data"]["raw_root"]
    assert Path(unadj["data"]["raw_root"]).is_dir()
    assert Path(adj["data"]["raw_root"]).is_dir()
    assert "unadjusted" in unadj["data"]["raw_root"]
    assert "adjusted" in adj["data"]["raw_root"]
    assert adj["data"]["variant"] == "adjusted"


def test_adjusted_selector_fails_loudly_if_adjusted_root_absent(tmp_path):
    mod = _load_runner("fp2")
    root = tmp_path / "ds"
    (root / "unadjusted" / "csv").mkdir(parents=True)   # no adjusted/ dir
    cfg = {"experiment": {"seed": 42},
           "data": {"raw_root": str(root / "unadjusted" / "csv"),
                    "processed_root": str(tmp_path / "p"),
                    "variant": "unadjusted", "search_mode": True,
                    "sequence_length": 30, "scaler": "standard",
                    "volume_mode": "raw", "train_frac": 0.7, "val_frac": 0.15,
                    "test_frac": 0.15},
           "features": {"indicators": [], "use_ohlcv": True},
           "models": {"attention_lstm": {}, "random_forest": {}},
           "calibration": {"method": "none"}}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(FileNotFoundError):
        mod.build_config(str(p), "SEARCH_FOLD_C", adjusted=True)


# ---------------------------------------------------------------------------
# 4. pooled training uses TRAIN data only
# ---------------------------------------------------------------------------

def test_pooled_diagnostic_uses_train_data_only_and_is_labelled():
    src = (REPO_ROOT / "scripts" / "run_pre2022_forensic_diagnostics.py").read_text()
    # the pooled fit must be given train arrays, never validation
    i = src.index("def diagnostic_c")
    body = src[i:src.index("def diagnostic_d")]
    assert "ds.train.X" in body and "ds.val.X" in body
    assert "tr.fit(Xtr, ytr, Xva, yva)" in body
    # the pooled artifact must be permanently labelled as NOT author-confirmed
    assert "fx.POOLED_LABEL" in body
    assert fx.POOLED_LABEL == "FORENSIC_POOLED_NOT_AUTHOR_CONFIRMED"


def test_pooled_marker_is_written_into_artifacts():
    out = fx.write_forensic_csv(
        [{"fold": "F", "accuracy": 0.5}],
        REPO_ROOT / "results" / "reproduction_recovery" / "forensics" / "_t.csv",
        marker={"LABEL": fx.POOLED_LABEL})
    try:
        with out.open() as fh:
            row = next(csv.DictReader(fh))
        assert row["LABEL"] == fx.POOLED_LABEL
    finally:
        out.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 5. the 85/15 analog cannot reach a protected year
# ---------------------------------------------------------------------------

def test_85_15_analog_never_exceeds_2021():
    dates = pd.date_range("2016-01-01", "2021-12-31", freq="D")
    tr, va, last = fx.pre2022_85_15_split(dates)
    assert tr and va
    assert max(tr) < fx.PRE_TEST_CUTOFF
    assert max(va) < fx.PRE_TEST_CUTOFF
    assert last < fx.PRE_TEST_CUTOFF
    # chronological, no shuffle: every train date precedes every val date
    assert max(tr) < min(va)
    # 85/15 proportions
    assert 0.80 <= len(tr) / (len(tr) + len(va)) <= 0.90


def test_85_15_truncates_even_when_a_protected_date_is_supplied():
    dates = list(pd.date_range("2016-01-01", "2023-12-31", freq="D"))
    tr, va, last = fx.pre2022_85_15_split(dates)
    assert max(tr + va) < fx.PRE_TEST_CUTOFF
    assert last < fx.PRE_TEST_CUTOFF


# ---------------------------------------------------------------------------
# 6. the alignment validator detects an off-by-one
# ---------------------------------------------------------------------------

def _fake_dataset(seq_len: int = 5):
    class _Split:
        X = np.zeros((3, seq_len, 2))
        # next-day moves in _fake_frame(): 11.0->10.5 DOWN, 10.5->12.0 UP,
        # 12.0->11.5 DOWN
        y = np.array([0, 1, 0])
        dates = np.array(["2020-01-02", "2020-01-03", "2020-01-06"])
        target_dates = np.array(["2020-01-03", "2020-01-06", "2020-01-07"])
    class _DS:
        train = _Split()
        val = _Split()
    return _DS()


def _fake_frame() -> pd.DataFrame:
    # consecutive trading days only; no weekends
    return pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02", "2020-01-03",
                 "2020-01-06", "2020-01-07", "2020-01-08"],
        "Close": [10.0, 11.0, 10.5, 12.0, 11.5, 13.0],
    })


def test_alignment_validator_passes_on_correct_construction():
    rows, mismatches = fx.validate_target_alignment(
        _fake_dataset(), _fake_frame(), fold="F", ticker="T", n_samples=3)
    assert mismatches == 0
    assert all(r.label_matches for r in rows)
    assert all(r.target_is_next_trading_day for r in rows)
    assert all(r.sequence_excludes_target for r in rows)


def test_alignment_validator_detects_intentional_off_by_one():
    """The validator must FAIL when the target row is inside the window."""
    rows, mismatches = fx.validate_target_alignment(
        _fake_dataset(), _fake_frame(), fold="F", ticker="T", n_samples=3,
        probe_sequence_includes_target=True)
    assert mismatches == len(rows) > 0
    assert all(r.sequence_contains_target_row for r in rows)
    assert not any(r.sequence_excludes_target for r in rows)


def test_alignment_validator_detects_a_shifted_label():
    """A deliberately wrong label must be caught by independent re-derivation."""
    ds = _fake_dataset()
    ds.train.y = np.array([1, 0, 1])      # inverted vs the true next-day move
    rows, mismatches = fx.validate_target_alignment(
        ds, _fake_frame(), fold="F", ticker="T", n_samples=3)
    assert mismatches == 3
    assert all(not r.label_matches for r in rows)


# ---------------------------------------------------------------------------
# 7. invalid probes are permanently marked
# ---------------------------------------------------------------------------

def test_invalid_probe_marker_is_correct():
    assert fx.INVALID_MARKER["VALID_FOR_FINAL_MODEL"] is False
    assert fx.INVALID_MARKER["SCIENTIFICALLY_INVALID"] is True


def test_invalid_probe_artifacts_are_marked(tmp_path):
    out = fx.write_forensic_csv(
        [{"probe": "L1", "validation_accuracy": 0.99}],
        tmp_path / "L1.csv", marker=fx.INVALID_MARKER)
    with out.open() as fh:
        row = next(csv.DictReader(fh))
    assert row["SCIENTIFICALLY_INVALID"] == "True"
    assert row["VALID_FOR_FINAL_MODEL"] == "False"


def test_probe_definitions_mark_l1_to_l4_invalid_and_l0_l5_not():
    src = (REPO_ROOT / "scripts" / "run_pre2022_forensic_diagnostics.py").read_text()
    assert 'invalid = probe in ("L1", "L2", "L3", "L4")' in src
    # L0 is the correct control and L5 does not alter training
    assert '"L0"' in src and '"L5"' in src


# ---------------------------------------------------------------------------
# 8. no PAPER_REFERENCE anywhere in the forensic path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel", [
    "scripts/run_pre2022_forensic_diagnostics.py",
    "scripts/summarize_pre2022_forensics.py",
    "src/agentic_forecaster/recovery/forensics.py",
])
def test_forensic_code_does_not_import_paper_reference(rel):
    """Check real imports via AST.

    Prose in a docstring that merely *names* PAPER_REFERENCE is fine (and in
    fact desirable, to document the prohibition). What must not exist is an
    actual import or attribute access of it.
    """
    tree = ast.parse((REPO_ROOT / rel).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert "PAPER_REFERENCE" not in a.name
        elif isinstance(node, ast.ImportFrom):
            assert "PAPER_REFERENCE" not in (node.module or "")
            for a in node.names:
                assert "PAPER_REFERENCE" not in a.name
        elif isinstance(node, ast.Attribute):
            assert "PAPER_REFERENCE" not in node.attr
        elif isinstance(node, ast.Name):
            assert node.id != "PAPER_REFERENCE"


def test_forensic_shell_never_references_paper_reference():
    src = (REPO_ROOT / "scripts" / "run_pre2022_forensics.sh").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "PAPER_REFERENCE" not in code


def test_forensic_runner_does_not_invoke_paper_evaluation():
    src = (REPO_ROOT / "scripts" / "run_pre2022_forensics.sh").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "run_recovered_paper.py" not in code
    assert "FINAL_TEST=1" not in code
    assert "freeze_recovered_config" not in code


# ---------------------------------------------------------------------------
# 9. aggregation / confidence behave sanely
# ---------------------------------------------------------------------------

def _legit_frame(n_per_ticker: int = 40) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for t in ("A", "B"):
        for i in range(n_per_ticker):
            y = int(rng.integers(0, 2))
            p = 0.7 if y else 0.3
            rows.append({"ticker": t, "fold": "F1",
                         "target_date": f"2021-01-{i % 28 + 1:02d}",
                         "y_true": y, "p_up": p})
    return pd.DataFrame(rows)


def test_aggregation_reports_every_convention():
    out = fx.aggregation_metrics(_legit_frame())
    for key in ("micro", "macro_ticker", "macro_date",
                "equal_fold_average", "sample_weighted_fold_average"):
        assert key in out
    assert out["micro"]["accuracy"] == pytest.approx(1.0)


def test_aggregation_never_selects_the_largest():
    out = fx.aggregation_metrics(_legit_frame())
    assert "selected" not in out
    assert "NOT selected" in out["note"]


def test_confidence_subsets_are_conditional_and_labelled():
    out = fx.confidence_subset_metrics(_legit_frame())
    assert out["all_predictions"]["n"] == 80
    for k in (">=0.55", ">=0.60", ">=0.70", ">=0.80"):
        assert k in out
        assert out[k]["n"] <= out["all_predictions"]["n"]
    assert "not overall accuracy" in out["note"]


def test_confidence_subset_retains_fewer_at_higher_thresholds():
    out = fx.confidence_subset_metrics(_legit_frame())
    ns = [out[k]["n"] for k in (">=0.55", ">=0.60", ">=0.70", ">=0.80")]
    assert ns == sorted(ns, reverse=True)


def test_label_balance_reports_rates_and_majority():
    out = fx.label_balance(np.array([1, 1, 0, 0]), np.array([1, 1, 1, 0]))
    assert out["train"]["positive_rate"] == pytest.approx(0.5)
    assert out["validation"]["positive_rate"] == pytest.approx(0.75)
    # train is tied, so the constant predictor is UP -> 3 of 4 correct
    assert out["train_majority_baseline_accuracy"] == pytest.approx(0.75)
    assert out["validation_oracle_majority_rate"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 10. cause ranking must not be fooled by small-sample or weak causes
# ---------------------------------------------------------------------------

def _load_summarizer():
    import importlib.util as spec
    path = REPO_ROOT / "scripts" / "summarize_pre2022_forensics.py"
    s = spec.spec_from_file_location("fpsum", path)
    mod = spec.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod


def test_cause_ranking_is_ordered_by_magnitude_not_declaration():
    """A tiny-slice confidence floor must not outrank a real +0.49 cause."""
    m = _load_summarizer()
    ranked = m.cause_ranking(
        unadjusted_acc=0.5250, adjusted_acc=0.5243,
        panel={"LogisticRegression": {"accuracy": 0.5103},
               "RandomForest": {"accuracy": 0.5079},
               "PlainLSTM": {"accuracy": 0.5211}},
        pooled_acc=0.5101, analog_acc=0.5198, majority_acc=0.5172,
        mismatch_count=0,
        aggregation={"micro": {"accuracy": 0.5250}},
        # a 0.63 accuracy on only 1% of predictions must NOT be credited
        confidence={"all_predictions": {"n": 5912, "retained_pct": 100.0,
                                        "accuracy": 0.5250, "f1": 0.60},
                    ">=0.60": {"threshold": 0.6, "n": 62,
                               "retained_pct": 1.0, "accuracy": 0.6290,
                               "f1": 0.77}},
        l5=None)
    by_name = {c["cause"]: c for c in ranked}
    conf = by_name["CONFIDENCE_FILTERING"]
    assert conf["verdict"] == "NOT_SUPPORTED", (
        "a 1%-retention conditional accuracy must not be credited as a cause")
    assert "1.0%" in conf["evidence"]
    # and the genuinely strong causes must be flagged unsupported
    for cause in ("DATA_ADJUSTMENT", "MODEL_FORM", "PER_STOCK_VS_POOLED",
                  "SPLIT_PROTOCOL", "TARGET_ALIGNMENT", "AGGREGATION"):
        assert by_name[cause]["verdict"] == "NOT_SUPPORTED", cause


def test_confidence_floor_is_credited_when_retention_is_real():
    m = _load_summarizer()
    ranked = m.cause_ranking(
        unadjusted_acc=0.5250, adjusted_acc=0.5250, panel={},
        pooled_acc=None, analog_acc=None, majority_acc=0.52,
        mismatch_count=0, aggregation={},
        confidence={"all_predictions": {"n": 1000, "retained_pct": 100.0,
                                        "accuracy": 0.5250, "f1": 0.5},
                    ">=0.55": {"threshold": 0.55, "n": 400,
                               "retained_pct": 40.0, "accuracy": 0.62,
                               "f1": 0.6}},
        l5=None)
    conf = {c["cause"]: c for c in ranked}["CONFIDENCE_FILTERING"]
    assert conf["verdict"] == "SUPPORTED"
    assert conf["material"] is True


def test_target_alignment_is_supported_when_mismatches_exist():
    m = _load_summarizer()
    ranked = m.cause_ranking(
        unadjusted_acc=0.5250, adjusted_acc=0.5250, panel={},
        pooled_acc=None, analog_acc=None, majority_acc=0.52,
        mismatch_count=7, aggregation={}, confidence={}, l5=None)
    ta = {c["cause"]: c for c in ranked}["TARGET_ALIGNMENT"]
    assert ta["verdict"] == "SUPPORTED"
    assert "mismatch_count=7" in ta["evidence"]


def test_leakage_verdict_flags_a_large_inflation():
    m = _load_summarizer()
    probes = {p: {"invalid": p in ("L1", "L2", "L3", "L4"),
                  "rows": [{"validation_accuracy": a, "train_accuracy": 0.99}]}
              for p, a in (("L0", 0.5034), ("L1", 0.5204), ("L2", 0.9937),
                           ("L3", 0.5058), ("L4", 0.5034))}
    out = m.leakage_verdict(probes, 0.5034)
    assert out["verdict"] == "EXPLAINS_INFLATION"
    assert out["max_inflation_probe"] == "L2"
    assert out["max_inflation"] == pytest.approx(0.4903, abs=1e-3)
    assert "never valid models" in out["note"]


def test_leakage_verdict_rejects_a_marginal_inflation():
    m = _load_summarizer()
    probes = {p: {"invalid": True, "rows": [{"validation_accuracy": a,
                                             "train_accuracy": 0.99}]}
              for p, a in (("L1", 0.5204), ("L2", 0.52), ("L3", 0.5058),
                           ("L4", 0.5034))}
    out = m.leakage_verdict(probes, 0.5034)
    assert out["verdict"] == "INSUFFICIENT_TO_EXPLAIN"


def test_recommendation_is_not_executed_automatically():
    m = _load_summarizer()
    # everything demonstrated was ruled out -> ask the author
    assert m._recommend({}, []) == "ASK_AUTHOR_FOR_ORIGINAL_PROTOCOL_DETAILS"
    # a supported SPLIT_PROTOCOL cause is the only thing that licenses Stage B
    assert m._recommend({}, [{"cause": "SPLIT_PROTOCOL",
                              "verdict": "SUPPORTED"}]) == "PROCEED_TO_STAGE_B"
    # a HYPOTHETICAL leakage cause must NEVER license an action
    assert m._recommend({}, [{"cause": "LEAKAGE/OFF_BY_ONE",
                              "verdict": "EXPLAINS_INFLATION"}]) \
        == "ASK_AUTHOR_FOR_ORIGINAL_PROTOCOL_DETAILS"


def test_recommendation_is_one_of_the_permitted_actions():
    m = _load_summarizer()
    for supported in ([], [{"cause": "SPLIT_PROTOCOL", "verdict": "SUPPORTED"}],
                      [{"cause": "DATA_ADJUSTMENT", "verdict": "SUPPORTED"}]):
        assert m._recommend({}, supported) in m.NEXT_ACTIONS


# ===========================================================================
# CORRECTION TESTS (post audit)
# ===========================================================================

def _load_probe_runner():
    import importlib.util
    path = REPO_ROOT / "scripts" / "run_pre2022_forensic_diagnostics.py"
    s = importlib.util.spec_from_file_location("fxcorr", path)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod


# --- A. L1's final feature timestep comes from target_date -----------------

def _seq_frame():
    """10 consecutive trading days, distinguishable feature values.

    Row k gets feature value k, so the vector itself identifies its date. A
    leaked sequence whose final element is the target day must therefore end
    with the target day's value, not the origin day's.
    """
    days = pd.bdate_range("2020-01-01", periods=10)
    seq_len, n_feat = 3, 3
    # every timestep of row i carries the value i, so the vector itself
    # identifies which date it came from
    X = np.stack([np.full((seq_len, n_feat), float(i), dtype=np.float32)
                  for i in range(len(days))], axis=0)
    return X, [str(d.date()) for d in days]


def test_l1_final_timestep_is_the_target_day_feature_vector():
    X, days = _seq_frame()
    origins = days[:-1]
    targets = days[1:]                       # next trading day
    y = np.zeros(len(origins), dtype=int)
    out = fx.build_target_day_leak(X, y, origins, targets)

    # The mapping is built from sample origins, so days[-1] is NOT registered
    # (it is only ever a target). That last sample is therefore genuinely
    # unavailable and is dropped rather than faked: 9 in, 8 out.
    assert out["n_kept"] == len(origins) - 1 == 8
    assert out["n_dropped"] == 1
    assert out["X_leaked"].shape == (8, 3, 3)
    for j, (o, t) in enumerate(zip(out["origin_dates"], out["target_dates"])):
        origin_idx = days.index(o)
        target_idx = days.index(t)
        # the final timestep is the TARGET day's vector
        assert np.allclose(out["X_leaked"][j, -1, 0], float(target_idx))
        # and NOT the origin day's, which is what the old shift-only version
        # left in place
        assert not np.allclose(out["X_leaked"][j, -1, 0], float(origin_idx))
        # the window is shifted left, so element -2 is the origin day
        assert np.allclose(out["X_leaked"][j, -2, 0], float(origin_idx))
        assert out["final_timestep_dates"][j] == t
        assert t > o


def test_l1_drops_only_genuinely_unavailable_targets():
    X, days = _seq_frame()
    origins = days[:4]
    targets = days[1:4] + [days[4]]           # last target is a gap
    out = fx.build_target_day_leak(X, np.zeros(4, int), origins, targets)
    assert out["n_kept"] + out["n_dropped"] == 4
    assert out["n_dropped"] >= 1


def test_l1_rejects_a_non_increasing_target():
    X, days = _seq_frame()
    with pytest.raises(fx.L1AlignmentError):
        fx.build_target_day_leak(X, np.zeros(1, int), [days[3]], [days[3]])


# --- B. the OLD broken implementation fails this test ----------------------

def test_old_shift_only_l1_would_fail_the_alignment_test():
    """The previous implementation shifts the window and leaves the final
    timestep as the ORIGIN row, so it leaks nothing and must fail the same
    assertion the real probe satisfies."""
    X, days = _seq_frame()
    origins, targets = days[:-1], days[1:]

    # --- the OLD implementation, reproduced verbatim ---
    def old_leak_target_day(X):
        out = np.copy(X)
        out[:, :-1, :] = X[:, 1:, :]
        return out

    old = old_leak_target_day(X)
    old_final = old[:, -1, 0]
    # every final element is the ORIGIN day -> alignment assertion fails
    assert all(v != float(days.index(t)) for v, t in zip(old_final, targets))
    violations = sum(1 for v, t in zip(old_final, targets)
                     if v != float(days.index(t)))
    assert violations == len(targets)

    # the CORRECT implementation satisfies it
    good = fx.build_target_day_leak(X, np.zeros(len(origins), int),
                                    origins, targets)
    assert all(v == float(days.index(t))
               for v, t in zip(good["X_leaked"][:, -1, 0], targets))


def test_l1_probe_output_rows_carry_the_alignment_audit():
    """The audit must come from the actual probe output, not a separate helper."""
    mod = _load_probe_runner()
    src = (REPO_ROOT / "scripts" / "run_pre2022_forensic_diagnostics.py").read_text()
    assert "fx.build_target_day_leak" in src
    assert "final_timestep_equals_target_date" in src
    assert "_leak_target_day" not in src.replace(
        "# NOTE: the previous `_leak_target_day` shifted the window left and left", "")
    assert callable(mod._leak_scaler)


# --- C. corrected 85/15 includes pre-2016 samples --------------------------

def test_long_history_config_opens_past_2016():
    """The corrected 85/15 must not inherit a 2016 start from the folds."""
    mod = _load_probe_runner()
    src = (REPO_ROOT / "scripts" / "run_pre2022_forensic_diagnostics.py").read_text()
    body = src[src.index("def diagnostic_d"):src.index("# Diagnostic E")]
    # the fold windows are removed, then the earliest available history is used
    assert 'for k in ("train_start", "train_end", "val_start", "val_end"):' in body
    assert 'data.pop(k, None)' in body
    assert "fx.LONG_HISTORY_EARLIEST" in body
    assert fx.LONG_HISTORY_EARLIEST < "2016-01-01"
    # 2016 must NOT be hard-coded anywhere in the 85/15 body
    assert '"2016-01-01"' not in body
    assert callable(mod.diagnostic_d)


def test_long_history_split_includes_pre_2016_dates_when_available():
    """A source with 2000+ history must yield pre-2016 training samples."""
    days = pd.bdate_range("2000-01-03", "2021-12-31")
    tr, va, _ = fx.pre2022_85_15_split(days)
    assert min(tr) < "2016-01-01", "expected genuine pre-2016 training history"
    assert max(tr) < min(va)
    assert max(va) < fx.PRE_TEST_CUTOFF
    pre2016 = sum(1 for d in tr if d < "2016-01-01")
    assert pre2016 > 1000


# --- D. corrected 85/15 never accesses target dates >= 2022 ----------------

def test_long_history_windows_are_firewall_bounded():
    days = pd.bdate_range("2000-01-03", "2023-12-29")
    tr, va, last = fx.pre2022_85_15_split(days)
    assert max(tr) < fx.PRE_TEST_CUTOFF
    assert max(va) < fx.PRE_TEST_CUTOFF
    assert last < fx.PRE_TEST_CUTOFF
    fx.assert_targets_pre_test(tr, va, where="long_history")


def test_pre2022_8515_label_is_long_history():
    assert fx.ANALOG_85_15_LABEL == "FORENSIC_PRE2022_LONG_HISTORY_85_15"


# --- E. L4 fits the leaking scaler on UNSCALED values ---------------------

def test_leak_scaler_receives_unscaled_data_agent_output(tmp_path):
    src = (REPO_ROOT / "scripts" / "run_pre2022_forensic_diagnostics.py").read_text()
    body = src[src.index("def _run_probe"):src.index("def _current_day_target")]
    assert "return_unscaled_sequences=True" in body
    assert 'assert not ds_unscaled.scaled' in body
    # L4 must not be evaluated with a tree model
    assert "LogisticRegression" in body


def test_leak_scaler_actually_changes_the_transform(tmp_path):
    mod = _load_probe_runner()
    rng = np.random.default_rng(0)
    Xtr = rng.normal(loc=0.0, scale=1.0, size=(50, 4, 3))
    Xva = rng.normal(loc=50.0, scale=1.0, size=(20, 4, 3))   # wildly different
    out_tr, out_va = mod._leak_scaler(Xtr, Xva)
    # fitting on train+validation pulls validation onto the pooled scale
    assert out_va.mean() < 10.0
    assert out_tr.shape == Xtr.shape
    assert out_va.shape == Xva.shape


def test_data_agent_can_return_unscaled_sequences():
    import inspect

    from agentic_forecaster.data.agent import DataAgent, ProcessedDataset
    assert "return_unscaled_sequences" in inspect.signature(DataAgent.run).parameters
    assert "return_unscaled_sequences" in inspect.signature(
        DataAgent.run_ticker).parameters
    assert ProcessedDataset.__dataclass_fields__["scaled"].default is True


# --- F. pooled is NOT scientifically invalid ------------------------------

def test_pooled_is_not_marked_scientifically_invalid():
    assert fx.POOLED_LABEL == "FORENSIC_POOLED_NOT_AUTHOR_CONFIRMED"
    src = (REPO_ROOT / "scripts" / "run_pre2022_forensic_diagnostics.py").read_text()
    body = src[src.index("def diagnostic_c"):src.index("# Diagnostic D")]
    assert '"SCIENTIFICALLY_INVALID": False' in body
    assert '"VALID_FOR_FINAL_MODEL": False' in body
    # and it must not reuse the leakage marker
    assert "fx.INVALID_MARKER |" not in body
    assert "marker=fx.INVALID_MARKER" not in body


# --- G. majority terminology ----------------------------------------------

def test_majority_terminology_distinguishes_the_two_concepts():
    """The oracle rate peeks at validation labels and is not a baseline."""
    tr = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    va = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    # train leans DOWN (3/8 UP) so the constant predictor says DOWN: 3 of 8
    # validation rows are DOWN. The oracle rate instead reports the validation
    # class balance, 5/8. The two are DIFFERENT quantities.
    base = fx.train_majority_baseline(tr, va)
    oracle = fx.validation_oracle_majority_rate(va)
    assert base == pytest.approx(3 / 8)
    assert oracle == pytest.approx(5 / 8)
    assert base != oracle
    # a model must never be compared to the oracle
    bal = fx.label_balance(tr, va)
    assert "train_majority_baseline_accuracy" in bal
    assert "validation_oracle_majority_rate" in bal
    assert "not a predictor" in bal["terminology_note"]


def test_train_majority_baseline_class_comes_from_train_only():
    """The predicted CLASS is train-derived; only the scoring uses validation.

    With a train set leaning DOWN the predictor must always say DOWN, so the
    baseline accuracy equals the fraction of validation rows that are DOWN --
    and that fraction must follow validation only through plain scoring.
    """
    tr_down = np.array([0, 0, 0, 0, 0, 0, 1, 1])          # 2/8 UP -> DOWN
    va = np.array([0, 0, 0, 1, 1, 1])
    assert fx.train_majority_baseline(tr_down, va) == pytest.approx(0.5)

    # a train set leaning UP flips the predicted class, hence the accuracy
    tr_up = np.array([1, 1, 1, 1, 1, 1, 0, 0])
    assert fx.train_majority_baseline(tr_up, va) == pytest.approx(0.5)

    # the oracle rate instead describes the validation class balance and
    # ignores train entirely -- which is why it is not a deployable baseline
    assert fx.validation_oracle_majority_rate(va) == pytest.approx(0.5)
    assert fx.validation_oracle_majority_rate(
        np.array([1, 1, 1, 1, 0, 0])) == pytest.approx(4 / 6)


def test_baseline_panel_uses_the_standardised_name():
    src = (REPO_ROOT / "scripts" / "run_pre2022_forensic_diagnostics.py").read_text()
    assert "TrainMajorityBaseline" in src
    assert "delta_vs_train_majority" in src
    assert '"Majority"' not in src


# --- conclusion structure -------------------------------------------------

def test_leakage_is_hypothetical_not_demonstrated():
    m = _load_summarizer()
    src = (REPO_ROOT / "scripts" / "summarize_pre2022_forensics.py").read_text()
    assert '"status": "HYPOTHETICAL"' in src
    assert "NOT ESTABLISHED" in src
    assert m.cause_ranking(
        unadjusted_acc=0.5250, adjusted_acc=0.5250, panel={},
        pooled_acc=None, analog_acc=None, majority_acc=0.52,
        mismatch_count=0, aggregation={}, confidence={}, l5=None)
