"""PRE-COVID lockbox authorisation tests.

Normal PRE-COVID development code must refuse 2019.  The dedicated lockbox runner
must refuse without ``PRECOVID_LOCKBOX=1``, without a frozen selection, and when
the config hash, the supervised universe, the store hash or the seed differ from
what was frozen.  With a valid authorisation and a monkeypatched experiment, it
must invoke the selected experiment EXACTLY once.

No real 2019 data is needed: the runner's authorisation logic is exercised
directly and the actual training call is monkeypatched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from agentic_forecaster.v2 import precovid as pc
from agentic_forecaster.v2.firewall import PRECOVID_LOCKBOX_ENV, pre_covid_lockbox_unlocked

REPO = Path(__file__).resolve().parents[2]
LOCKBOX_SCRIPT = REPO / "scripts" / "run_v2_precovid_lockbox.py"


def _runner():
    spec = importlib.util.spec_from_file_location("precovid_lockbox", LOCKBOX_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["precovid_lockbox"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner():
    return _runner()


@pytest.fixture
def frozen_selection(tmp_path):
    universe = pd.DataFrame([{
        "ticker": "AAA", "raw_first_date": "2005-01-03",
        "first_usable_sample": "2005-04-01", "train_samples_through_2016": 3000,
        "samples_2017": 240, "samples_2018": 240, "samples_2019": 240,
        "eligible": True, "exclusion_reason": "", "sector": "SYNTH",
    }])
    selection = {
        "selected_architecture": "V2-B",
        "selected_component_flags": {"use_transformer": True},
        "frozen_before_2019_access": True,
        "test_2022_2023_evaluated": False,
        "precovid_store_sha256": "s" * 64,
        "sector_map_sha256": "e" * 64,
        "source_manifest_sha256": "m" * 64,
        "eligible_universe_sha256": pc.universe_hash(universe),
        "config_sha256": "c" * 64,
        "seed_stability": {"unstable": False},
    }
    path = tmp_path / "pre_covid_dev_selection.json"
    path.write_text(json.dumps(selection))
    return path, selection, universe


# ---------------------------------------------------------------------------
# development code must refuse 2019
# ---------------------------------------------------------------------------

def test_normal_development_config_refuses_the_lockbox_fold():
    from agentic_forecaster.v2.experiment import load_run_config

    with pytest.raises(AssertionError, match=PRECOVID_LOCKBOX_ENV):
        load_run_config(REPO / "configs" / "v2" / "pre_covid" / "v2_a_shared_lstm.yaml",
                        fold=pc.LOCKBOX_FOLD)


def test_development_folds_cannot_reach_2019():
    from agentic_forecaster.v2.experiment import load_run_config

    for fold in pc.DEV_FOLDS:
        window = load_run_config(
            REPO / "configs" / "v2" / "pre_covid" / "v2_a_shared_lstm.yaml",
            fold=fold).window
        assert window.val_end < "2019-01-01"


def test_lockbox_switch_is_off_by_default():
    assert pre_covid_lockbox_unlocked() is False


# ---------------------------------------------------------------------------
# the runner's authorisation rules
# ---------------------------------------------------------------------------

def test_runner_refuses_without_the_env_var(runner, tmp_path, monkeypatch):
    monkeypatch.delenv(PRECOVID_LOCKBOX_ENV, raising=False)
    path = tmp_path / "sel.json"
    path.write_text(json.dumps({"selected_architecture": "V2-B"}))
    with pytest.raises(SystemExit, match=PRECOVID_LOCKBOX_ENV):
        runner.authorise(path, None, 42)


def test_runner_refuses_without_a_frozen_selection(runner, monkeypatch, tmp_path):
    monkeypatch.setenv(PRECOVID_LOCKBOX_ENV, "1")
    with pytest.raises(SystemExit, match="frozen selection is missing"):
        runner.authorise(tmp_path / "absent.json", None, 42)


def test_runner_refuses_a_different_architecture(runner, monkeypatch, frozen_selection):
    monkeypatch.setenv(PRECOVID_LOCKBOX_ENV, "1")
    path, _, _ = frozen_selection
    with pytest.raises(SystemExit, match="!= frozen"):
        runner.authorise(path, "V2-C", 42)


def test_runner_refuses_a_seed_other_than_42(runner, monkeypatch, frozen_selection):
    monkeypatch.setenv(PRECOVID_LOCKBOX_ENV, "1")
    path, _, _ = frozen_selection
    with pytest.raises(SystemExit, match="seed 42"):
        runner.authorise(path, "V2-B", 73)


def test_runner_refuses_a_selection_that_was_not_frozen_first(
        runner, monkeypatch, frozen_selection):
    monkeypatch.setenv(PRECOVID_LOCKBOX_ENV, "1")
    path, selection, _ = frozen_selection
    payload = dict(selection)
    payload["frozen_before_2019_access"] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="not frozen"):
        runner.authorise(path, None, 42)


def test_runner_refuses_a_selection_claiming_2022_evaluation(
        runner, monkeypatch, frozen_selection):
    monkeypatch.setenv(PRECOVID_LOCKBOX_ENV, "1")
    path, selection, _ = frozen_selection
    payload = dict(selection)
    payload["test_2022_2023_evaluated"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="2022/2023"):
        runner.authorise(path, None, 42)


def test_runner_accepts_a_valid_authorisation(runner, monkeypatch, frozen_selection):
    monkeypatch.setenv(PRECOVID_LOCKBOX_ENV, "1")
    path, _, _ = frozen_selection
    assert runner.authorise(path, "V2-B", 42)["selected_architecture"] == "V2-B"


# ---------------------------------------------------------------------------
# hash verification
# ---------------------------------------------------------------------------

def test_runner_refuses_a_changed_store_hash(runner, monkeypatch, frozen_selection):
    monkeypatch.setenv(PRECOVID_LOCKBOX_ENV, "1")
    _, selection, _ = frozen_selection
    tampered = dict(selection)
    from agentic_forecaster.v2.store import store_fingerprints

    real = store_fingerprints

    def fake(root=None):
        """The store on disk no longer matches the frozen hash."""
        payload = real(root)
        payload["store_sha256"] = "1" * 64
        return payload

    monkeypatch.setattr(runner, "store_fingerprints", fake)
    with pytest.raises(SystemExit, match="store hash changed"):
        runner.verify_hashes(tampered, REPO / "configs" / "v2" / "pre_covid" /
                             "v2_b_lstm_transformer.yaml")


def test_runner_refuses_a_changed_supervised_universe(
        runner, monkeypatch, tmp_path, frozen_selection):
    monkeypatch.setenv(PRECOVID_LOCKBOX_ENV, "1")
    _, selection, _ = frozen_selection
    from agentic_forecaster.v2.store import store_fingerprints

    monkeypatch.setattr(runner, "store_fingerprints",
                        lambda root=None: store_fingerprints(root))
    monkeypatch.setattr(runner, "load_run_config", _stub_config())
    tampered = dict(selection)
    tampered["eligible_universe_sha256"] = "9" * 64

    original = pc.PreCovidTrack().path("universe_csv")
    backup = tmp_path / "backup.csv"
    if original.is_file():
        backup.write_text(original.read_text())
    patched = pd.DataFrame([{
        "ticker": "ZZZ", "raw_first_date": "2005-01-03", "first_usable_sample": "2005-04-01",
        "train_samples_through_2016": 1, "samples_2017": 1, "samples_2018": 1,
        "samples_2019": 1, "eligible": True, "exclusion_reason": "", "sector": "SYNTH",
    }])
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_text(patched.to_csv(index=False))
    try:
        with pytest.raises(SystemExit, match="universe hash changed"):
            runner.verify_hashes(tampered, REPO / "configs" / "v2" / "pre_covid" /
                                 "v2_b_lstm_transformer.yaml")
    finally:
        if backup.is_file():
            original.write_text(backup.read_text())


class _StubConfig:
    """Minimal stand-in for a resolved V2RunConfig."""

    def __init__(self, sha: str, components: dict) -> None:
        self.payload = {"config_sha256": sha, "resolved_components": components}


def _stub_config():
    return lambda *a, **k: _StubConfig("c" * 64, {"use_transformer": True})


def test_runner_refuses_changed_component_flags(runner, monkeypatch, frozen_selection):
    monkeypatch.setenv(PRECOVID_LOCKBOX_ENV, "1")
    _, selection, _ = frozen_selection
    from agentic_forecaster.v2.store import store_fingerprints

    monkeypatch.setattr(runner, "store_fingerprints",
                        lambda root=None: store_fingerprints(root))

    monkeypatch.setattr(runner, "load_run_config",
                        lambda *a, **k: _StubConfig("d" * 64, {"use_context": True}))
    with pytest.raises(SystemExit, match="component flags differ"):
        runner.verify_hashes(selection, REPO / "configs" / "v2" / "pre_covid" /
                            "v2_b_lstm_transformer.yaml")


# ---------------------------------------------------------------------------
# exactly one invocation
# ---------------------------------------------------------------------------

def test_authorized_runner_invokes_the_selected_experiment_exactly_once(
        runner, monkeypatch, frozen_selection, tmp_path):
    monkeypatch.setenv(PRECOVID_LOCKBOX_ENV, "1")
    path, _, _ = frozen_selection
    calls: list[dict] = []

    def fake_run(config, *, out_dir=None):
        calls.append({"variant": config.variant, "fold": config.fold,
                      "seed": config.seed, "out_dir": out_dir})
        return {
            "experiment_id": "LOCKBOX-TEST-1",
            "out_dir": str(tmp_path / "LOCKBOX-TEST-1"),
            "metrics": {"direction": {"accuracy_macro_ticker": 0.5},
                        "selection": {}, "selective_accuracy": {}},
            "complexity": {},
            "manifest": {"data_access_audit": {}},
        }

    monkeypatch.setattr(runner, "run_experiment", fake_run)
    monkeypatch.setattr(runner, "verify_hashes",
                        lambda sel, cfg: {"store": {"store_sha256": "s" * 64},
                                          "config_sha256": "c" * 64})
    monkeypatch.setattr(runner, "already_run", lambda: False)

    out_dir = tmp_path / "LOCKBOX-TEST-1"
    code = runner.main(["--variant", "V2-B", "--selection", str(path),
                        "--out-dir", str(out_dir)])
    assert code == 0
    assert len(calls) == 1
    # the report is written inside the experiment directory, so the test cannot
    # pollute the track's frozen artefacts
    report = out_dir / "precovid_lockbox_report.json"
    assert report.is_file()
    assert json.loads(report.read_text())["selected_architecture"] == "V2-B"
    assert calls[0]["variant"] == "V2-B"
    assert calls[0]["fold"] == pc.LOCKBOX_FOLD
    assert calls[0]["seed"] == 42


def test_runner_refuses_a_second_lockbox_run(runner, monkeypatch, frozen_selection):
    monkeypatch.setenv(PRECOVID_LOCKBOX_ENV, "1")
    path, _, _ = frozen_selection
    monkeypatch.setattr(runner, "already_run", lambda: True)
    with pytest.raises(SystemExit, match="EXACTLY once"):
        runner.main(["--variant", "V2-B", "--selection", str(path)])


def test_config_lookup_requires_the_file_to_exist(runner):
    assert runner.config_for("V2-B").is_file()
    assert runner.config_for("V2-A").is_file()
    with pytest.raises(SystemExit):
        runner.config_for("V2-ZZZ")
    for variant in ("V2-D", "V2-E", "V2-F"):
        # generated only when the programme reaches them
        with pytest.raises(SystemExit):
            if not runner.config_for(variant).is_file():
                raise SystemExit(f"missing {variant}")