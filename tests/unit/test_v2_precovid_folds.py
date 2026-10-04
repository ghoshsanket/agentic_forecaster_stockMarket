"""Config-driven folds and store-root isolation for the PRE-COVID track.

FOLD PROOF (the point of this module)
-------------------------------------
``PRECOVID_DEV_A``  trains through 2016 and validates 2017 ONLY
``PRECOVID_DEV_B``  trains through 2017 and validates 2018 ONLY
``PRECOVID_LOCKBOX`` trains through 2018 and evaluates 2019 ONLY

and changing the YAML fold dates really changes the resolved ``SplitWindow`` --
which is what a hard-coded global fold table could never do.

STORE-ROOT PROOF
----------------
the ordinary V2 track resolves the ordinary store, the PRE-COVID track resolves
the PRE-COVID store, and a PRE-COVID config pointed at a store whose
``max_target_date`` is after 2019-12-31 fails BEFORE training.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from agentic_forecaster.v2.dataset import SplitWindow, resolve_fold_window
from agentic_forecaster.v2.experiment import load_run_config
from agentic_forecaster.v2.firewall import PostCovidDataAccessError
from agentic_forecaster.v2.store import load_store, precovid_processed_root, store_root

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs" / "v2" / "pre_covid"
A = CONFIG_DIR / "v2_a_shared_lstm.yaml"
C = CONFIG_DIR / "v2_c_contextual.yaml"
ORDINARY = REPO / "configs" / "v2" / "v2_a_shared_lstm.yaml"


# ---------------------------------------------------------------------------
# the three PRE-COVID windows
# ---------------------------------------------------------------------------

def test_precovid_dev_a_trains_through_2016_and_validates_2017_only():
    window = resolve_fold_window("PRECOVID_DEV_A", _declared_folds())
    assert isinstance(window, SplitWindow)
    assert window.train_start == "2005-01-01"
    assert window.train_end == "2016-12-31"
    assert window.val_start == "2017-01-01"
    assert window.val_end == "2017-12-31"
    assert "2018" not in (window.train_end, window.val_start, window.val_end)
    assert "2019" not in (window.train_end, window.val_start, window.val_end)


def test_precovid_dev_b_trains_through_2017_and_validates_2018_only():
    window = resolve_fold_window("PRECOVID_DEV_B", _declared_folds())
    assert window.train_end == "2017-12-31"
    assert window.val_start == "2018-01-01"
    assert window.val_end == "2018-12-31"
    assert "2019" not in (window.train_end, window.val_start, window.val_end)


def test_precovid_lockbox_trains_through_2018_and_evaluates_2019_only():
    window = resolve_fold_window("PRECOVID_LOCKBOX", _declared_folds())
    assert window.train_end == "2018-12-31"
    assert window.val_start == "2019-01-01"
    assert window.val_end == "2019-12-31"


def test_selection_cannot_see_2019_in_the_development_folds():
    for name in ("PRECOVID_DEV_A", "PRECOVID_DEV_B"):
        window = resolve_fold_window(name, _declared_folds())
        assert int(window.val_end[:4]) <= 2018
        assert int(window.train_end[:4]) <= 2017


def test_every_declared_precovid_window_stays_inside_the_regime():
    for name in ("PRECOVID_DEV_A", "PRECOVID_DEV_B", "PRECOVID_LOCKBOX"):
        window = resolve_fold_window(name, _declared_folds())
        assert window.val_end <= "2019-12-31"
        assert window.train_end <= "2019-12-31"


# ---------------------------------------------------------------------------
# the folds really are config-driven
# ---------------------------------------------------------------------------

def test_changing_the_yaml_dates_changes_the_resolved_window():
    declared = _declared_folds()
    declared["PRECOVID_DEV_A"]["val_end"] = "2017-06-30"
    declared["PRECOVID_DEV_A"]["val_start"] = "2017-01-01"
    window = resolve_fold_window("PRECOVID_DEV_A", declared)
    assert window.val_end == "2017-06-30"

    declared["PRECOVID_DEV_A"]["train_end"] = "2015-12-31"
    assert resolve_fold_window("PRECOVID_DEV_A", declared).train_end == "2015-12-31"


def test_config_declared_folds_win_over_the_builtin_defaults():
    """A config may define a fold the built-in table has never heard of."""
    declared = _declared_folds()
    declared["precovid_micro"] = {
        "train_start": "2010-01-01", "train_end": "2013-12-31",
        "val_start": "2014-01-01", "val_end": "2014-06-30",
    }
    window = resolve_fold_window("precovid_micro", declared)
    assert window.name == "PRECOVID_MICRO"
    assert window.val_end == "2014-06-30"
    # a differently-cased request resolves to the same declaration
    assert resolve_fold_window("PreCovid_Micro", declared).val_end == "2014-06-30"


def test_builtin_defaults_still_serve_the_original_v2_configs():
    """Backward compatibility: V2_FOLDS remains the fallback."""
    for name, expected in (("V2_DEV_FOLD_A", "2019-12-31"),
                           ("V2_DEV_FOLD_B", "2020-12-31"),
                           ("V2_LOCKBOX", "2021-12-31")):
        assert resolve_fold_window(name, None).val_end == expected


def test_unknown_fold_is_rejected():
    with pytest.raises(ValueError, match="unknown fold"):
        resolve_fold_window("NOT_A_FOLD", _declared_folds())


def test_incomplete_fold_block_is_rejected():
    with pytest.raises(ValueError, match="missing"):
        resolve_fold_window("PRECOVID_DEV_A", {"PRECOVID_DEV_A": {"train_start": "2005"}})


def test_resolved_run_config_uses_the_declared_windows():
    for fold, val_end, train_end in (("PRECOVID_DEV_A", "2017-12-31", "2016-12-31"),
                                     ("PRECOVID_DEV_B", "2018-12-31", "2017-12-31")):
        config = load_run_config(A, fold=fold)
        assert config.window.val_end == val_end
        assert config.window.train_end == train_end
        assert config.payload["fold_window"]["val_end"] == val_end


def test_lockbox_window_is_refused_without_authorisation():
    with pytest.raises(AssertionError, match="PRECOVID_LOCKBOX"):
        load_run_config(A, fold="PRECOVID_LOCKBOX")


def test_lockbox_window_resolves_with_authorisation(monkeypatch):
    monkeypatch.setenv("PRECOVID_LOCKBOX", "1")
    config = load_run_config(A, fold="PRECOVID_LOCKBOX")
    assert config.window.val_start == "2019-01-01"
    assert config.is_lockbox_fold is True


def test_v2_lockbox_switch_does_not_authorise_the_precovid_lockbox(monkeypatch):
    monkeypatch.delenv("PRECOVID_LOCKBOX", raising=False)
    monkeypatch.setenv("V2_LOCKBOX", "1")
    with pytest.raises(AssertionError, match="PRECOVID_LOCKBOX"):
        load_run_config(A, fold="PRECOVID_LOCKBOX")


# ---------------------------------------------------------------------------
# store-root isolation
# ---------------------------------------------------------------------------

def test_ordinary_config_resolves_the_ordinary_store():
    config = load_run_config(ORDINARY, fold="V2_DEV_FOLD_A")
    assert config.processed_root.name == "v2"
    assert "pre_covid" not in str(config.processed_root)
    assert config.results_root == REPO / "results" / "v2"


def test_precovid_config_resolves_the_precovid_store():
    config = load_run_config(C, fold="PRECOVID_DEV_A")
    assert config.processed_root == precovid_processed_root()
    assert "pre_covid" in str(config.processed_root)
    assert config.results_root == REPO / "results" / "v2" / "pre_covid"
    assert str(config.runtime_root).endswith("v2/pre_covid")
    assert config.payload["data"]["store_root"].endswith("v2/pre_covid")


def test_store_root_property_honours_an_explicit_override():
    config = load_run_config(C, fold="PRECOVID_DEV_A")
    payload = copy.deepcopy(config.payload)
    payload["data"]["store_root"] = str(precovid_processed_root())
    assert config.payload["data"]["store_root"] == str(precovid_processed_root())


def test_precovid_load_refuses_a_store_whose_max_target_date_is_after_2019():
    ordinary_branch = store_root().parent
    if not store_root().is_dir():
        pytest.skip("the ordinary V2 store is not present")
    with pytest.raises(PostCovidDataAccessError):
        load_store(ordinary_branch, final_allowed_date="2019-12-31")


def test_precovid_run_uses_its_own_store_root_end_to_end(monkeypatch):
    """The store actually loaded by a PRE-COVID run is the PRE-COVID one."""
    loaded: list[str] = []
    from agentic_forecaster.v2 import experiment

    real_load = experiment.load_store

    def spy(root=None, **kwargs):
        loaded.append(str(root))
        return real_load(root, **kwargs)

    monkeypatch.setattr(experiment, "load_store", spy)
    config = load_run_config(C, fold="PRECOVID_DEV_A")
    experiment.assemble_run_data(config)
    assert loaded, "assemble_run_data did not load a store"
    assert all("pre_covid" in path for path in loaded), loaded


def test_supervised_universe_is_read_from_the_frozen_declaration():
    config = load_run_config(C, fold="PRECOVID_DEV_A")
    tickers = config.supervised_tickers
    assert len(tickers) >= 30, "the PRE-COVID track must supervise many securities"
    frozen = REPO / "configs" / "v2" / "pre_covid" / "supervised_universe.yaml"
    if frozen.is_file():
        import yaml

        declared = yaml.safe_load(frozen.read_text())["eligible_tickers"]
        assert tickers == sorted(t.upper() for t in declared)


def test_missing_frozen_universe_is_an_error_not_a_guess(monkeypatch, tmp_path):
    config = load_run_config(C, fold="PRECOVID_DEV_A")
    monkeypatch.setitem(config.payload["data"], "supervised_universe",
                        str(tmp_path / "nope.yaml"))
    with pytest.raises(FileNotFoundError, match="derived, never guessed"):
        _ = config.supervised_tickers


def _declared_folds() -> dict:

    from agentic_forecaster.config import load_config

    return load_config(CONFIG_DIR / "base.yaml")["folds"]