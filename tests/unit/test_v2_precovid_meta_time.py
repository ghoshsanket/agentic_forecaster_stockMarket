"""PRE-COVID meta-learning time safety.

The meta stage, when the programme reaches it, must build its episodes from TRAIN
data only:

* support precedes query;
* every support/query target stays inside the fold's TRAIN period;
* no 2017 label enters a PRECOVID_DEV_A adaptation;
* no 2018 label enters a PRECOVID_DEV_B adaptation;
* no 2019 label enters any architecture-development adaptation;
* no 2020+ label can exist anywhere in the track.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agentic_forecaster.v2 import precovid as pc
from agentic_forecaster.v2.dataset import (
    V2_DEV_FOLD_A,
    V2SequenceDataset,
    build_sample_table,
    resolve_fold_window,
    split_masks,
)
from agentic_forecaster.v2.experiment import load_run_config
from agentic_forecaster.v2.firewall import PostCovidDataAccessError
from agentic_forecaster.v2.meta import MetaConfig, build_meta_episodes, validation_support_frames

REPO_ROOT = Path(__file__).resolve().parents[2]
SMALL_META = MetaConfig(support_size=20, query_size=5, stride=5, inner_steps=1,
                        inner_lr=0.01, meta_step_size=0.05, episodes_per_meta_batch=2,
                        meta_epochs=1, validation_support_size=20,
                        validation_inner_steps=1)


def _declared_folds() -> dict:
    from agentic_forecaster.config import load_config

    return load_config(REPO_ROOT / "configs" / "v2" / "pre_covid" / "base.yaml")["folds"]


# ---------------------------------------------------------------------------
# episodes stay inside the TRAIN period of their fold
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fold_name,train_end", (("PRECOVID_DEV_A", "2016-12-31"),
                                                  ("PRECOVID_DEV_B", "2017-12-31")))
def test_every_episode_target_stays_inside_the_train_period(
        fold_name, train_end, arrays_fixture):
    """Support AND query targets must be <= the fold's train end."""
    window = resolve_fold_window(fold_name, _declared_folds())
    frame = pd.DataFrame({
        "ticker": ["AAA"] * 400,
        "origin_date": pd.bdate_range("2005-01-03", periods=400),
    })
    frame["target_date"] = pd.bdate_range(frame["origin_date"].iloc[0],
                                          periods=400 + 1)[1:]
    frame["row"] = np.arange(400)
    frame["y_direction"] = 1
    # synthetic SampleTable view over the real fixture's arrays
    from agentic_forecaster.v2.dataset import SampleTable

    samples = SampleTable(frame=frame, arrays=arrays_fixture, sequence_length=8)
    mask = split_masks(samples, window)["train"]
    episodes = build_meta_episodes(samples, config=SMALL_META, train_mask=mask,
                                   where=f"unit {fold_name}")
    assert episodes
    for episode in episodes:
        rows = samples.frame.loc[episode.support_indices + episode.query_indices]
        assert (pd.to_datetime(rows["target_date"]) <= pd.Timestamp(train_end)).all()
        assert (pd.to_datetime(rows["origin_date"]) <= pd.Timestamp(train_end)).all()


def test_support_always_precedes_query(arrays_fixture):
    from agentic_forecaster.v2.dataset import SampleTable

    frame = pd.DataFrame({
        "ticker": ["AAA"] * 400,
        "origin_date": pd.bdate_range("2005-01-03", periods=400),
    })
    frame["target_date"] = pd.bdate_range(frame["origin_date"].iloc[0],
                                          periods=401)[1:]
    frame["row"] = np.arange(400)
    frame["y_direction"] = 1
    samples = SampleTable(frame=frame, arrays=arrays_fixture, sequence_length=8)
    window = resolve_fold_window("PRECOVID_DEV_A", _declared_folds())
    episodes = build_meta_episodes(samples, config=SMALL_META,
                                   train_mask=split_masks(samples, window)["train"])
    for episode in episodes:
        episode.validate()
        assert max(episode.support_origin_dates) < min(episode.query_origin_dates)


def test_no_2017_label_enters_a_dev_a_adaptation(arrays_fixture):
    from agentic_forecaster.v2.dataset import SampleTable

    frame = pd.DataFrame({
        "ticker": ["AAA"] * 400,
        "origin_date": pd.bdate_range("2005-01-03", periods=400),
    })
    frame["target_date"] = pd.bdate_range(frame["origin_date"].iloc[0],
                                          periods=401)[1:]
    frame["row"] = np.arange(400)
    frame["y_direction"] = 1
    samples = SampleTable(frame=frame, arrays=arrays_fixture, sequence_length=8)
    window = resolve_fold_window("PRECOVID_DEV_A", _declared_folds())
    episodes = build_meta_episodes(samples, config=SMALL_META,
                                   train_mask=split_masks(samples, window)["train"])
    for episode in episodes:
        rows = samples.frame.loc[episode.support_indices + episode.query_indices]
        assert (rows["target_date"].dt.year <= 2016).all()


def test_no_2018_label_enters_a_dev_b_adaptation(arrays_fixture):
    from agentic_forecaster.v2.dataset import SampleTable

    frame = pd.DataFrame({
        "ticker": ["AAA"] * 500,
        "origin_date": pd.bdate_range("2005-01-03", periods=500),
    })
    frame["target_date"] = pd.bdate_range(frame["origin_date"].iloc[0],
                                          periods=501)[1:]
    frame["row"] = np.arange(500)
    frame["y_direction"] = 1
    samples = SampleTable(frame=frame, arrays=arrays_fixture, sequence_length=8)
    window = resolve_fold_window("PRECOVID_DEV_B", _declared_folds())
    episodes = build_meta_episodes(samples, config=SMALL_META,
                                   train_mask=split_masks(samples, window)["train"])
    for episode in episodes:
        rows = samples.frame.loc[episode.support_indices + episode.query_indices]
        assert (rows["target_date"].dt.year <= 2017).all()


def test_no_2019_label_enters_any_development_adaptation(arrays_fixture):
    """The validation-time support loader for both development folds."""
    from agentic_forecaster.v2.dataset import SampleTable

    frame = pd.DataFrame({
        "ticker": ["AAA"] * 900,
        "origin_date": pd.bdate_range("2005-01-03", periods=900),
    })
    frame["target_date"] = pd.bdate_range(frame["origin_date"].iloc[0],
                                          periods=901)[1:]
    frame["row"] = np.arange(900)
    frame["y_direction"] = 1
    samples = SampleTable(frame=frame, arrays=arrays_fixture, sequence_length=8)
    for fold_name in pc.DEV_FOLDS:
        window = resolve_fold_window(fold_name, _declared_folds())
        support = validation_support_frames(
            samples, ticker="AAA", n_support=SMALL_META.validation_support_size,
            train_end=window.train_end, val_start=window.val_start)
        if support.empty:
            continue
        assert (pd.to_datetime(support["target_date"])
                <= pd.Timestamp(window.train_end)).all()
        assert (pd.to_datetime(support["target_date"])
                < pd.Timestamp(window.val_start)).all()
        assert (pd.to_datetime(support["target_date"]).dt.year <= 2018).all()


def test_a_2020_label_cannot_exist_in_any_episode(arrays_fixture):
    from agentic_forecaster.v2.dataset import SampleTable

    frame = pd.DataFrame({
        "ticker": ["AAA"] * 200,
        "origin_date": pd.bdate_range("2015-01-01", periods=200),
    })
    frame["target_date"] = pd.bdate_range(frame["origin_date"].iloc[0], periods=201)[1:]
    # a MIDDLE row, so it is certainly inside some episode's support/query block
    frame.loc[frame.index[100], "target_date"] = pd.Timestamp("2020-01-02")
    frame["row"] = np.arange(200)
    frame["y_direction"] = 1
    samples = SampleTable(frame=frame, arrays=arrays_fixture, sequence_length=8)
    window = resolve_fold_window("PRECOVID_DEV_A", _declared_folds())

    # with the real split mask the 2020 row is outside the window and therefore
    # never covered
    episodes = build_meta_episodes(samples, config=SMALL_META,
                                   train_mask=split_masks(samples, window)["train"])
    for episode in episodes:
        rows = samples.frame.loc[episode.support_indices + episode.query_indices]
        assert (rows["target_date"] <= pd.Timestamp("2019-12-31")).all()

    # and if a caller hands the episode builder a mask that DOES cover it, the
    # firewall refuses instead of building a task from a post-2019 label
    with pytest.raises(PostCovidDataAccessError):
        build_meta_episodes(samples, config=SMALL_META,
                            train_mask=np.ones(len(samples.frame), dtype=bool),
                            where="unit 2020",
                            final_allowed_date="2019-12-31")


# ---------------------------------------------------------------------------
# the split masks themselves
# ---------------------------------------------------------------------------

def test_development_splits_contain_no_2019_sample(arrays_fixture):
    from agentic_forecaster.v2.dataset import SampleTable

    frame = pd.DataFrame({
        "ticker": ["AAA"] * 900,
        "origin_date": pd.bdate_range("2005-01-03", periods=900),
    })
    frame["target_date"] = pd.bdate_range(frame["origin_date"].iloc[0],
                                          periods=901)[1:]
    frame["row"] = np.arange(900)
    frame["y_direction"] = 1
    samples = SampleTable(frame=frame, arrays=arrays_fixture, sequence_length=8)
    for fold_name in pc.DEV_FOLDS:
        window = resolve_fold_window(fold_name, _declared_folds())
        for split in ("train", "val"):
            mask = split_masks(samples, window)[split]
            dates = pd.to_datetime(samples.frame.loc[mask, "target_date"])
            assert (dates < pd.Timestamp("2019-01-01")).all()


def test_the_meta_config_declares_what_is_adapted():
    payload = SMALL_META.to_dict()
    assert payload["label"] == "REPTILE_STYLE_HEAD_ADAPTER"
    assert "MAML" in payload["not"]
    assert payload["adapted"] == "residual adapter + task heads"
    assert "LSTM" in payload["frozen"]


def test_the_ordinary_v2_meta_folds_are_unchanged():
    """The original V2 programme keeps its own folds and its own boundaries."""
    assert V2_DEV_FOLD_A.val_end == "2019-12-31"
    from agentic_forecaster.v2.dataset import resolve_fold_window as resolve

    assert resolve("V2_DEV_FOLD_A", None).train_end == "2018-12-31"


def test_lockbox_adaptation_support_is_pre_2019_by_construction():
    """Even the lockbox fold's adaptation support ends in 2018."""
    window = resolve_fold_window("PRECOVID_LOCKBOX", _declared_folds())
    assert window.train_end == "2018-12-31"
    assert window.val_start == "2019-01-01"


def test_the_precovid_variant_config_declares_the_regime():
    """When the programme generates the F config it must carry the regime keys."""
    config = load_run_config(
        REPO_ROOT / "configs" / "v2" / "pre_covid" / "v2_b_lstm_transformer.yaml",
        fold="PRECOVID_DEV_A")
    assert config.payload["experiment_regime"] == pc.EXPERIMENT_REGIME
    assert config.payload["pre_covid_mode"] is True
    assert config.final_allowed_date == "2019-12-31"
    assert config.payload["experiment"]["survivorship_bias_label"] == pc.BIAS_LABEL
    assert config.payload["experiment"]["regime_label"] == pc.REGIME_LABEL


def test_dataset_slicing_still_yields_finite_windows(arrays_fixture, targets_fixture):
    samples = build_sample_table(arrays_fixture, targets_fixture, sequence_length=8,
                                 require_targets=["y_direction"],
                                 max_date="2019-12-31",
                                 final_allowed_date="2019-12-31")
    dataset = V2SequenceDataset(samples, samples.frame)
    assert len(dataset) > 0
    assert np.isfinite(dataset[0]["stock_sequence"].numpy()).all()
    assert np.isfinite(dataset[0]["context_sequence"].numpy()).all()