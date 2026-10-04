"""V2-F meta-learning tests, on synthetic fixtures only.

The properties asserted here are the ones that would invalidate the meta stage if
they did not hold:

1. support dates always precede query dates;
2. no episode crosses a fold boundary or escapes the TRAIN window;
3. the shared encoder stays frozen during the inner adaptation;
4. the adapter / head parameters DO change during the inner adaptation;
5. query predictions can change after adaptation;
6. no validation label is consumed by the support loader;
7. no target on/after 2022 can enter an episode;
8. reloading the meta checkpoint reproduces the PRE-adaptation base prediction.

It is also asserted that this scheme is labelled honestly: Reptile-style head /
adapter adaptation, NOT MAML.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from agentic_forecaster.v2.dataset import (
    V2_DEV_FOLD_A,
    V2SequenceDataset,
    build_sample_table,
    split_masks,
)
from agentic_forecaster.v2.meta import (
    META_LABEL,
    NOT_MAML,
    MetaConfig,
    MetaEpisode,
    ReptileMetaTrainer,
    build_meta_episodes,
    validation_support_frames,
)
from agentic_forecaster.v2.model import ContextualLSTMTransformer

from .v2_fixtures import make_model_config

SMALL_META = MetaConfig(support_size=20, query_size=5, stride=5, inner_steps=2,
                        inner_lr=0.05, meta_step_size=0.10,
                        episodes_per_meta_batch=2, meta_epochs=2,
                        validation_support_size=20, validation_inner_steps=2)


def _meta_model(samples):
    """A V2-F model whose input widths match the synthetic sample table."""
    config = make_model_config(
        use_adapter=True, use_multitask=True, use_film=True,
        n_stock_features=samples.arrays.n_stock_features,
        n_context_features=samples.arrays.n_context_features,
        n_regime_features=samples.arrays.n_regime_features,
        n_tickers=max(len(samples.arrays.ticker_vocab), 1),
        n_sectors=max(len(samples.arrays.sector_vocab), 1),
        ticker_vocab=list(samples.arrays.ticker_vocab),
        sector_vocab=list(samples.arrays.sector_vocab),
    )
    torch.manual_seed(11)
    return ContextualLSTMTransformer(config), config


def _trainer(model, config=None):
    return ReptileMetaTrainer(model, config or SMALL_META, device="cpu", batch_size=16)


# ---------------------------------------------------------------------------
# labelling
# ---------------------------------------------------------------------------

def test_scheme_is_labelled_honestly():
    assert META_LABEL == "REPTILE_STYLE_HEAD_ADAPTER"
    assert "MAML" in NOT_MAML and "NOT" in NOT_MAML
    assert "maml" not in META_LABEL.lower().replace("_style_head_adapter", "")


def test_meta_config_declares_what_is_and_is_not_adapted():
    payload = SMALL_META.to_dict()
    assert payload["adapted"] == "residual adapter + task heads"
    assert "LSTM" in payload["frozen"] and "Transformer" in payload["frozen"]


# ---------------------------------------------------------------------------
# 1. chronology
# ---------------------------------------------------------------------------

def test_support_dates_precede_query_dates(samples):
    masks = split_masks(samples, V2_DEV_FOLD_A)
    episodes = build_meta_episodes(samples, config=SMALL_META, train_mask=masks["train"])
    assert episodes, "no episodes were built from the TRAIN window"
    for episode in episodes:
        episode.validate()
        assert max(episode.support_origin_dates) < min(episode.query_origin_dates)
        assert len(episode.support_indices) == SMALL_META.support_size
        assert len(episode.query_indices) == SMALL_META.query_size


def test_episodes_are_chronological_blocks_not_random_mixtures(samples):
    masks = split_masks(samples, V2_DEV_FOLD_A)
    episodes = build_meta_episodes(samples, config=SMALL_META, train_mask=masks["train"])
    frame = samples.frame
    for episode in episodes[:20]:
        support = frame.loc[episode.support_indices].sort_values("origin_date")
        query = frame.loc[episode.query_indices].sort_values("origin_date")
        assert list(support["origin_date"]) == sorted(support["origin_date"])
        assert list(query["origin_date"]) == sorted(query["origin_date"])
        assert support["origin_date"].max() < query["origin_date"].min()


def test_episode_validation_rejects_a_backwards_episode():
    dates = pd.bdate_range("2020-01-01", periods=4)
    bad = MetaEpisode(ticker="AAA", support_indices=[2, 3], query_indices=[0, 1],
                      support_origin_dates=[dates[2], dates[3]],
                      query_origin_dates=[dates[0], dates[1]])
    with pytest.raises(ValueError, match="query date precedes"):
        bad.validate()


def test_a_meta_task_is_a_security_plus_an_episode_not_just_a_security(samples):
    masks = split_masks(samples, V2_DEV_FOLD_A)
    episodes = build_meta_episodes(samples, config=SMALL_META, train_mask=masks["train"])
    by_ticker: dict[str, list[MetaEpisode]] = {}
    for episode in episodes:
        by_ticker.setdefault(episode.ticker, []).append(episode)
    assert len(episodes) > len(by_ticker), (
        "a meta task must be a security PLUS a chronological episode, so there must "
        "be more tasks than securities")


# ---------------------------------------------------------------------------
# 2. fold boundaries and the 2022 firewall
# ---------------------------------------------------------------------------

def test_no_episode_crosses_the_training_window(samples):
    masks = split_masks(samples, V2_DEV_FOLD_A)
    episodes = build_meta_episodes(samples, config=SMALL_META, train_mask=masks["train"])
    frame = samples.frame
    train_max = frame.loc[masks["train"], "target_date"].max()
    for episode in episodes:
        rows = frame.loc[episode.support_indices + episode.query_indices]
        assert rows["target_date"].max() <= train_max
        assert (pd.to_datetime(rows["origin_date"])
                <= pd.Timestamp(V2_DEV_FOLD_A.train_end)).all()


def test_episodes_never_include_validation_rows(samples):
    masks = split_masks(samples, V2_DEV_FOLD_A)
    episodes = build_meta_episodes(samples, config=SMALL_META, train_mask=masks["train"])
    frame = samples.frame
    for episode in episodes:
        rows = frame.loc[episode.support_indices + episode.query_indices]
        assert (pd.to_datetime(rows["origin_date"])
                < pd.Timestamp(V2_DEV_FOLD_A.val_start)).all()


def test_no_episode_can_contain_a_2022_target(samples):
    masks = split_masks(samples, V2_DEV_FOLD_A)
    frame = samples.frame.copy()
    poisoned = type(samples)(frame=frame, arrays=samples.arrays,
                            sequence_length=samples.sequence_length)
    train_mask = masks["train"].copy()
    # move a MIDDLE TRAIN sample's target into 2022 (a middle row is guaranteed to
    # sit inside some episode): the builder must refuse rather than quietly build a
    # task from a firewalled label
    middle = poisoned.frame.loc[train_mask].index[
        len(poisoned.frame.loc[train_mask]) // 2]
    poisoned.frame.loc[middle, "target_date"] = pd.Timestamp("2022-03-01")
    with pytest.raises(Exception, match="2022|refusing target date"):
        build_meta_episodes(poisoned, config=SMALL_META, train_mask=train_mask,
                            where="unit-test")


# ---------------------------------------------------------------------------
# 3/4. what moves during the inner loop and what does not
# ---------------------------------------------------------------------------

def test_encoder_stays_frozen_while_adapter_and_heads_move(samples):
    model, _ = _meta_model(samples)
    trainer = _trainer(model)
    masks = split_masks(samples, V2_DEV_FOLD_A)
    episodes = build_meta_episodes(samples, config=SMALL_META, train_mask=masks["train"])
    episode = episodes[0]

    adapted, losses = trainer._adapt_episode(samples, episode, trainer.meta_init)
    assert losses, "the inner loop produced no loss"

    # the task-specific parameters returned are exactly the adaptable ones, and
    # they differ from the meta initialisation they started from
    assert set(adapted) == set(model.adaptable_parameter_names())
    moved = [name for name, tensor in adapted.items()
             if not torch.allclose(tensor, trainer.meta_init[name])]
    assert moved, "the inner adaptation changed nothing at all"

    # the shared encoder is untouched: it is absent from the adapted set and its
    # own parameters are bit-identical to their pre-adaptation values
    shared = set(model.shared_parameter_names())
    assert not (shared & set(adapted))
    before_shared = model.state_dict()
    trainer._adapt_episode(samples, episode, trainer.meta_init)
    for name, parameter in model.named_parameters():
        if name in shared:
            assert torch.allclose(parameter.detach(), before_shared[name]), (
                f"shared encoder parameter {name} moved during the inner adaptation")


def test_inner_adaptation_returns_the_model_to_its_pre_adaptation_state(samples):
    model, _ = _meta_model(samples)
    trainer = _trainer(model)
    masks = split_masks(samples, V2_DEV_FOLD_A)
    episodes = build_meta_episodes(samples, config=SMALL_META, train_mask=masks["train"])
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    trainer._adapt_episode(samples, episodes[0], trainer.meta_init)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.detach(), before[name], rtol=0, atol=0)


def test_meta_training_moves_only_the_meta_initialisation(samples):
    model, _ = _meta_model(samples)
    trainer = _trainer(model)
    masks = split_masks(samples, V2_DEV_FOLD_A)
    before = {k: v.clone() for k, v in trainer.meta_init.items()}
    result = trainer.fit(samples, masks["train"], seed=5, where="unit-test")
    moved = [k for k, v in trainer.meta_init.items() if not torch.allclose(v, before[k])]
    assert moved, "Reptile must move the meta initialisation towards the task parameters"
    assert all(k in set(model.adaptable_parameter_names()) for k in moved)
    assert result["label"] == META_LABEL
    assert len(result["history"]) == SMALL_META.meta_epochs


# ---------------------------------------------------------------------------
# 5. predictions can change after adaptation
# ---------------------------------------------------------------------------

def test_query_predictions_change_after_adaptation(samples):
    model, _ = _meta_model(samples)
    trainer = _trainer(model)
    masks = split_masks(samples, V2_DEV_FOLD_A)
    query_rows = samples.frame.loc[masks["val"]].head(24).reset_index(drop=True)
    dataset = V2SequenceDataset(samples, query_rows)
    batch = {k: torch.stack([dataset[i][k] for i in range(len(dataset))])
             for k in dataset[0]}

    def predict():
        model.eval()
        with torch.no_grad():
            return model(batch["stock_sequence"], batch["ticker_id"],
                         context_sequence=batch["context_sequence"],
                         sector_id=batch["sector_id"],
                         regime_vector=batch["regime_vector"])["direction_logit"].clone()

    before = predict()
    adapted = trainer.adapt_for_prediction(samples, support_frame=query_rows,
                                           ticker=query_rows["ticker"].iloc[0])
    from agentic_forecaster.v2.meta import _load_adaptable

    _load_adaptable(model, adapted)
    after = predict()
    assert not torch.allclose(before, after, atol=1e-8), (
        "adaptation left every query prediction unchanged")


def test_score_with_adaptation_reports_a_finite_delta(samples):
    model, _ = _meta_model(samples)
    trainer = _trainer(model)
    masks = split_masks(samples, V2_DEV_FOLD_A)
    support_rows = samples.frame.loc[masks["train"]].groupby("ticker").head(
        SMALL_META.validation_support_size)
    query = samples.frame.loc[masks["val"]].head(16).reset_index(drop=True)
    ticker = str(query["ticker"].iloc[0])
    support = support_rows.loc[support_rows["ticker"] == ticker].reset_index(drop=True)
    report = trainer.score_with_adaptation(samples, support_frame=support,
                                          query_frame=query, ticker=ticker)
    assert np.isfinite(report["loss_before_adaptation"])
    assert np.isfinite(report["loss_after_adaptation"])
    assert report["delta"] == pytest.approx(
        report["loss_after_adaptation"] - report["loss_before_adaptation"])
    assert report["n_query"] == len(query)


# ---------------------------------------------------------------------------
# 6. the support loader never reads a validation label
# ---------------------------------------------------------------------------

def test_validation_support_frames_only_contains_train_rows(samples):
    for ticker in sorted(samples.frame["ticker"].unique()):
        support = validation_support_frames(
            samples, ticker=ticker, n_support=SMALL_META.validation_support_size,
            train_end=V2_DEV_FOLD_A.train_end, val_start=V2_DEV_FOLD_A.val_start)
        if support.empty:
            continue
        assert len(support) <= SMALL_META.validation_support_size
        assert (pd.to_datetime(support["target_date"])
                <= pd.Timestamp(V2_DEV_FOLD_A.train_end)).all()
        assert (pd.to_datetime(support["target_date"])
                < pd.Timestamp(V2_DEV_FOLD_A.val_start)).all()
        assert set(support["y_direction"]).issubset({0, 1})
        # it really is the most recent TRAIN slice
        newest_train = samples.frame.loc[
            (samples.frame["ticker"] == ticker)
            & (pd.to_datetime(samples.frame["target_date"])
               <= pd.Timestamp(V2_DEV_FOLD_A.train_end))].sort_values("origin_date")
        assert (pd.to_datetime(support["origin_date"]).to_numpy()
                == pd.to_datetime(newest_train["origin_date"].tail(
                    len(support)).to_numpy())).all()


def test_support_loader_does_not_depend_on_the_validation_labels(samples):
    """Corrupting every validation label must not change the support set."""
    masks = split_masks(samples, V2_DEV_FOLD_A)
    ticker = str(samples.frame.loc[masks["val"], "ticker"].iloc[0])
    original = validation_support_frames(
        samples, ticker=ticker, n_support=SMALL_META.validation_support_size,
        train_end=V2_DEV_FOLD_A.train_end, val_start=V2_DEV_FOLD_A.val_start)
    corrupted_frame = samples.frame.copy()
    corrupted_frame.loc[masks["val"], "y_direction"] = 1 - corrupted_frame.loc[
        masks["val"], "y_direction"]
    corrupted = type(samples)(frame=corrupted_frame, arrays=samples.arrays,
                              sequence_length=samples.sequence_length)
    after = validation_support_frames(
        corrupted, ticker=ticker, n_support=SMALL_META.validation_support_size,
        train_end=V2_DEV_FOLD_A.train_end, val_start=V2_DEV_FOLD_A.val_start)
    pd.testing.assert_frame_equal(
        original.drop(columns=["y_direction"]), after.drop(columns=["y_direction"]))


# ---------------------------------------------------------------------------
# 8. checkpoint round trip reproduces the pre-adaptation base prediction
# ---------------------------------------------------------------------------

def test_meta_checkpoint_reload_reproduces_the_base_prediction(samples, tmp_path):
    from agentic_forecaster.v2.checkpoint import load_checkpoint, save_checkpoint
    from agentic_forecaster.v2.dataset import GroupFeatureScaler, V2ScalerBundle

    model, config = _meta_model(samples)
    trainer = _trainer(model)
    masks = split_masks(samples, V2_DEV_FOLD_A)
    train_rows = samples.frame.loc[masks["train"]].reset_index(drop=True)
    scalers = V2ScalerBundle(
        stock=GroupFeatureScaler(samples.arrays.stock_features).fit(
            np.zeros((len(train_rows), len(samples.arrays.stock_features)))),
        context=GroupFeatureScaler(samples.arrays.context_features,
                                   passthrough=samples.arrays.percentile_features).fit(
            np.zeros((len(train_rows),
                      len(samples.arrays.context_features)
                      - len(samples.arrays.percentile_features)))),
    )
    save_checkpoint(tmp_path / "checkpoint", model, scalers=scalers,
                    feature_schema={"stock_features": samples.arrays.stock_features},
                    sector_map_hash="s" * 64, feature_store_hash="f" * 64,
                    training_history={}, manifest={},
                    meta_config=SMALL_META.to_dict(),
                    meta_initialization=trainer.meta_init, model_config=config)

    query_rows = samples.frame.loc[masks["val"]].head(12).reset_index(drop=True)
    dataset = V2SequenceDataset(samples, query_rows, scalers=scalers)
    batch = {k: torch.stack([dataset[i][k] for i in range(len(dataset))])
             for k in dataset[0]}

    def predict(module):
        module.eval()
        with torch.no_grad():
            return module(batch["stock_sequence"], batch["ticker_id"],
                          context_sequence=batch["context_sequence"],
                          sector_id=batch["sector_id"],
                          regime_vector=batch["regime_vector"])["direction_logit"]

    base_before = predict(model)
    restored = load_checkpoint(tmp_path / "checkpoint", device="cpu")
    assert restored.meta_config is not None
    assert restored.meta_initialization is not None
    torch.testing.assert_close(predict(restored.model), base_before, rtol=0, atol=1e-6)

    # the stored meta initialisation is the PRE-adaptation parameter set

    for name, tensor in restored.meta_initialization.items():
        current = dict(restored.model.named_parameters())[name]
        torch.testing.assert_close(current.detach(), tensor, rtol=0, atol=1e-7)


def test_loading_a_checkpoint_without_meta_files_is_fine(samples, tmp_path):
    from agentic_forecaster.v2.checkpoint import load_checkpoint, save_checkpoint
    from agentic_forecaster.v2.dataset import GroupFeatureScaler, V2ScalerBundle

    model, config = _meta_model(samples)
    scalers = V2ScalerBundle(
        stock=GroupFeatureScaler(samples.arrays.stock_features).fit(
            np.zeros((4, len(samples.arrays.stock_features)))),
        context=GroupFeatureScaler([]).fit(np.zeros((4, 0))),
    )
    save_checkpoint(tmp_path / "plain", model, scalers=scalers,
                    feature_schema={}, sector_map_hash="s", feature_store_hash="f",
                    training_history={}, manifest={}, model_config=config)
    restored = load_checkpoint(tmp_path / "plain")
    assert restored.meta_config is None
    assert restored.meta_initialization is None


def test_samples_with_a_missing_window_are_excluded_from_episodes(arrays, targets):
    """A too-long window yields no samples, hence no episodes for that security."""
    tiny = build_sample_table(arrays, targets, sequence_length=500,
                              require_targets=["y_direction"])
    assert len(tiny) == 0
    assert build_meta_episodes(tiny, config=SMALL_META,
                               train_mask=np.ones(len(tiny), dtype=bool)) == []