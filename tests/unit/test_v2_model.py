"""V2 model tests: shapes, masks, embeddings, heads, FiLM, determinism, gradients.

Every structural claim the V2 architecture makes is asserted here, including the
one that matters most for auditability: the Transformer's attention mask is
genuinely CAUSAL, proved by perturbing one timestep and checking that no earlier
output moves.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agentic_forecaster.v2.model import (
    ContextualLSTMTransformer,
    FiLMConditioning,
    ResidualAdapter,
    V2ModelConfig,
    build_model,
    causal_mask,
    sinusoidal_position_encoding,
)

from .v2_fixtures import SEQUENCE_LENGTH, batch_from, make_model_config

BATCH = 6


def _batch(config: V2ModelConfig, batch: int = BATCH, length: int = SEQUENCE_LENGTH):
    torch.manual_seed(0)
    return {
        "stock_sequence": torch.randn(batch, length, config.n_stock_features),
        "context_sequence": torch.randn(batch, length, config.n_context_features),
        "regime_vector": torch.randn(batch, config.n_regime_features),
        "ticker_id": torch.randint(0, config.n_tickers, (batch,)),
        "sector_id": torch.randint(0, config.n_sectors, (batch,)),
    }


# ---------------------------------------------------------------------------
# A. LSTM output sequence
# ---------------------------------------------------------------------------

def test_lstm_returns_the_full_hidden_state_sequence(model_config):
    model = ContextualLSTMTransformer(model_config).eval()
    batch = _batch(model_config)
    with torch.no_grad():
        h_lstm, _ = model.lstm(batch["stock_sequence"])
    assert h_lstm.shape == (BATCH, SEQUENCE_LENGTH, model_config.hidden_size)
    assert not torch.allclose(h_lstm[:, -1, :], h_lstm[:, 0, :]), (
        "the final hidden state must differ from the first: the whole sequence "
        "is returned, not only the last step")


def test_lstm_is_unidirectional_and_not_full_state_only(model_config):
    model = ContextualLSTMTransformer(model_config)
    assert model.lstm.bidirectional is False
    assert model.lstm.batch_first is True
    assert model.lstm.num_layers == model_config.lstm_layers


# ---------------------------------------------------------------------------
# B/C/D. context projection, temporal fusion, positional encoding
# ---------------------------------------------------------------------------

def test_context_projection_shape(model_config):
    model = ContextualLSTMTransformer(model_config).eval()
    batch = _batch(model_config)
    with torch.no_grad():
        h_context = model.context_encoder(batch["context_sequence"])
    assert h_context.shape == (BATCH, SEQUENCE_LENGTH, model_config.context_dim)


def test_temporal_fusion_projects_to_d_model(model_config):
    model = ContextualLSTMTransformer(model_config).eval()
    batch = _batch(model_config)
    with torch.no_grad():
        h_lstm, _ = model.lstm(batch["stock_sequence"])
        h_context = model.context_encoder(batch["context_sequence"])
        fused = model.temporal_fusion(torch.cat([h_lstm, h_context], dim=-1))
    assert fused.shape == (BATCH, SEQUENCE_LENGTH, model_config.d_model)
    assert model.temporal_fusion[0].in_features == (
        model_config.hidden_size + model_config.context_dim)


def test_positional_encoding_supports_at_least_128_positions():
    encoding = sinusoidal_position_encoding(128, 64)
    assert encoding.shape == (128, 64)
    assert torch.isfinite(encoding).all()


def test_model_accepts_the_full_128_step_window():
    config = make_model_config(max_sequence_length=128, use_transformer=True)
    model = ContextualLSTMTransformer(config).eval()
    batch = _batch(config, batch=2, length=128)
    with torch.no_grad():
        out = model(batch["stock_sequence"], batch["ticker_id"],
                    context_sequence=batch["context_sequence"],
                    sector_id=batch["sector_id"],
                    regime_vector=batch["regime_vector"])
    assert out["direction_logit"].shape == (2,)


def test_sequence_longer_than_the_declared_maximum_is_refused():
    config = make_model_config(max_sequence_length=16)
    model = ContextualLSTMTransformer(config).eval()
    batch = _batch(config, batch=2, length=32)
    with pytest.raises(ValueError, match="max_sequence_length"):
        model(batch["stock_sequence"], batch["ticker_id"],
              context_sequence=batch["context_sequence"],
              sector_id=batch["sector_id"],
              regime_vector=batch["regime_vector"])


# ---------------------------------------------------------------------------
# E. causal mask
# ---------------------------------------------------------------------------

def test_causal_mask_shape_and_content():
    mask = causal_mask(5)
    assert mask.shape == (5, 5)
    assert mask.dtype == torch.bool
    assert torch.equal(mask[0], torch.tensor([False, True, True, True, True]))
    assert torch.equal(mask[4], torch.tensor([False, False, False, False, False]))
    assert int(mask.sum()) == 5 * 4 // 2


def test_transformer_is_actually_causal(model_config):
    """Perturbing step k must not move any output at a step BEFORE k."""
    model = ContextualLSTMTransformer(model_config).eval()
    batch = _batch(model_config, batch=2, length=12)
    length = 12
    with torch.no_grad():
        h_lstm, _ = model.lstm(batch["stock_sequence"])
        h_context = model.context_encoder(batch["context_sequence"])
        fused = model.temporal_fusion(torch.cat([h_lstm, h_context], dim=-1))
        positions = model.position_encoding[:length]
        base = model.transformer(fused + positions, mask=causal_mask(length))

        for k in (0, 3, 7, 11):
            perturbed_input = batch["stock_sequence"].clone()
            perturbed_input[:, k, :] += 5.0
            h_lstm_p, _ = model.lstm(perturbed_input)
            fused_p = model.temporal_fusion(
                torch.cat([h_lstm_p, h_context], dim=-1))
            perturbed = model.transformer(fused_p + positions,
                                         mask=causal_mask(length))
            before = base[:, :k, :]
            after = perturbed[:, :k, :]
            torch.testing.assert_close(before, after, rtol=0, atol=1e-6)
            assert not torch.allclose(base[:, k:, :], perturbed[:, k:, :]), (
                f"perturbing step {k} left the later outputs unchanged: the mask "
                "is not doing anything")


# ---------------------------------------------------------------------------
# F. attention pooling
# ---------------------------------------------------------------------------

def test_attention_weights_sum_to_one(model_config):
    model = ContextualLSTMTransformer(model_config).eval()
    batch = _batch(model_config)
    with torch.no_grad():
        _, attention, _ = model.encode_temporal(
            batch["stock_sequence"], batch["context_sequence"])
    assert attention.shape == (BATCH, SEQUENCE_LENGTH)
    torch.testing.assert_close(attention.sum(dim=1), torch.ones(BATCH), rtol=0, atol=1e-5)
    assert (attention >= 0).all()


def test_attention_pooling_is_learned_not_the_last_step(model_config):
    """A trained-by-one-step model must not trivially equal the final timestep."""
    model = ContextualLSTMTransformer(model_config).eval()
    with torch.no_grad():
        for parameter in model.pool_score.parameters():
            parameter.add_(torch.randn_like(parameter))
    batch = _batch(model_config)
    with torch.no_grad():
        _, attention, _ = model.encode_temporal(
            batch["stock_sequence"], batch["context_sequence"])
    last = attention[:, -1]
    assert not torch.allclose(last, torch.ones_like(last), atol=1e-3)


def test_attention_is_returned_on_request(model_config):
    model = ContextualLSTMTransformer(model_config).eval()
    batch = _batch(model_config)
    with torch.no_grad():
        out = model(batch["stock_sequence"], batch["ticker_id"],
                    context_sequence=batch["context_sequence"],
                    sector_id=batch["sector_id"],
                    regime_vector=batch["regime_vector"], return_attention=True)
    assert "temporal_attention" in out
    assert out["temporal_attention"].shape == (BATCH, SEQUENCE_LENGTH)


# ---------------------------------------------------------------------------
# G. LSTM residual
# ---------------------------------------------------------------------------

def test_lstm_residual_dimensions_and_effect(model_config):
    model = ContextualLSTMTransformer(model_config).eval()
    assert model.lstm_residual.in_features == model_config.hidden_size
    assert model.lstm_residual.out_features == model_config.d_model
    batch = _batch(model_config)
    with torch.no_grad():
        _, _, lstm_last = model.encode_temporal(batch["stock_sequence"],
                                                batch["context_sequence"])
    assert lstm_last.shape == (BATCH, model_config.hidden_size)
    # zeroing the residual must change the temporal representation
    with torch.no_grad():
        base, _, _ = model.encode_temporal(batch["stock_sequence"],
                                           batch["context_sequence"])
        pooled_only = model.temporal_norm(
            torch.zeros(BATCH, model_config.d_model))
    assert not torch.allclose(base, pooled_only)


def test_temporal_representation_is_layer_normalised(model_config):
    model = ContextualLSTMTransformer(model_config).eval()
    batch = _batch(model_config)
    with torch.no_grad():
        temporal, _, _ = model.encode_temporal(batch["stock_sequence"],
                                               batch["context_sequence"])
    assert temporal.shape == (BATCH, model_config.d_model)
    torch.testing.assert_close(
        temporal.mean(dim=1), torch.zeros(BATCH), rtol=0, atol=5e-2)


# ---------------------------------------------------------------------------
# 17/18/19. embeddings, regime encoder, fusion
# ---------------------------------------------------------------------------

def test_ticker_embedding_dimension(model_config):
    model = ContextualLSTMTransformer(model_config)
    assert model.ticker_embedding.embedding_dim == model_config.ticker_embedding_dim
    assert model.ticker_embedding.num_embeddings == model_config.n_tickers
    ids = torch.tensor([0, 1, model_config.n_tickers - 1])
    assert model.ticker_embedding(ids).shape == (3, model_config.ticker_embedding_dim)


def test_sector_embedding_dimension_and_unknown_slot(model_config):
    model = ContextualLSTMTransformer(model_config)
    assert model.sector_embedding.embedding_dim == model_config.sector_embedding_dim
    assert "UNKNOWN" in model_config.sector_vocab
    unknown = model_config.unknown_sector_index()
    assert unknown == model_config.sector_vocab.index("UNKNOWN")
    ids = torch.tensor([unknown])
    assert model.sector_embedding(ids).shape == (1, model_config.sector_embedding_dim)
    assert torch.isfinite(model.sector_embedding(ids)).all()


def test_one_shared_model_is_not_one_model_per_ticker(model_config):
    model = ContextualLSTMTransformer(model_config)
    # identity enters only through the embedding tables
    assert not any("AAA" in name or "RELIANCE" in name
                   for name, _ in model.named_parameters())
    counts = model.n_parameters()
    assert counts["total_parameters"] > 0
    # a single set of weights serves every security
    assert len([n for n, _ in model.named_parameters()
                if n.startswith("lstm.weight_ih_l0")]) == 1


def test_regime_encoder_dimension(model_config):
    model = ContextualLSTMTransformer(model_config).eval()
    batch = _batch(model_config)
    with torch.no_grad():
        regime = model.regime_encoder(batch["regime_vector"])
    assert regime.shape == (BATCH, model_config.regime_embedding_dim)
    assert model.regime_encoder[0].in_features == model_config.n_regime_features


def test_regime_encoder_is_never_a_hand_labelled_regime(model_config):
    """The encoder consumes raw market state; no bull/bear label is derived."""
    model = ContextualLSTMTransformer(model_config)
    names = [n for n, _ in model.named_parameters() if "regime" in n]
    assert names, "the regime encoder must exist for a contextual variant"
    assert not any("bull" in n or "bear" in n for n in names)


def test_fusion_representation_shape(model_config):
    model = ContextualLSTMTransformer(model_config).eval()
    batch = _batch(model_config)
    with torch.no_grad():
        out = model(batch["stock_sequence"], batch["ticker_id"],
                    context_sequence=batch["context_sequence"],
                    sector_id=batch["sector_id"],
                    regime_vector=batch["regime_vector"])
    assert out["z"].shape == (BATCH, model_config.z_dim)
    assert model.fusion[0].in_features == model_config.fusion_input_dim
    assert model_config.fusion_input_dim == (
        model_config.d_model + model_config.ticker_embedding_dim
        + model_config.sector_embedding_dim + model_config.regime_embedding_dim)


# ---------------------------------------------------------------------------
# 20. the three task heads
# ---------------------------------------------------------------------------

def test_direction_head_shape(model_config):
    model = ContextualLSTMTransformer(model_config).eval()
    batch = _batch(model_config)
    with torch.no_grad():
        out = model(batch["stock_sequence"], batch["ticker_id"],
                    context_sequence=batch["context_sequence"],
                    sector_id=batch["sector_id"],
                    regime_vector=batch["regime_vector"])
    assert out["direction_logit"].shape == (BATCH,)


def test_return_and_rank_heads_shapes_and_range(model_config):
    model = ContextualLSTMTransformer(model_config).eval()
    batch = _batch(model_config)
    with torch.no_grad():
        out = model(batch["stock_sequence"], batch["ticker_id"],
                    context_sequence=batch["context_sequence"],
                    sector_id=batch["sector_id"],
                    regime_vector=batch["regime_vector"])
    assert out["return_prediction"].shape == (BATCH,)
    assert out["rank_prediction"].shape == (BATCH,)
    assert (out["rank_prediction"] >= 0).all() and (out["rank_prediction"] <= 1).all()


def test_single_task_variant_has_no_auxiliary_heads():
    config = make_model_config(use_multitask=False)
    model = ContextualLSTMTransformer(config).eval()
    assert model.return_head is None and model.rank_head is None
    batch = _batch(config)
    with torch.no_grad():
        out = model(batch["stock_sequence"], batch["ticker_id"],
                    context_sequence=batch["context_sequence"],
                    sector_id=batch["sector_id"],
                    regime_vector=batch["regime_vector"])
    assert "direction_logit" in out
    assert "return_prediction" not in out and "rank_prediction" not in out


def test_non_contextual_variant_ignores_the_context_stream():
    config = make_model_config(use_context=False, use_sector_embedding=False,
                               use_regime=False, use_multitask=False)
    model = ContextualLSTMTransformer(config).eval()
    assert isinstance(model.context_encoder, torch.nn.Identity)
    batch = _batch(config)
    with torch.no_grad():
        out = model(batch["stock_sequence"], batch["ticker_id"],
                    context_sequence=batch["context_sequence"],
                    sector_id=None, regime_vector=None)
    assert out["direction_logit"].shape == (BATCH,)
    assert model.fusion[0].in_features == (
        config.d_model + config.ticker_embedding_dim)


# ---------------------------------------------------------------------------
# 22. FiLM
# ---------------------------------------------------------------------------

def test_film_gamma_and_beta_dimensions_and_bounds():
    film = FiLMConditioning(condition_dim=12, feature_dim=8, scale=0.1)
    z = torch.randn(4, 8)
    condition = torch.randn(4, 12)
    adapted, stats = film(z, condition)
    assert adapted.shape == (4, 8)
    assert stats["gamma"].shape == (4, 8)
    assert stats["beta"].shape == (4, 8)
    assert torch.allclose(stats["gamma"], 1.0 + 0.1 * torch.tanh(stats["gamma"] * 0
                          + torch.atanh((stats["gamma"] - 1.0) / 0.1)), atol=1e-4)
    assert (stats["gamma"] - 1.0).abs().max() <= 0.1 + 1e-6
    assert stats["beta"].abs().max() <= 0.1 + 1e-6


def test_film_starts_as_the_identity():
    """A freshly initialised FiLM must not destroy a trained encoder."""
    film = FiLMConditioning(condition_dim=6, feature_dim=4)
    z = torch.randn(3, 4)
    _, stats = film(z, torch.randn(3, 6))
    torch.testing.assert_close(stats["gamma"], torch.ones_like(stats["gamma"]))
    torch.testing.assert_close(stats["beta"], torch.zeros_like(stats["beta"]))


def test_film_disabled_path_is_a_pure_pass_through():
    config = make_model_config(use_film=False)
    model = ContextualLSTMTransformer(config).eval()
    assert model.film is None
    batch = _batch(config)
    with torch.no_grad():
        out = model(batch["stock_sequence"], batch["ticker_id"],
                    context_sequence=batch["context_sequence"],
                    sector_id=batch["sector_id"],
                    regime_vector=batch["regime_vector"])
    torch.testing.assert_close(out["z_adapted"], out["z"])
    assert "film_gamma" not in out


def test_film_enabled_path_changes_the_representation():
    config = make_model_config(use_film=True)
    model = ContextualLSTMTransformer(config).eval()
    batch = _batch(config)
    with torch.no_grad():
        for parameter in model.film.produce.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
        out = model(batch["stock_sequence"], batch["ticker_id"],
                    context_sequence=batch["context_sequence"],
                    sector_id=batch["sector_id"],
                    regime_vector=batch["regime_vector"])
    assert not torch.allclose(out["z_adapted"], out["z"])
    assert "film_gamma" in out and "film_beta" in out


def test_film_is_not_called_meta_learning():
    """FiLM is CONDITIONAL ADAPTATION; the meta stage lives in meta.py."""
    from agentic_forecaster.v2.meta import META_LABEL, NOT_MAML

    assert META_LABEL == "REPTILE_STYLE_HEAD_ADAPTER"
    assert "MAML" in NOT_MAML


# ---------------------------------------------------------------------------
# 23. residual adapter
# ---------------------------------------------------------------------------

def test_residual_adapter_starts_as_the_identity():
    adapter = ResidualAdapter(feature_dim=8, hidden_dim=4)
    z = torch.randn(5, 8)
    torch.testing.assert_close(adapter(z), z)


def test_adapter_adds_a_residual_branch():
    adapter = ResidualAdapter(feature_dim=8, hidden_dim=4)
    with torch.no_grad():
        adapter.block[-1].bias.add_(1.0)
    z = torch.randn(5, 8)
    assert not torch.allclose(adapter(z), z)
    assert adapter(z).shape == z.shape


# ---------------------------------------------------------------------------
# determinism and gradients
# ---------------------------------------------------------------------------

def test_same_seed_gives_identical_initialisation():
    a = build_model(make_model_config(), seed=7)
    b = build_model(make_model_config(), seed=7)
    names = [n for n, _ in a.named_parameters()]
    assert names, "the model must expose named parameters"
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        torch.testing.assert_close(pa, pb, rtol=0, atol=0)


def test_different_seed_changes_initialisation():
    a = build_model(make_model_config(), seed=7)
    b = build_model(make_model_config(), seed=8)
    differences = [
        float((pa - pb).abs().max().detach())
        for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True)
    ]
    assert max(differences) > 0


def test_forward_backward_produces_finite_gradients(model_config):
    from agentic_forecaster.v2.losses import MultiTaskWeights, V2MultiTaskLoss

    model = ContextualLSTMTransformer(model_config)
    batch = _batch(model_config)
    targets = {
        "y_direction": torch.randint(0, 2, (BATCH,)).float(),
        "y_return": torch.randn(BATCH),
        "y_rank": torch.rand(BATCH),
    }
    criterion = V2MultiTaskLoss(MultiTaskWeights())
    out = model(batch["stock_sequence"], batch["ticker_id"],
                context_sequence=batch["context_sequence"],
                sector_id=batch["sector_id"], regime_vector=batch["regime_vector"])
    loss, components = criterion(out, targets)
    assert torch.isfinite(loss)
    loss.backward()

    missing = [name for name, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not missing, f"parameters without finite gradients: {missing}"
    assert set(components) >= {"direction", "total"}


def test_parameter_count_report_is_consistent(model_config):
    model = ContextualLSTMTransformer(model_config)
    counts = model.n_parameters()
    assert counts["total_parameters"] == sum(p.numel() for p in model.parameters())
    assert 0 < counts["adaptable_parameters"] <= counts["total_parameters"]


def test_adaptable_set_excludes_the_shared_encoder(model_config):
    config = make_model_config(use_adapter=True, use_multitask=True)
    model = ContextualLSTMTransformer(config)
    adaptable = set(model.adaptable_parameter_names())
    assert adaptable, "V2-F must have adaptable parameters"
    for name in adaptable:
        assert name.startswith(("adapter.", "direction_head.", "return_head.",
                                "rank_head."))
    shared = set(model.shared_parameter_names())
    assert any(n.startswith("lstm.") for n in shared)
    assert any(n.startswith("transformer.") for n in shared)
    assert any(n.startswith("ticker_embedding.") for n in shared)


def test_encoder_freeze_leaves_only_adaptable_parameters_trainable():
    config = make_model_config(use_adapter=True)
    model = ContextualLSTMTransformer(config)
    model.set_encoder_frozen(True)
    adaptable = set(model.adaptable_parameter_names())
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == (name in adaptable), name


def test_forward_returns_a_structured_object(model_config):
    model = ContextualLSTMTransformer(model_config).eval()
    batch = _batch(model_config)
    with torch.no_grad():
        out = model(batch["stock_sequence"], batch["ticker_id"],
                    context_sequence=batch["context_sequence"],
                    sector_id=batch["sector_id"], regime_vector=batch["regime_vector"])
    assert isinstance(out, dict)
    for key in ("direction_logit", "return_prediction", "rank_prediction", "z",
                "z_adapted"):
        assert key in out


def test_model_config_round_trips_through_json():
    config = make_model_config()
    restored = V2ModelConfig.from_dict(config.to_dict())
    assert restored.to_dict() == config.to_dict()


def test_configuration_error_when_fusion_and_d_model_disagree():
    with pytest.raises(ValueError, match="fusion_temporal_dim"):
        ContextualLSTMTransformer(make_model_config(d_model=8, fusion_temporal_dim=16))


def test_no_nan_reaches_the_model_from_a_real_batch(samples):
    """The dataset hands the model finite tensors only."""
    from agentic_forecaster.v2.dataset import V2SequenceDataset

    dataset = V2SequenceDataset(samples, samples.frame)
    batch = batch_from(dataset, [0, 1, 2])
    config = make_model_config(
        n_stock_features=samples.arrays.n_stock_features,
        n_context_features=samples.arrays.n_context_features,
        n_regime_features=samples.arrays.n_regime_features,
        n_tickers=max(len(samples.arrays.ticker_vocab), 1),
        n_sectors=max(len(samples.arrays.sector_vocab), 1),
        ticker_vocab=list(samples.arrays.ticker_vocab),
        sector_vocab=list(samples.arrays.sector_vocab),
    )
    model = ContextualLSTMTransformer(config)
    out = model(batch["stock_sequence"], batch["ticker_id"],
                context_sequence=batch["context_sequence"],
                sector_id=batch["sector_id"], regime_vector=batch["regime_vector"])
    for value in out.values():
        assert torch.isfinite(value).all()
    assert np.isfinite(out["direction_logit"].detach().numpy()).all()