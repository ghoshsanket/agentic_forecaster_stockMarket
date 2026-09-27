import torch

from agentic_forecaster.models.attention import TemporalAttention


class TestTemporalAttention:
    def test_forward_output_shape(self):
        attn = TemporalAttention(hidden_dim=64, attention_dim=64)
        x = torch.randn(2, 30, 64)
        context, weights = attn(x)
        assert context.shape == (2, 64)
        assert weights.shape == (2, 30)

    def test_attention_weights_sum_to_one(self):
        attn = TemporalAttention(hidden_dim=64, attention_dim=64)
        x = torch.randn(2, 30, 64)
        _, weights = attn(x)
        sums = weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_attention_weights_non_negative(self):
        attn = TemporalAttention(hidden_dim=64, attention_dim=64)
        x = torch.randn(2, 30, 64)
        _, weights = attn(x)
        assert (weights >= 0).all()

    def test_attention_weights_shape(self):
        attn = TemporalAttention(hidden_dim=32, attention_dim=16)
        x = torch.randn(4, 10, 32)
        _, weights = attn(x)
        assert weights.shape == (4, 10)

    def test_batch_size_one(self):
        attn = TemporalAttention(hidden_dim=64, attention_dim=64)
        x = torch.randn(1, 30, 64)
        context, weights = attn(x)
        assert context.shape == (1, 64)
        assert weights.shape == (1, 30)

    def test_single_timestep(self):
        attn = TemporalAttention(hidden_dim=64, attention_dim=64)
        x = torch.randn(2, 1, 64)
        context, weights = attn(x)
        assert context.shape == (2, 64)
        assert weights.shape == (2, 1)
        assert torch.allclose(weights, torch.ones(2, 1), atol=1e-5)

    def test_long_sequence(self):
        attn = TemporalAttention(hidden_dim=64, attention_dim=64)
        x = torch.randn(2, 100, 64)
        context, weights = attn(x)
        assert context.shape == (2, 64)
        assert weights.shape == (2, 100)

    def test_zero_input(self):
        attn = TemporalAttention(hidden_dim=64, attention_dim=64)
        x = torch.zeros(2, 30, 64)
        context, weights = attn(x)
        assert context.shape == (2, 64)
        assert weights.shape == (2, 30)

    def test_identical_inputs(self):
        attn = TemporalAttention(hidden_dim=64, attention_dim=64)
        x = torch.ones(2, 30, 64)
        _, weights = attn(x)
        assert torch.allclose(weights, torch.full((2, 30), 1.0 / 30), atol=1e-5)

    def test_deterministic_output(self):
        attn = TemporalAttention(hidden_dim=64, attention_dim=64)
        x = torch.randn(2, 30, 64)
        ctx1, w1 = attn(x)
        ctx2, w2 = attn(x)
        assert torch.allclose(ctx1, ctx2)
        assert torch.allclose(w1, w2)
