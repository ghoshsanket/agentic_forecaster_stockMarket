"""SHAP attribution must be finite and non-zero for a non-constant model (item 9)."""

from __future__ import annotations

import numpy as np
import torch

from agentic_forecaster.explainability.shap_explainer import (
    BinaryLogitWrapper,
    ShapExplainer,
    sample_background,
)
from agentic_forecaster.models import AttentionLSTM


def _model(seed: int = 0) -> AttentionLSTM:
    torch.manual_seed(seed)
    return AttentionLSTM(input_size=4, hidden_size=8, num_layers=1, dropout=0.0)


def test_wrapper_returns_2d_output():
    model = _model()
    wrapper = BinaryLogitWrapper(model)
    out = wrapper(torch.randn(3, 30, 4))
    assert out.shape == (3, 1)


def test_wrapper_preserves_logit_value():
    model = _model()
    model.eval()
    x = torch.randn(3, 30, 4)
    with torch.no_grad():
        raw = model(x)
        wrapped = BinaryLogitWrapper(model)(x)
    assert torch.allclose(raw, wrapped.squeeze(-1), atol=1e-6)


def test_single_prediction_attribution_is_nonzero_and_finite():
    model = _model()
    rng = np.random.default_rng(0)
    background = rng.normal(size=(64, 30, 4)).astype(np.float32)
    x = rng.normal(size=(30, 4)).astype(np.float32)
    # Nudge the classifier away from zero so the model is non-constant.
    with torch.no_grad():
        model.classifier.bias.fill_(0.7)

    explainer = ShapExplainer(model, background, ["a", "b", "c", "d"])
    values, method = explainer.explain(x[None, ...])
    assert values.shape == (1, 30, 4)
    assert np.isfinite(values).all()
    assert (np.abs(values) > 1e-12).sum() > 0, "attribution must not be all zeros"
    assert method in {"gradient_shap", "integrated_gradients"}


def test_integrated_gradients_fallback_is_real():
    model = _model()
    with torch.no_grad():
        model.classifier.bias.fill_(0.5)
    rng = np.random.default_rng(1)
    background = rng.normal(size=(32, 30, 4)).astype(np.float32)
    x = rng.normal(size=(1, 30, 4)).astype(np.float32)

    explainer = ShapExplainer(model, background, ["a", "b", "c", "d"])
    values = explainer._integrated_gradients(x)
    assert values.shape == (1, 30, 4)
    assert np.isfinite(values).all()
    assert (np.abs(values) > 1e-12).sum() > 0


def test_background_sampling_uses_training_data_only():
    train = np.random.default_rng(2).normal(size=(100, 30, 4)).astype(np.float32)
    bg = sample_background(train, size=32)
    assert bg.shape == (32, 30, 4)
    for row in bg:
        assert any(np.allclose(row, t) for t in train)


def test_explain_single_returns_method_label():
    model = _model()
    with torch.no_grad():
        model.classifier.bias.fill_(0.4)
    rng = np.random.default_rng(3)
    background = rng.normal(size=(16, 30, 4)).astype(np.float32)
    x = rng.normal(size=(30, 4)).astype(np.float32)
    out = ShapExplainer(model, background, ["a", "b", "c", "d"]).explain_single(x)
    assert out["attribution_shape"] == [30, 4]
    assert out["method"] in {"gradient_shap", "integrated_gradients"}
    assert len(out["top_features"]) >= 1


def test_attribution_works_on_both_devices():
    """SHAP must produce finite non-zero attributions on CPU and CUDA alike.

    The fused cuDNN RNN cannot run a backward pass from eval mode, so the
    explainer takes the unfused path for attribution on CUDA.
    """

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rng = np.random.default_rng(7)
    background = rng.normal(size=(32, 30, 4)).astype(np.float32)
    x = rng.normal(size=(1, 30, 4)).astype(np.float32)
    for device in devices:
        model = _model().to(device)
        with torch.no_grad():
            model.classifier.bias.fill_(0.6)
        values, method = ShapExplainer(
            model, background, ["a", "b", "c", "d"]
        ).explain(x)
        assert values.shape == (1, 30, 4)
        assert np.isfinite(values).all()
        assert (np.abs(values) > 1e-12).sum() > 0, f"{device} produced zero attribution"
        assert method in {"gradient_shap", "integrated_gradients"}
