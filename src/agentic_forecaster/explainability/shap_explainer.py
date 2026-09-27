"""Per-prediction feature attribution for the Attention-LSTM.

Primary method
    ``shap.GradientExplainer`` applied to a wrapper that exposes the model's
    single binary logit in the 2-D shape ``(batch, 1)`` that SHAP expects.
    The wrapper is a pure re-shaping: the underlying semantics remain a
    single binary logit, and ``p_up = sigmoid(logit)``.

Fallback method
    Integrated Gradients, computed against a background of real TRAINING
    sequences.  This is a genuine attribution method and is always labelled
    accurately in the output (``method`` field).  It never silently returns
    zero-valued one-sample permutation differences.

Both methods use a background drawn from the training split only.
"""

from __future__ import annotations

import contextlib
import logging
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

logger = logging.getLogger("agentic_forecaster.explainability")

DEFAULT_BACKGROUND_SIZE = 64


class BinaryLogitWrapper(nn.Module):
    """Expose a 1-D binary logit as ``(batch, 1)`` for SHAP compatibility.

    SHAP's gradient explainers expect a 2-D output.  This wrapper reshapes
    the model's single logit without changing its value, so the attribution
    semantics of the binary model are preserved exactly.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        return out


def sample_background(train_X: np.ndarray, size: int = DEFAULT_BACKGROUND_SIZE,
                      seed: int = 0) -> np.ndarray:
    """Draw a deterministic background sample from TRAINING sequences only."""
    if len(train_X) == 0:
        raise ValueError("Cannot build a SHAP background from an empty train set")
    size = min(size, len(train_X))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_X), size=size, replace=False)
    return train_X[idx].astype(np.float32)


class ShapExplainer:
    """Per-prediction attributions for the Attention-LSTM.

    Parameters
    ----------
    model : the fitted Attention-LSTM (single binary logit)
    background : real training sequences, shape (n_bg, T, F)
    feature_names : ordered feature names
    """

    def __init__(self, model, background: np.ndarray, feature_names: list[str]):
        self.model = model
        self.background = np.asarray(background, dtype=np.float32)
        self.feature_names = feature_names
        self._explainer = None
        self._method: str | None = None
        self.device = next(model.parameters()).device
        try:
            import shap  # noqa: F401

            self._has_shap = True
        except ImportError:
            self._has_shap = False
            logger.warning("shap not installed; using Integrated Gradients fallback")

    # ------------------------------------------------------------------ API

    def explain(self, X: np.ndarray) -> tuple[np.ndarray, str]:
        """Return ``(attributions, method)`` with shape ``(n, T, F)``."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[None, ...]
        if self._has_shap:
            try:
                return self._gradient_shap(X), "gradient_shap"
            except Exception as exc:
                logger.warning("GradientExplainer unavailable (%s); using Integrated Gradients", exc)
        return self._integrated_gradients(X), "integrated_gradients"

    def explain_single(self, x: np.ndarray) -> dict:
        """Explain one prediction; always returns a per-feature breakdown."""
        values, method = self.explain(x)
        sv = values[0]  # (T, F)
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:5]
        return {
            "method": method,
            "attribution_shape": list(sv.shape),
            "top_features": [
                {"feature": self.feature_names[i], "mean_abs_shap": float(mean_abs[i])}
                for i in order
            ],
            "attributions": sv,
        }

    def top_features(self, attributions: np.ndarray, k: int = 5) -> list[dict]:
        mean_abs = np.abs(attributions).mean(axis=(0, 1))
        order = np.argsort(mean_abs)[::-1][:k]
        return [
            {"feature": self.feature_names[i], "mean_abs_shap": float(mean_abs[i])}
            for i in order
        ]

    # ------------------------------------------------------------- methods

    @contextmanager
    def _deterministic_grad(self):
        """Yield with the model in a deterministic, gradient-capable state.

        On CUDA the fused cuDNN RNN refuses to run a backward pass while the
        module is in eval mode.  Both SHAP's GradientExplainer and Integrated
        Gradients need input gradients, so the model is temporarily switched
        to train mode with dropout disabled, which keeps the network
        deterministic -- as an attribution must be.
        """
        was_training = self.model.training
        dropout_states = [
            (m, m.p) for m in self.model.modules() if isinstance(m, nn.Dropout)
        ]
        self.model.train()
        for module, _ in dropout_states:
            module.p = 0.0
        try:
            yield
        finally:
            for module, p in dropout_states:
                module.p = p
            self.model.train(was_training)

    def _gradient_shap(self, X: np.ndarray) -> np.ndarray:
        import shap

        # SHAP's GradientExplainer puts the model in eval mode internally, and
        # the fused cuDNN RNN cannot run a backward pass from eval mode.  On
        # CUDA we therefore take the unfused RNN path for attribution only.
        stack = (
            torch.backends.cudnn.flags(enabled=False)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with stack, self._deterministic_grad():
            if self._explainer is None:
                wrapper = BinaryLogitWrapper(self.model).to(self.device).eval()
                bg = torch.tensor(
                    self.background, dtype=torch.float32, device=self.device
                )
                self._explainer = shap.GradientExplainer(wrapper, bg)
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            values = self._explainer.shap_values(X_t)
        arr = np.asarray(values)
        if arr.ndim == 4 or (arr.ndim == 3 and arr.shape[-1] == 1):
            arr = arr[..., 0]          # (n, T, F, 1) -> (n, T, F)
        return arr

    def _integrated_gradients(self, X: np.ndarray, n_steps: int = 32) -> np.ndarray:
        """Integrated Gradients from a background baseline to the input.

        ``IG_i = (x_i - baseline_i) * mean_k d f(base + k*(x-base))/dx_i``
        """
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        n = X_t.shape[0]
        bg_idx = np.linspace(0, len(self.background) - 1, n).astype(int)
        base = torch.tensor(
            self.background[bg_idx], dtype=torch.float32, device=self.device
        )
        with self._deterministic_grad():
            total = torch.zeros_like(X_t)
            alphas = torch.linspace(1.0 / n_steps, 1.0, n_steps, device=self.device)
            for a in alphas:
                point = (base + a * (X_t - base)).detach().requires_grad_(True)
                logit = self.model(point)
                grad = torch.autograd.grad(logit.sum(), point, retain_graph=False)[0]
                total = total + grad
            avg_grad = total / n_steps
        return ((X_t - base) * avg_grad).detach().cpu().numpy()
