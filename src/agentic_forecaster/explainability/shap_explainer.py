"""SHAP-based feature attribution for the Attention-LSTM.

Uses ``shap.DeepExplainer`` when available; falls back to a permutation
approximation so the pipeline never hard-fails when SHAP is not installed.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger("agentic_forecaster.explainability")


class ShapExplainer:
    def __init__(self, model, background: np.ndarray, feature_names: list[str]):
        self.model = model
        self.background = background
        self.feature_names = feature_names
        self._explainer = None
        try:
            import shap  # noqa: F401

            self._has_shap = True
        except ImportError:
            self._has_shap = False
            logger.warning("shap not installed; using permutation fallback")

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Return SHAP values with shape ``(n_samples, seq_len, n_features)``."""
        if self._has_shap:
            try:
                return self._explain_shap(X)
            except (ValueError, RuntimeError, ImportError) as exc:
                logger.warning(
                    "SHAP DeepExplainer failed (%s); using permutation fallback", exc
                )
        return self._explain_permutation(X)

    def _explain_shap(self, X: np.ndarray) -> np.ndarray:
        import shap

        device = next(self.model.parameters()).device
        if self._explainer is None:
            bg = torch.tensor(self.background, dtype=torch.float32, device=device)
            self._explainer = shap.DeepExplainer(self.model, bg)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        values = self._explainer.shap_values(X_t)
        arr = np.asarray(values)
        if arr.ndim == 3 and arr.shape[0] == 2:
            arr = arr[1]
        return arr

    def _explain_permutation(self, X: np.ndarray) -> np.ndarray:
        """Permutation baseline: shuffle each feature and measure logit change."""
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            base_logits = self.model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
        n, _t, f = X.shape
        shap_values = np.zeros_like(X, dtype=np.float64)
        rng = np.random.default_rng(0)
        for j in range(f):
            X_perm = X.copy()
            perm = rng.permutation(n)
            X_perm[:, :, j] = X[perm][:, :, j]
            with torch.no_grad():
                perm_logits = self.model(torch.tensor(X_perm, dtype=torch.float32, device=device)).cpu().numpy()
            shap_values[:, :, j] = (base_logits[:, 1] - perm_logits[:, 1])[:, None]
        return shap_values

    def top_features(self, shap_values: np.ndarray, k: int = 5) -> list[dict]:
        """Aggregate SHAP over the time axis and return the top-k features."""
        mean_abs = np.abs(shap_values).mean(axis=(0, 1))
        order = np.argsort(mean_abs)[::-1][:k]
        return [
            {"feature": self.feature_names[i], "mean_abs_shap": float(mean_abs[i])}
            for i in order
        ]
