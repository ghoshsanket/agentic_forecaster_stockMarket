"""Explainer Agent — per-prediction explanations.

For an individual prediction returns:
    ticker, date, p_up, direction, confidence,
    indicator values, top feature contributions, top attention dates, reason codes.

SHAP GradientExplainer is the primary reconstructed method; a permutation
fallback is explicitly reported and never mislabeled as SHAP.

LLM narration receives only structured evidence with grounding instructions.
The package remains usable with deterministic explanations and no LLM.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from agentic_forecaster.explainability import ShapExplainer

logger = logging.getLogger("agentic_forecaster.agents.explainer")

GROUNDING_INSTRUCTIONS = """You are a financial explanation engine. Rules:
- Do NOT invent news, events, or market developments.
- Do NOT invent fundamentals (earnings, revenue, etc.).
- Do NOT invent prices or price targets.
- Use ONLY the supplied indicator values, SHAP feature contributions,
  attention weights, and model outputs.
- Do NOT claim guaranteed returns or certainty.
- State that this is a model-based reconstruction, not investment advice.
- Be concise and factual."""


class ExplainerAgent:
    def __init__(self, config: dict):
        self.config = config
        self.exp_cfg = config.get("explainability", {})
        self.llm_cfg = config.get("reporting", {}).get("llm", {})

    def explain_prediction(
        self,
        fitted_model,
        x: np.ndarray,
        ticker: str,
        date: str,
        p_up: float,
        indicator_values: dict | None = None,
    ) -> dict:
        """Produce a per-prediction explanation bundle."""
        model = fitted_model.model
        device = next(model.parameters()).device

        background = fitted_model._background if hasattr(fitted_model, "_background") else None
        if background is None:
            background = x[np.newaxis, :]

        shap = ShapExplainer(model, background, fitted_model.feature_names)
        shap_result = shap.explain_single(x)

        attention_evidence = None
        if fitted_model.kind == "torch":
            model.eval()
            with torch.no_grad():
                _, weights = model(
                    torch.tensor(x[np.newaxis, :], dtype=torch.float32, device=device),
                    return_attention=True,
                )
            w = weights.cpu().numpy()[0]
            seq_len = x.shape[0]
            top_idx = np.argsort(w)[::-1][:5]
            attention_evidence = [
                {
                    "day_offset": int(seq_len - 1 - idx),
                    "attention_weight": float(w[idx]),
                }
                for idx in top_idx
            ]

        direction = "UP" if p_up >= 0.5 else "DOWN"
        confidence = p_up if direction == "UP" else 1 - p_up

        reason_codes = [
            f"direction={direction}",
            f"confidence={confidence:.4f}",
            f"shap_method={shap_result['method']}",
        ]

        explanation = {
            "ticker": ticker,
            "date": str(date),
            "p_up": float(p_up),
            "direction": direction,
            "confidence": float(confidence),
            "indicator_values": indicator_values or {},
            "top_features": shap_result["top_features"],
            "shap_method": shap_result["method"],
            "attention_evidence": attention_evidence,
            "reason_codes": reason_codes,
        }

        if self.llm_cfg.get("enabled", False):
            explanation["llm_narrative"] = self._llm_narrate(explanation)
        else:
            explanation["llm_narrative"] = self._deterministic_narrate(explanation)

        return explanation

    def _deterministic_narrate(self, explanation: dict) -> str:
        feats = explanation.get("top_features", [])
        if not feats:
            return "No dominant features identified."
        parts = [f"{f['feature']} (|SHAP|={f['mean_abs_shap']:.4f})" for f in feats[:3]]
        return (
            "The model's prediction is primarily driven by "
            + ", ".join(parts)
            + ". Attention evidence indicates which days in the lookback "
            "window most influenced the output."
        )

    def _llm_narrate(self, explanation: dict) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai not installed; using deterministic fallback")
            return self._deterministic_narrate(explanation)
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.llm_cfg.get("model", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": GROUNDING_INSTRUCTIONS},
                    {"role": "user", "content": str(explanation)},
                ],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("LLM narration failed (%s); using fallback", exc)
            return self._deterministic_narrate(explanation)
