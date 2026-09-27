"""Explainer Agent — per-prediction explanations.

For an individual prediction returns:
    ticker, date, p_up_raw, p_up_calibrated, direction, confidence,
    indicator values, top feature contributions, top attention dates,
    reason codes, and a narrative.

Attribution uses ``shap.GradientExplainer`` on a binary-logit wrapper, with
Integrated Gradients as an explicitly-labelled fallback.  The background is
drawn from TRAINING sequences only.

LLM narration (optional) receives only structured evidence plus grounding
instructions that forbid inventing news, fundamentals or prices.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from agentic_forecaster.explainability.shap_explainer import (
    DEFAULT_BACKGROUND_SIZE,
    ShapExplainer,
    sample_background,
)

logger = logging.getLogger("agentic_forecaster.agents.explainer")

GROUNDING_INSTRUCTIONS = """You are a financial explanation engine. Rules:
- Do NOT invent news, events, or market developments.
- Do NOT invent fundamentals (earnings, revenue, guidance, etc.).
- Do NOT invent prices or price targets.
- Use ONLY the supplied indicator values, feature attributions, attention
  weights and model outputs.
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
        p_up_raw: float | None = None,
    ) -> dict:
        """Produce a per-prediction explanation bundle."""
        device = next(fitted_model.model.parameters()).device
        background = getattr(fitted_model, "_background", None)
        if background is None or len(background) == 0:
            background = x[np.newaxis, :]
        else:
            background = np.asarray(background, dtype=np.float32)

        shap = ShapExplainer(fitted_model.model, background, fitted_model.feature_names)
        result = shap.explain_single(x)

        attention_evidence = None
        if fitted_model.kind == "torch":
            fitted_model.model.eval()
            with torch.no_grad():
                _, weights = fitted_model.model(
                    torch.tensor(x[np.newaxis, :], dtype=torch.float32, device=device),
                    return_attention=True,
                )
            w = weights.cpu().numpy()[0]
            seq_len = x.shape[0]
            top_idx = np.argsort(w)[::-1][:5]
            attention_evidence = [
                {"day_offset": int(seq_len - 1 - int(i)),
                 "attention_weight": float(w[int(i)])}
                for i in top_idx
            ]

        direction = "UP" if p_up >= 0.5 else "DOWN"
        confidence = p_up if direction == "UP" else 1.0 - p_up
        if p_up_raw is None:
            p_up_raw = p_up

        reason_codes = [
            f"direction={direction}",
            f"confidence={confidence:.4f}",
            f"attribution_method={result['method']}",
        ]

        explanation = {
            "ticker": ticker,
            "date": str(date),
            "p_up_raw": float(p_up_raw),
            "p_up": float(p_up),
            "direction": direction,
            "confidence": float(confidence),
            "indicator_values": indicator_values or {},
            "top_features": result["top_features"],
            "attribution_method": result["method"],
            "attribution_shape": result["attribution_shape"],
            "attention_evidence": attention_evidence,
            "reason_codes": reason_codes,
        }
        explanation["llm_narrative"] = (
            self._llm_narrate(explanation)
            if self.llm_cfg.get("enabled", False)
            else self._deterministic_narrate(explanation)
        )
        return explanation

    def _deterministic_narrate(self, explanation: dict) -> str:
        feats = explanation.get("top_features", [])
        if not feats:
            return "No dominant features identified."
        parts = [f"{f['feature']} (|attribution|={f['mean_abs_shap']:.4f})"
                 for f in feats[:3]]
        return (
            f"{explanation['ticker']} on {explanation['date']}: "
            f"{explanation['direction']} with confidence "
            f"{explanation['confidence']:.1%}. The prediction is primarily driven by "
            + ", ".join(parts)
            + ". Attention evidence indicates which days in the 30-day lookback "
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


__all__ = [
    "DEFAULT_BACKGROUND_SIZE",
    "GROUNDING_INSTRUCTIONS",
    "ExplainerAgent",
    "sample_background",
]
