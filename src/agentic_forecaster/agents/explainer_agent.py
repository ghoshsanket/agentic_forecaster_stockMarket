"""Explainer Agent — third agent in the five-agent workflow.

Combines SHAP feature attribution with attention evidence to produce an
explanation bundle for each prediction.  When an LLM backend is configured
the bundle is narrated by the LLM; otherwise a deterministic template
fallback is used.
"""

from __future__ import annotations

import logging

import numpy as np
import openai

from agentic_forecaster.explainability import ShapExplainer, extract_attention_evidence

logger = logging.getLogger("agentic_forecaster.agents.explainer")


class ExplainerAgent:
    def __init__(self, config: dict):
        self.config = config
        self.exp_cfg = config.get("explainability", {})
        self.llm_cfg = config.get("reporting", {}).get("llm", {})

    def run(self, fitted_model, dataset) -> dict:
        X_test = dataset.test.X
        background = dataset.train.X[
            np.random.default_rng(0).choice(
                len(dataset.train.X),
                size=min(int(self.exp_cfg.get("background_samples", 100)), len(dataset.train.X)),
                replace=False,
            )
        ]
        shap = ShapExplainer(fitted_model.model, background, dataset.feature_names)
        shap_values = shap.explain(X_test[:100])
        top_features = shap.top_features(shap_values, k=int(self.exp_cfg.get("top_k_features", 5)))

        attention_evidence = None
        if self.exp_cfg.get("attention_evidence", False) and fitted_model.kind == "torch":
            attention_evidence = extract_attention_evidence(
                fitted_model.model,
                X_test[:100],
                dataset.test.dates[:100],
                top_k=5,
            )

        explanation = {
            "top_features": top_features,
            "attention_evidence": attention_evidence,
            "shap_summary": {
                "n_samples": len(shap_values),
                "mean_abs_shap_per_feature": {
                    dataset.feature_names[i]: float(np.abs(shap_values[:, :, i]).mean())
                    for i in range(len(dataset.feature_names))
                },
            },
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

            client = OpenAI()
            response = client.chat.completions.create(
                model=self.llm_cfg.get("model", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are a financial explainer."},
                    {"role": "user", "content": str(explanation)},
                ],
            )
            return response.choices[0].message.content or ""
        except (openai.OpenAIError, ValueError, RuntimeError) as exc:
            logger.warning("LLM narration failed (%s); using fallback", exc)
            return self._deterministic_narrate(explanation)
