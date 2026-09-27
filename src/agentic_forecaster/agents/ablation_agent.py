"""Ablation Agent — real ablations WITHOUT retraining what already exists.

Ablations:

1. ``attention_vs_plain_lstm`` — Attention-LSTM vs the plain-LSTM baseline.
2. ``raw_vs_calibrated``      — uncalibrated vs temperature-calibrated probs.
3. ``ohlcv_only_vs_full``     — OHLCV-only vs OHLCV + Phase-1 technical features.

The primary Attention-LSTM, the plain-LSTM baseline and the full-feature result
are **reused** from the already-trained models handed in by the Pipeline.  The
only additional neural model trained here is the OHLCV-only Attention-LSTM, so
a ticker/fold costs three neural fits in total (primary, plain-LSTM baseline,
OHLCV-only) plus the lightweight sklearn/majority baselines.
"""

from __future__ import annotations

import logging

from agentic_forecaster.agents.model_agent import FittedModel, ModelAgent
from agentic_forecaster.data.agent import DataAgent
from agentic_forecaster.evaluation import compute_metrics
from agentic_forecaster.models import AttentionLSTM
from agentic_forecaster.utils import seed_everything

logger = logging.getLogger("agentic_forecaster.agents.ablation")

TRACKED_KEYS = ("accuracy", "f1", "brier_calibrated", "ece_calibrated", "roc_auc")


class AblationAgent:
    def __init__(self, config: dict):
        self.config = config

    def run_ticker(
        self,
        ticker: str,
        dataset,
        fold: str = "fold_0",
        device: str | None = None,
        already_trained_primary: FittedModel | None = None,
        already_trained_baselines: dict[str, FittedModel] | None = None,
    ) -> dict:
        """Build the ablation table for one ticker.

        ``already_trained_primary`` / ``already_trained_baselines`` are reused
        as-is when supplied, so no duplicate neural training occurs.
        """
        if not self.config.get("ablations", {}).get("enabled", True):
            return {}
        seed_everything(int(self.config.get("experiment", {}).get("seed", 42)))
        agent = ModelAgent(self.config)
        reused = already_trained_primary is not None

        primary = already_trained_primary or agent.train_ticker(
            ticker, dataset, fold=fold, device=device
        )
        baselines = already_trained_baselines
        if baselines is None:
            baselines = agent.train_baselines(
                ticker, dataset, fold=fold, device=device
            )

        results: dict[str, dict] = {
            "attention_lstm": {k: primary.metrics.get(k) for k in TRACKED_KEYS},
            "attention_lstm_calibrated": {k: primary.metrics.get(k)
                                          for k in TRACKED_KEYS},
            "attention_lstm_raw": compute_metrics(
                dataset.test.y, primary.predict_proba_raw(dataset.test.X)
            ),
            "ohlcv_plus_technical": {k: primary.metrics.get(k)
                                     for k in TRACKED_KEYS},
        }
        results["_reused_primary"] = reused

        results["raw_vs_calibrated"] = {
            "brier_raw": primary.metrics.get("brier_raw"),
            "brier_calibrated": primary.metrics.get("brier_calibrated"),
            "ece_raw": primary.metrics.get("ece_raw"),
            "ece_calibrated": primary.metrics.get("ece_calibrated"),
            "temperature": primary.temperature,
        }

        plain = (baselines or {}).get("lstm")
        if plain is not None:
            results["plain_lstm"] = {k: plain.metrics.get(k) for k in TRACKED_KEYS}
        results["attention_vs_plain_lstm"] = {
            "attention_lstm_accuracy": results["attention_lstm"].get("accuracy"),
            "plain_lstm_accuracy": (
                plain.metrics.get("accuracy") if plain is not None else None
            ),
            "delta_accuracy": (
                (results["attention_lstm"].get("accuracy") or 0.0)
                - ((plain.metrics.get("accuracy") or 0.0) if plain is not None else 0.0)
            ),
        }

        results["ohlcv_only"] = self._ohlcv_only(ticker, device)
        return results

    def _ohlcv_only(self, ticker: str, device: str | None) -> dict:
        """Train the ONLY extra neural model: Attention-LSTM on OHLCV only."""
        try:
            dataset = DataAgent(self.config).run(ticker=ticker, indicators=[])
        except Exception as exc:
            logger.warning("OHLCV-only ablation unavailable for %s: %s", ticker, exc)
            return {"error": str(exc)}

        if not len(dataset.train.y):
            return {"error": "no samples for OHLCV-only variant"}

        agent = ModelAgent(self.config)
        cfg = self.config["models"]["attention_lstm"]
        model = AttentionLSTM(
            input_size=dataset.train.X.shape[-1],
            hidden_size=int(cfg.get("hidden_size", 64)),
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.2)),
        )
        trainer = agent._trainer(model, cfg, device)
        trainer.fit(dataset.train.X, dataset.train.y,
                    dataset.val.X, dataset.val.y)
        p = agent._raw(model, dataset.test.X, trainer.device)
        return compute_metrics(dataset.test.y, p)
