"""Ablation Agent — executes real ablation variants and records real metrics.

Ablations implemented (each actually trains and evaluates a model):

1. ``attention_vs_plain_lstm`` — Attention-LSTM vs the plain-LSTM baseline.
2. ``raw_vs_calibrated``      — uncalibrated vs temperature-calibrated probs.
3. ``ohlcv_only_vs_full``     — OHLCV-only vs OHLCV + Phase-1 technical features.

All numbers come from real training runs; nothing here is a placeholder.
"""

from __future__ import annotations

import logging

from agentic_forecaster.agents.model_agent import ModelAgent
from agentic_forecaster.data.agent import DataAgent
from agentic_forecaster.evaluation import compute_metrics
from agentic_forecaster.models import AttentionLSTM
from agentic_forecaster.utils import seed_everything

logger = logging.getLogger("agentic_forecaster.agents.ablation")

TRACKED_KEYS = ("accuracy", "f1", "brier_calibrated", "ece_calibrated", "roc_auc")


class AblationAgent:
    def __init__(self, config: dict):
        self.config = config

    def run_ticker(self, ticker: str, dataset, fold: str = "fold_0",
                   device: str | None = None) -> dict:
        if not self.config.get("ablations", {}).get("enabled", True):
            return {}
        seed_everything(int(self.config.get("experiment", {}).get("seed", 42)))
        agent = ModelAgent(self.config)
        results: dict[str, dict] = {}

        # 1 + 2 -----------------------------------------------------------
        primary = agent.train_ticker(ticker, dataset, fold=fold, device=device)
        results["attention_lstm"] = {
            k: primary.metrics.get(k) for k in TRACKED_KEYS
        }
        results["attention_lstm_raw"] = {
            k: v for k, v in compute_metrics(
                dataset.test.y, primary.predict_proba_raw(dataset.test.X)
            ).items()
        }
        results["attention_lstm_calibrated"] = {
            k: primary.metrics.get(k) for k in TRACKED_KEYS
        }
        results["raw_vs_calibrated"] = {
            "brier_raw": primary.metrics.get("brier_raw"),
            "brier_calibrated": primary.metrics.get("brier_calibrated"),
            "ece_raw": primary.metrics.get("ece_raw"),
            "ece_calibrated": primary.metrics.get("ece_calibrated"),
            "temperature": primary.temperature,
        }

        baselines = agent.train_baselines(ticker, dataset, fold=fold, device=device)
        if "lstm" in baselines:
            results["plain_lstm"] = {k: baselines["lstm"].metrics.get(k)
                                     for k in TRACKED_KEYS}
        results["attention_vs_plain_lstm"] = {
            "attention_lstm_accuracy": results["attention_lstm"].get("accuracy"),
            "plain_lstm_accuracy": results.get("plain_lstm", {}).get("accuracy"),
            "delta_accuracy": (
                (results["attention_lstm"].get("accuracy") or 0.0)
                - (results.get("plain_lstm", {}).get("accuracy") or 0.0)
            ),
        }

        # 3 ---------------------------------------------------------------
        results["ohlcv_only"] = self._ohlcv_only(ticker, device)
        results["ohlcv_plus_technical"] = {
            k: primary.metrics.get(k) for k in TRACKED_KEYS
        }
        return results

    def _ohlcv_only(self, ticker: str, device) -> dict:
        """Retrain the primary architecture on OHLCV features only."""
        try:
            data_agent = DataAgent(self.config)
            dataset = data_agent.run(ticker=ticker, indicators=[])
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
        trainer.fit(dataset.train.X, dataset.train.y, dataset.val.X, dataset.val.y)
        p = agent._raw(model, dataset.test.X, trainer.device)
        return {k: v for k, v in compute_metrics(dataset.test.y, p).items()}
