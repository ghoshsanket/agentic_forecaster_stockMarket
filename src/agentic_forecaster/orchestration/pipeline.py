"""End-to-end pipeline orchestrating the five agents.

Workflow:
    DataAgent -> ModelAgent -> ExplainerAgent -> RiskAgent -> ReportAgent

Heavy runtime artefacts (checkpoints, processed data, logs) are written under
Category B roots (``$AGENTIC_*``).  Final submission-worthy artefacts are
exported into the repository by ``scripts/package_submission.py`` or by
``reproduce-paper --export-final-results``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from agentic_forecaster.agents import ExplainerAgent, ModelAgent, ReportAgent
from agentic_forecaster.data import DataAgent
from agentic_forecaster.risk import RiskAgent
from agentic_forecaster.utils import atomic_json_dump, ensure_dir, seed_everything

logger = logging.getLogger("agentic_forecaster.orchestration")


@dataclass
class PipelineResult:
    dataset: object = None
    fitted_models: dict = field(default_factory=dict)
    explanation: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    reports: list = field(default_factory=list)
    run_dir: str = ""


class Pipeline:
    def __init__(self, config: dict):
        self.config = config
        self.exp_cfg = config.get("experiment", {})
        self.seed = int(self.exp_cfg.get("seed", 42))
        self.output_root = Path(
            self.exp_cfg.get("output_dir", "./outputs/run")
        )

    def run(self) -> PipelineResult:
        seed_everything(self.seed)
        run_dir = ensure_dir(self.output_root)

        logger.info("=== Agent 1/5: Data ===")
        data_agent = DataAgent(self.config)
        dataset = data_agent.run()
        processed_root = self.config["data"].get("processed_root")
        if processed_root:
            dataset.save(processed_root)

        logger.info("=== Agent 2/5: Model ===")
        model_agent = ModelAgent(self.config)
        fitted = model_agent.run(dataset)

        cal_cfg = self.config.get("calibration", {})
        if cal_cfg.get("method") == "temperature":
            import torch

            from agentic_forecaster.calibration import TemperatureCalibrator
            for name, fm in fitted.items():
                if fm.kind == "torch":
                    calibrator = TemperatureCalibrator()
                    device = next(fm.model.parameters()).device
                    val_logits = fm.model(torch.tensor(dataset.val.X, dtype=torch.float32, device=device)).detach().cpu().numpy()
                    calibrator.fit(val_logits, dataset.val.y)
                    fm.calibration = calibrator.to_dict()
                    logger.info("Calibrated %s: T=%.4f", name, calibrator.temperature)

        logger.info("=== Agent 3/5: Explainer ===")
        explainer_agent = ExplainerAgent(self.config)
        explanation = explainer_agent.run(fitted["attention_lstm"], dataset)

        logger.info("=== Agent 4/5: Risk ===")
        risk_agent = RiskAgent(
            kelly_fraction=float(self.config.get("risk", {}).get("kelly_fraction", 0.25)),
            max_position_pct=float(self.config.get("risk", {}).get("max_position_pct", 0.10)),
            stop_loss_pct=float(self.config.get("risk", {}).get("stop_loss_pct", 0.05)),
            volatility_target=float(self.config.get("risk", {}).get("volatility_target", 0.15)),
        )

        logger.info("=== Agent 5/5: Report ===")
        report_agent = ReportAgent(self.config)
        reports = []
        proba = fitted["attention_lstm"].predict_proba(dataset.test.X)
        for i in range(min(10, len(proba))):
            ticker = str(dataset.test.tickers[i])
            date = str(dataset.test.dates[i])
            direction = "UP" if proba[i, 1] >= 0.5 else "DOWN"
            conviction = float(proba[i, 1] if direction == "UP" else proba[i, 0])
            risk = risk_agent.decide(ticker, date, direction, conviction, 0.02)
            rep = report_agent.run(
                ticker=ticker, date=date, direction=direction,
                conviction=conviction, explanation=explanation,
                risk_decision=risk,
                metrics=fitted["attention_lstm"].metrics,
                output_dir=run_dir / "reports",
            )
            reports.append(rep)

        metrics_summary = {name: fm.metrics for name, fm in fitted.items()}
        atomic_json_dump(metrics_summary, run_dir / "metrics.json")
        atomic_json_dump(explanation, run_dir / "explanation.json")

        # Persist arrays needed by the figure generator (Category B runtime).
        alm = fitted["attention_lstm"]
        proba = alm.predict_proba(dataset.test.X)
        np.savez_compressed(
            run_dir / "test_predictions.npz",
            y=dataset.test.y,
            p=proba[:, 1],
        )
        if alm.kind == "torch":
            import torch

            alm.model.eval()
            device = next(alm.model.parameters()).device
            with torch.no_grad():
                _, w = alm.model(
                    torch.tensor(dataset.test.X, dtype=torch.float32, device=device),
                    return_attention=True,
                )
            np.save(run_dir / "attention_weights.npy", w.cpu().numpy())
        atomic_json_dump(alm.history, run_dir / "training_history.json")

        return PipelineResult(
            dataset=dataset,
            fitted_models=fitted,
            explanation=explanation,
            metrics=metrics_summary,
            reports=reports,
            run_dir=str(run_dir),
        )
