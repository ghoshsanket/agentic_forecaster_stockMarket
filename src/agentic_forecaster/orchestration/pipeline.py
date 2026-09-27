"""Per-stock orchestration for the five-agent workflow.

Workflow (per stock, no pooling):

    DataAgent.run_ticker(ticker)
        -> ModelAgent.train_ticker(ticker, dataset)
        -> ModelAgent.train_baselines(ticker, dataset)
        -> ExplainerAgent.explain_prediction(...)
        -> RiskAgent.decide(...)
        -> ReportAgent.run(...)

``Pipeline`` is a thin wrapper over this for a single ticker (used by tests
and the smoke test).  ``walk_forward`` runs it across the paper's folds and
writes the aggregate reproduction outputs.

There is deliberately NO pooled/"ALL" model and NO position-sizing logic:
the Phase-1 risk implementation is ATR-based.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_forecaster.agents.explainer_agent import ExplainerAgent
from agentic_forecaster.agents.model_agent import FittedModel, ModelAgent
from agentic_forecaster.agents.report_agent import ReportAgent
from agentic_forecaster.data.agent import DataAgent
from agentic_forecaster.risk import RiskAgent
from agentic_forecaster.utils import atomic_json_dump, ensure_dir, seed_everything

logger = logging.getLogger("agentic_forecaster.orchestration")

REPORT_INDICATORS = (
    "close", "atr_14", "rsi_14", "macd", "macd_signal",
    "realized_volatility_20",
)


@dataclass
class PredictionRecord:
    """One calibrated prediction, with everything needed for auditing."""
    ticker: str
    fold: str
    origin_date: str
    target_date: str
    y_true: float
    p_up_raw: float
    p_up_calibrated: float
    direction: str
    confidence: float


@dataclass
class PipelineResult:
    ticker: str = ""
    fold: str = ""
    dataset: object = None
    primary: FittedModel | None = None
    baselines: dict = field(default_factory=dict)
    ablations: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    predictions: pd.DataFrame | None = None
    report: dict | None = None
    explanation: dict | None = None
    run_dir: str = ""


class Pipeline:
    """Runs the full five-agent workflow for ONE stock."""

    def __init__(self, config: dict, fold: str = "fold_0"):
        self.config = config
        self.fold = fold
        self.seed = int(config.get("experiment", {}).get("seed", 42))
        self.run_root = ensure_dir(
            Path(config.get("experiment", {}).get("output_dir", "./outputs/run"))
            / fold
        )

    def run(self, ticker: str, device: str | None = None,
            n_reports: int = 1, report_dir: Path | None = None) -> PipelineResult:
        seed_everything(self.seed)

        data_agent = DataAgent(self.config)
        dataset = data_agent.run(ticker=ticker)

        model_agent = ModelAgent(self.config)
        primary = model_agent.train_ticker(ticker, dataset, fold=self.fold, device=device)
        baselines = model_agent.train_baselines(ticker, dataset, fold=self.fold, device=device)

        # Reuse the already-trained primary and baselines for the ablations;
        # the only extra neural fit is the OHLCV-only variant.
        from agentic_forecaster.agents.ablation_agent import AblationAgent
        ablations = AblationAgent(self.config).run_ticker(
            ticker, dataset, fold=self.fold, device=device,
            already_trained_primary=primary,
            already_trained_baselines=baselines,
        )

        p_raw = primary.predict_proba_raw(dataset.test.X)
        p_cal = primary.predict_proba(dataset.test.X)
        predictions = pd.DataFrame({
            "date": dataset.test.dates.astype("datetime64[D]").astype(str),
            "ticker": ticker,
            "y": dataset.test.y.astype(float),
            "raw_p_up": p_raw.astype(float),
            "calibrated_p_up": p_cal.astype(float),
        })
        predictions["fold"] = self.fold
        predictions["direction"] = np.where(p_cal >= 0.5, "UP", "DOWN")
        predictions["confidence"] = np.where(
            p_cal >= 0.5, p_cal, 1.0 - p_cal
        )
        predictions["origin_date"] = predictions["date"]
        predictions["target_date"] = dataset.test.target_dates.astype(
            "datetime64[D]"
        ).astype(str)

        explainer = ExplainerAgent(self.config)
        risk_agent = RiskAgent()
        report_agent = ReportAgent(self.config)

        n = min(n_reports, len(dataset.test.X))
        reports, explanation = [], None
        for i in range(n - 1, -1, -1):
            snapshot = data_agent.origin_snapshot(dataset.test, i)
            indicators = {k: v for k, v in snapshot.items() if k in REPORT_INDICATORS}
            explanation = explainer.explain_prediction(
                primary,
                dataset.test.X[i],
                ticker=ticker,
                date=str(dataset.test.dates[i]),
                p_up=float(p_cal[i]),
                indicator_values=indicators,
                p_up_raw=float(p_raw[i]),
            )
            close = float(snapshot.get("close", np.nan))
            atr = float(snapshot.get("atr_14", np.nan))
            if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
                continue
            risk = risk_agent.decide(
                ticker=ticker,
                date=str(dataset.test.dates[i]),
                p_up=float(p_cal[i]),
                close=close,
                atr=atr,
            )
            reports.append(report_agent.run(
                ticker=ticker,
                origin_date=str(dataset.test.dates[i]),
                target_date=str(dataset.test.target_dates[i]),
                latest_close=close,
                p_up_raw=float(p_raw[i]),
                p_up_calibrated=float(p_cal[i]),
                direction=risk.direction,
                confidence=risk.confidence,
                confidence_level=risk.confidence_level,
                explanation=explanation,
                risk_decision=risk,
                metrics=primary.metrics,
                technical_indicators=indicators,
                model_id=f"{ticker}:{self.fold}:attention_lstm",
                config_id=str(self.config.get("experiment", {}).get("name", "paper")),
                output_dir=report_dir or (self.run_root / "reports"),
            ))

        metrics = {
            "attention_lstm": primary.metrics,
            **{name: fm.metrics for name, fm in baselines.items()},
        }
        atomic_json_dump(metrics, self.run_root / f"metrics_{ticker}.json")

        return PipelineResult(
            ticker=ticker, fold=self.fold, dataset=dataset,
            primary=primary, baselines=baselines, ablations=ablations,
            metrics=metrics, predictions=predictions,
            report=reports[0] if reports else None,
            explanation=explanation, run_dir=str(self.run_root),
        )
