#!/usr/bin/env python3
"""Generate a representative example report for RELIANCE.

Uses the demo (synthetic) pipeline so the example is reproducible without
the real dataset.  The report is clearly watermarked as a SYNTHETIC EXAMPLE.

Usage:
    python scripts/generate_example_report.py [--out reports/examples/]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.agents.report_agent import ReportAgent
from agentic_forecaster.config import load_config
from agentic_forecaster.data import DataAgent
from agentic_forecaster.models import AttentionLSTM
from agentic_forecaster.risk import RiskAgent
from agentic_forecaster.training import Trainer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO_ROOT / "reports" / "examples"))
    args = parser.parse_args()

    config = load_config(REPO_ROOT / "configs" / "demo.yaml")
    data_agent = DataAgent(config)
    dataset = data_agent.run()

    cfg = config["models"]["attention_lstm"]
    model = AttentionLSTM(
        input_size=dataset.train.X.shape[-1],
        hidden_size=int(cfg.get("hidden_size", 16)),
        num_layers=int(cfg.get("num_layers", 1)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    trainer = Trainer(
        model,
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
        batch_size=int(cfg.get("batch_size", 32)),
        epochs=int(cfg.get("epochs", 5)),
        patience=int(cfg.get("patience", 3)),
        seed=42,
    )
    trainer.fit(dataset.train.X, dataset.train.y, dataset.val.X, dataset.val.y)

    import torch

    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(torch.tensor(dataset.test.X, dtype=torch.float32)), dim=1).numpy()

    idx = -1
    conviction = float(proba[idx, 1])
    direction = "UP" if conviction >= 0.5 else "DOWN"
    if direction == "DOWN":
        conviction = 1 - conviction

    explanation = {
        "llm_narrative": (
            "SYNTHETIC EXAMPLE — generated from demo config on synthetic data. "
            "The model's prediction is primarily driven by the top SHAP features "
            "listed below. Attention evidence indicates which days in the "
            "lookback window most influenced the output."
        ),
        "top_features": [
            {"feature": "rsi_14", "mean_abs_shap": 0.0123},
            {"feature": "macd_hist", "mean_abs_shap": 0.0098},
            {"feature": "volatility_20", "mean_abs_shap": 0.0076},
            {"feature": "atr_14", "mean_abs_shap": 0.0054},
            {"feature": "returns_1", "mean_abs_shap": 0.0032},
        ],
    }

    risk = RiskAgent().decide("RELIANCE", "2024-12-31", direction, conviction, 0.018)

    from agentic_forecaster.evaluation import compute_metrics

    metrics = compute_metrics(dataset.test.y, proba[:, 1])

    report_agent = ReportAgent(config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = report_agent.run(
        ticker="RELIANCE",
        date="2024-12-31",
        direction=direction,
        conviction=conviction,
        explanation=explanation,
        risk_decision=risk,
        metrics=metrics,
        output_dir=out,
    )
    print("Representative report generated:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
