"""Streamlit demo app for the Explanation-First Agentic Forecaster.

Run with:
    streamlit run app/streamlit_app.py

The app loads a processed dataset (or generates synthetic data), runs the
Attention-LSTM, and displays the forecast, explanation and risk decision.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_forecaster.config import load_config
from agentic_forecaster.data import DataAgent
from agentic_forecaster.models import AttentionLSTM
from agentic_forecaster.risk import RiskAgent


def main() -> None:
    st.set_page_config(page_title="Agentic Forecaster", layout="wide")
    st.title("Explanation-First Agentic Forecaster for Stock Market")
    st.caption("DOI: 10.1109/IEMENTECH202669403.2026.11434302")

    config_path = Path(__file__).resolve().parents[1] / "configs" / "demo.yaml"
    config = load_config(config_path)

    with st.spinner("Running pipeline..."):
        data_agent = DataAgent(config)
        dataset = data_agent.run()

        cfg = config["models"]["attention_lstm"]
        model = AttentionLSTM(
            input_size=dataset.train.X.shape[-1],
            hidden_size=int(cfg.get("hidden_size", 16)),
            num_layers=int(cfg.get("num_layers", 1)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
        from agentic_forecaster.training import Trainer

        trainer = Trainer(
            model,
            learning_rate=float(cfg.get("learning_rate", 1e-3)),
            batch_size=int(cfg.get("batch_size", 32)),
            epochs=int(cfg.get("epochs", 5)),
            patience=int(cfg.get("patience", 3)),
            seed=int(config.get("experiment", {}).get("seed", 42)),
        )
        trainer.fit(dataset.train.X, dataset.train.y, dataset.val.X, dataset.val.y)

        proba = model.eval() and _predict(model, dataset.test.X)

    st.success("Pipeline complete")

    st.subheader("Latest prediction")
    idx = -1
    ticker = str(dataset.test.tickers[idx])
    date = str(dataset.test.dates[idx])
    up_proba = float(proba[idx, 1])
    direction = "UP" if up_proba >= 0.5 else "DOWN"
    conviction = up_proba if direction == "UP" else 1 - up_proba

    col1, col2, col3 = st.columns(3)
    col1.metric("Ticker", ticker)
    col2.metric("Direction", direction)
    col3.metric("Conviction", f"{conviction:.1%}")

    risk = RiskAgent().decide(ticker, date, direction, conviction, 0.02)
    st.write(f"**Risk level:** {risk.risk_level} | "
             f"**Position size:** {risk.position_size:.2%} | "
             f"**Stop loss:** {risk.stop_loss:.2%}")

    st.subheader("Test-split metrics")
    from agentic_forecaster.evaluation import compute_metrics

    metrics = compute_metrics(dataset.test.y, proba[:, 1])
    st.json(metrics)

    st.subheader("Prediction probability distribution")
    st.bar_chart(pd.DataFrame({"up_proba": proba[:, 1]}))


def _predict(model, X: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()


if __name__ == "__main__":
    main()
