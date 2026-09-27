"""Streamlit app for the Explanation-First Agentic Forecaster.

Two modes:
    DEMO  — trains a small synthetic model at startup (no data required).
    REAL  — loads trained ticker models from $AGENTIC_MODEL_ROOT and runs
            real inference. Does NOT retrain.

Run with:
    streamlit run app/streamlit_app.py
    streamlit run app/streamlit_app.py -- --mode real
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_forecaster.config import get_env_roots, load_config
from agentic_forecaster.models.checkpoint import load_model_bundle


def main() -> None:
    st.set_page_config(page_title="Agentic Forecaster", layout="wide")
    st.title("Explanation-First Agentic Forecaster for Stock Market")
    st.caption("DOI: 10.1109/IEMENTECH202669403.2026.11434302")

    mode = st.sidebar.radio("Mode", ["DEMO", "REAL"])

    if mode == "DEMO":
        _run_demo()
    else:
        _run_real()


def _run_demo() -> None:
    from agentic_forecaster.data import DataAgent
    from agentic_forecaster.models import AttentionLSTM
    from agentic_forecaster.training import Trainer

    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "demo.yaml")
    with st.spinner("Training synthetic demo model..."):
        data_agent = DataAgent(config)
        dataset = data_agent.run(ticker="SYN00")
        cfg = config["models"]["attention_lstm"]
        model = AttentionLSTM(
            input_size=dataset.train.X.shape[-1],
            hidden_size=int(cfg.get("hidden_size", 16)),
            num_layers=int(cfg.get("num_layers", 1)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
        trainer = Trainer(model, epochs=int(cfg.get("epochs", 5)),
                          batch_size=int(cfg.get("batch_size", 32)),
                          patience=int(cfg.get("patience", 3)), device="cpu")
        trainer.fit(dataset.train.X, dataset.train.y, dataset.val.X, dataset.val.y)
        p_cal = model.eval() and _predict(model, dataset.test.X)

    st.success("Demo pipeline complete")
    st.write("Test metrics:")
    from agentic_forecaster.evaluation import compute_metrics
    st.json(compute_metrics(dataset.test.y, p_cal))


def _run_real() -> None:
    roots = get_env_roots()
    model_root = Path(roots["AGENTIC_MODEL_ROOT"]) / "trained"

    if not model_root.exists():
        st.error(f"Model directory not found: {model_root}. Run train-all first.")
        return

    ticker_dirs = sorted([d.name for d in model_root.iterdir() if d.is_dir()])
    if not ticker_dirs:
        st.error("No trained models found. Run train-all first.")
        return

    ticker = st.sidebar.selectbox("Ticker", ticker_dirs)
    bundle = model_root / ticker / "fold_0"

    if not bundle.exists():
        st.error(f"No fold_0 bundle for {ticker}")
        return

    with st.spinner(f"Loading {ticker} model..."):
        fitted = load_model_bundle(bundle, device="cpu")

    st.success(f"Loaded {ticker} model (T={fitted.temperature:.4f})")

    st.subheader("Model info")
    st.write(f"Ticker: {fitted.ticker}")
    st.write(f"Fold: {fitted.fold}")
    st.write(f"Temperature: {fitted.temperature:.4f}")
    st.write("Metrics:")
    st.json(fitted.metrics)


def _predict(model, X: np.ndarray) -> np.ndarray:
    import torch
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()


if __name__ == "__main__":
    main()
