#!/usr/bin/env python3
"""Real-data smoke test for ONE ticker (RELIANCE).

Pipeline:
    raw 1-minute CSV -> daily resampling -> features -> 30-day sequences
    -> ticker-specific scaler -> 3-epoch Attention-LSTM
    -> temperature calibration -> prediction -> SHAP/attention
    -> ATR risk -> HTML/PDF report -> checkpoint reload -> re-prediction

Usage:
    python scripts/smoke_test_real.py [--ticker RELIANCE]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.agents.explainer_agent import ExplainerAgent
from agentic_forecaster.agents.model_agent import ModelAgent
from agentic_forecaster.agents.report_agent import ReportAgent
from agentic_forecaster.config import get_env_roots, load_config
from agentic_forecaster.data.agent import DataAgent
from agentic_forecaster.models.checkpoint import load_model_bundle, save_model_bundle
from agentic_forecaster.risk import RiskAgent
from agentic_forecaster.utils import seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="RELIANCE")
    parser.add_argument("--config", default="configs/paper.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    seed_everything(42)
    config = load_config(args.config)
    roots = get_env_roots()

    print(f"=== 1. DataAgent: {args.ticker} ===")
    da = DataAgent(config)
    ds = da.run(ticker=args.ticker)
    print(f"  train={len(ds.train.y)} val={len(ds.val.y)} test={len(ds.test.y)} features={len(ds.feature_names)}")
    print(f"  features: {ds.feature_names}")
    print(f"  train: {ds.train.dates[0]} .. {ds.train.dates[-1]}")
    print(f"  val:   {ds.val.dates[0]} .. {ds.val.dates[-1]}")
    print(f"  test:  {ds.test.dates[0]} .. {ds.test.dates[-1]}")

    print(f"=== 2. ModelAgent: train {args.ticker} ===")
    ma = ModelAgent(config)
    fitted = ma.train_ticker(args.ticker, ds, device="cpu")
    print(f"  metrics: {fitted.metrics}")
    print(f"  temperature: {fitted.temperature:.4f}")

    print("=== 3. Save checkpoint ===")
    bundle = save_model_bundle(
        fitted, Path(roots["AGENTIC_MODEL_ROOT"]) / "trained" / args.ticker / "fold_0"
    )
    print(f"  bundle: {bundle}")

    print("=== 4. Load checkpoint ===")
    loaded = load_model_bundle(bundle, device="cpu")
    print(f"  loaded ticker={loaded.ticker} T={loaded.temperature:.4f}")

    print("=== 5. Predict (calibrated) ===")
    p_cal = loaded.predict_proba(ds.test.X)
    p_raw = loaded.predict_proba_raw(ds.test.X)
    print(f"  p_cal range: [{p_cal.min():.4f}, {p_cal.max():.4f}]")
    print(f"  p_raw range: [{p_raw.min():.4f}, {p_raw.max():.4f}]")
    print(f"  calibrated != raw: {not np.allclose(p_cal, p_raw)}")

    print("=== 6. Explain last prediction ===")
    explainer = ExplainerAgent(config)
    last_idx = -1
    x = ds.test.X[last_idx]
    explanation = explainer.explain_prediction(
        loaded, x, args.ticker, str(ds.test.dates[last_idx]), p_cal[last_idx]
    )
    print(f"  direction={explanation['direction']} confidence={explanation['confidence']:.4f}")
    print(f"  top features: {explanation['top_features'][:3]}")
    print(f"  shap_method: {explanation['shap_method']}")

    print("=== 7. ATR Risk ===")
    ra = RiskAgent()
    risk = ra.decide(args.ticker, str(ds.test.dates[last_idx]), p_cal[last_idx], close=100.0, atr=2.0)
    print(f"  direction={risk.direction} level={risk.confidence_level}")
    print(f"  SL={risk.stop_loss:.2f} TP={risk.take_profit:.2f} RRR={risk.rrr:.4f} RiskScore={risk.risk_score:.6f}")

    print("=== 8. Report ===")
    report_agent = ReportAgent(config)
    report = report_agent.run(
        ticker=args.ticker,
        origin_date=str(ds.test.dates[last_idx]),
        target_date=str(ds.test.target_dates[last_idx]),
        latest_close=100.0,
        p_up_raw=float(p_raw[last_idx]),
        p_up_calibrated=float(p_cal[last_idx]),
        direction=explanation["direction"],
        confidence=explanation["confidence"],
        confidence_level=risk.confidence_level,
        explanation=explanation,
        risk_decision=risk,
        metrics=fitted.metrics,
        technical_indicators={"rsi_14": 55.0, "macd": 0.5, "atr_14": 2.0},
        model_id="attention_lstm_v1",
        config_id="paper_v1",
        run_id="smoke_test",
        output_dir=Path(roots["AGENTIC_OUTPUT_ROOT"]) / "reports",
    )
    print(f"  report: {report}")

    print("=== 9. Re-predict after reload ===")
    p_reload = loaded.predict_proba(ds.test.X)
    print(f"  re-prediction matches: {np.allclose(p_cal, p_reload)}")

    print()
    print("=== SMOKE TEST COMPLETE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
