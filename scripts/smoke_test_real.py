#!/usr/bin/env python3
"""Real-data smoke test for ONE ticker.

Performs the genuine pipeline and prints every real value used::

    raw 1-minute CSV
      -> daily OHLCV
      -> actual Phase-1 features
      -> real 30-day sequence
      -> ticker-specific StandardScaler
      -> 3-epoch Attention-LSTM
      -> temperature calibration
      -> calibrated prediction
      -> SHAP / attention attribution
      -> ATR-based risk (real close, real ATR)
      -> HTML + PDF report
      -> save bundle -> reload -> identical prediction

No placeholder values (close=100, atr=2, rsi=55) are used anywhere.

Usage:
    python scripts/smoke_test_real.py [--ticker RELIANCE] [--config configs/paper.yaml]
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
from agentic_forecaster.orchestration.pipeline import REPORT_INDICATORS
from agentic_forecaster.risk import RiskAgent
from agentic_forecaster.utils import seed_everything

logger = logging.getLogger("smoke_test_real")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="RELIANCE")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    seed_everything(42)
    config = load_config(args.config)
    roots = get_env_roots()
    ticker = args.ticker.upper()

    # 1 ------------------------------------------------------------------
    print(f"=== 1. DataAgent: {ticker} ===")
    data_agent = DataAgent(config)
    ds = data_agent.run(ticker=ticker)
    print(f"  train={len(ds.train.y)}  val={len(ds.val.y)}  test={len(ds.test.y)}")
    print(f"  features ({len(ds.feature_names)}): {ds.feature_names}")
    print(f"  train window: {ds.train.dates[0]} .. {ds.train.dates[-1]}")
    print(f"  test  window: {ds.test.dates[0]} .. {ds.test.dates[-1]}")

    # 2 ------------------------------------------------------------------
    print(f"=== 2. Train Attention-LSTM ({ticker}, {config['models']['attention_lstm']['epochs']} epochs) ===")
    fitted = ModelAgent(config).train_ticker(ticker, ds, fold="fold_0", device=args.device)
    for key in ("accuracy", "f1", "brier_raw", "brier_calibrated",
                "ece_raw", "ece_calibrated", "roc_auc"):
        if key in fitted.metrics:
            print(f"  {key}: {fitted.metrics[key]:.6f}")
    print(f"  temperature: {fitted.temperature:.6f}")

    # 3 ------------------------------------------------------------------
    print("=== 3. Real unscaled values at the prediction origin ===")
    idx = len(ds.test.X) - 1
    origin_date = str(ds.test.dates[idx])
    target_date = str(ds.test.target_dates[idx])
    snapshot = data_agent.origin_snapshot(ds.test, idx)
    close = float(snapshot["close"])
    atr = float(snapshot["atr_14"])
    print(f"  origin_date   : {origin_date}")
    print(f"  target_date   : {target_date}  (must be > origin_date: "
          f"{target_date > origin_date})")
    for name in REPORT_INDICATORS:
        if name in snapshot:
            print(f"  {name:<26}: {snapshot[name]:.6f}")
    assert target_date > origin_date, "target_date must be after origin_date"
    assert close > 0 and atr > 0, "unscaled close/ATR must be usable"

    # 4 ------------------------------------------------------------------
    print("=== 4. Calibrated vs raw probability ===")
    p_cal = fitted.predict_proba(ds.test.X)
    p_raw = fitted.predict_proba_raw(ds.test.X)
    print(f"  p_up_raw       [{p_raw.min():.4f}, {p_raw.max():.4f}]")
    print(f"  p_up_calibrated[{p_cal.min():.4f}, {p_cal.max():.4f}]")
    print(f"  calibration changed the probabilities: {not np.allclose(p_raw, p_cal)}")
    print(f"  chosen sample p_up_raw={p_raw[idx]:.6f} p_up_calibrated={p_cal[idx]:.6f}")

    # 5 ------------------------------------------------------------------
    print("=== 5. SHAP + attention attribution (single prediction) ===")
    explainer = ExplainerAgent(config)
    explanation = explainer.explain_prediction(
        fitted, ds.test.X[idx], ticker, origin_date, float(p_cal[idx]),
        indicator_values={k: v for k, v in snapshot.items() if k in REPORT_INDICATORS},
        p_up_raw=float(p_raw[idx]),
    )
    print(f"  method: {explanation['attribution_method']}  shape={explanation['attribution_shape']}")
    for feat in explanation["top_features"][:5]:
        print(f"    {feat['feature']:<26} |attribution|={feat['mean_abs_shap']:.8f}")
    print(f"  top attention days: {explanation['attention_evidence'][:3]}")

    from agentic_forecaster.explainability.shap_explainer import ShapExplainer
    sg = ShapExplainer(fitted.model, fitted._background, fitted.feature_names)
    attrs = sg.explain(ds.test.X[idx:idx + 1])[0]
    nonzero = int((np.abs(attrs) > 1e-12).sum())
    print(f"  attribution shape={attrs.shape} finite={np.isfinite(attrs).all()} "
          f"nonzero={nonzero}/{attrs.size}")
    assert np.isfinite(attrs).all() and nonzero > 0, "attributions must be finite and non-zero"

    # 6 ------------------------------------------------------------------
    print("=== 6. ATR-based risk (real close / real ATR) ===")
    risk = RiskAgent().decide(ticker, origin_date, float(p_cal[idx]), close=close, atr=atr)
    print(f"  direction={risk.direction} confidence={risk.confidence:.6f} "
          f"level={risk.confidence_level}")
    print(f"  lambda_SL={risk.lambda_sl} lambda_TP={risk.lambda_tp}")
    print(f"  close={close:.4f} atr={atr:.4f}")
    print(f"  stop_loss={risk.stop_loss:.4f}  take_profit={risk.take_profit:.4f}")
    print(f"  RRR={risk.rrr:.6f}  RiskScore={risk.risk_score:.8f}")
    if risk.direction == "UP":
        assert abs(risk.stop_loss - (close - risk.lambda_sl * atr)) < 1e-9
    else:
        assert abs(risk.stop_loss - (close + risk.lambda_sl * atr)) < 1e-9

    # 7 ------------------------------------------------------------------
    print("=== 7. Report (HTML + PDF) ===")
    report = ReportAgent(config).run(
        ticker=ticker, origin_date=origin_date, target_date=target_date,
        latest_close=close, p_up_raw=float(p_raw[idx]),
        p_up_calibrated=float(p_cal[idx]), direction=risk.direction,
        confidence=risk.confidence, confidence_level=risk.confidence_level,
        explanation=explanation, risk_decision=risk, metrics=fitted.metrics,
        technical_indicators={k: v for k, v in snapshot.items() if k in REPORT_INDICATORS},
        model_id=f"{ticker}:fold_0:attention_lstm",
        config_id=str(config.get("experiment", {}).get("name", "paper")),
        run_id="smoke_test_real",
        output_dir=Path(roots["AGENTIC_OUTPUT_ROOT"]) / "reports" / "smoke",
    )
    print(f"  HTML: {report['html']}")
    print(f"  PDF : {report.get('pdf')}")
    assert Path(report["html"]).is_file()
    if report.get("pdf"):
        assert Path(report["pdf"]).is_file()

    # 8 ------------------------------------------------------------------
    print("=== 8. Save -> reload -> identical prediction ===")
    bundle = save_model_bundle(
        fitted,
        Path(roots["AGENTIC_MODEL_ROOT"]) / "trained" / ticker / "fold_0",
    )
    print(f"  bundle: {bundle}")
    for name in ("model.pt", "scaler.joblib", "calibration.json", "config.json",
                 "metrics.json", "manifest.json", "feature_names.txt", "background.npy"):
        assert (bundle / name).is_file(), f"missing bundle file: {name}"
    print("  all bundle files present")

    for device in (args.device, "auto", "cpu"):
        reloaded = load_model_bundle(bundle, device=device)
        p_again = reloaded.predict_proba(ds.test.X)
        identical = bool(np.allclose(p_cal, p_again, atol=1e-6))
        print(f"  reload(device={device!r}) identical={identical} "
              f"T={reloaded.temperature:.6f}")
        assert identical, "reloaded model must reproduce identical predictions"

    print("\n=== SMOKE TEST PASSED (real data, real values) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
