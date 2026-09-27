# Paper-to-Code Traceability

**Paper:** Explanation-First Agentic Forecaster for Stock Market
**DOI:** 10.1109/IEMENTECH202669403.2026.11434302

## Classification legend

| Label | Meaning |
|---|---|
| PAPER-DEFINED | Explicitly stated in the paper |
| RECONSTRUCTION-ASSUMED | Chosen by the reconstruction team where the paper is ambiguous |

## Equation-to-code map

| Paper component | Classification | Implementation | Test |
|---|---|---|---|
| 50 NIFTY-50 stocks | PAPER-DEFINED | `configs/nifty50.yaml` | `test_ticker_universe` |
| Intraday → daily resampling | PAPER-DEFINED | `data/resampling.py:resample_intraday_to_daily` | `test_daily_resampling` |
| OHLCV input | PAPER-DEFINED | `features/engineer.py:build_feature_frame` | `tests/unit/test_features.py` |
| log_return | PAPER-DEFINED | `features/engineer.py:log_return` | `tests/unit/test_features.py` |
| realized_volatility_20 | PAPER-DEFINED | `features/engineer.py:realized_volatility` | `tests/unit/test_features.py` |
| RSI-14 | PAPER-DEFINED | `features/engineer.py:rsi` | `tests/unit/test_features.py` |
| MACD | PAPER-DEFINED | `features/engineer.py:macd` | `tests/unit/test_features.py` |
| ATR-14 | PAPER-DEFINED | `features/engineer.py:atr` | `tests/unit/test_features.py` |
| Next-day target | PAPER-DEFINED | `features/engineer.py` target construction | `test_target_next_day` |
| 30-day lookback | PAPER-DEFINED | `data/agent.py:run_ticker` | `test_sequence_alignment` |
| One-model-per-stock | PAPER-DEFINED | `agents/model_agent.py:train_ticker` | `test_one_model_per_stock` |
| Per-stock scaler | PAPER-DEFINED | `data/agent.py:run_ticker` | `test_per_stock_scalers` |
| Attention mechanism | RECONSTRUCTION-ASSUMED | `models/attention_lstm.py` | `tests/unit/test_models.py` |
| 2 layers, hidden 64, dropout .2 | RECONSTRUCTION-ASSUMED | `models/attention_lstm.py` | — |
| Binary logit + BCE | PAPER-DEFINED | `models/attention_lstm.py`, `training/trainer.py` | `tests/unit/test_models.py` |
| 3 epochs max | PAPER-DEFINED | `training/trainer.py` | `test_training_3_epochs` |
| Gradient clipping | RECONSTRUCTION-ASSUMED | `training/trainer.py` | — |
| Temperature calibration | RECONSTRUCTION-ASSUMED | `calibration/temperature.py` | `tests/unit/test_calibration.py` |
| SHAP (GradientExplainer) | RECONSTRUCTION-ASSUMED | `explainability/shap_explainer.py` | — |
| Attention evidence | PAPER-DEFINED | `explainability/attention.py` | — |
| ATR risk (SL/TP/RRR/RiskScore) | PAPER-DEFINED | `risk/risk_agent.py` | `test_atr_risk` |
| Cross-sectional P@3 | PAPER-DEFINED | `evaluation/metrics.py:precision_at_3_cross_sectional` | `test_precision_at_3_cross_sectional` |
| Brier / ECE / F1 / Accuracy | PAPER-DEFINED | `evaluation/metrics.py` | `tests/unit/test_metrics.py` |
| Paper walk-forward (2 folds) | PAPER-DEFINED | `orchestration/walk_forward.py` | — |
| HTML/PDF reports | PAPER-DEFINED | `agents/report_agent.py` | — |
| Five-agent workflow | PAPER-DEFINED | `orchestration/pipeline.py` | `tests/integration/test_pipeline.py` |

## Five-agent workflow

| Paper agent | Entry point |
|---|---|
| Data Agent | `DataAgent.run_ticker(ticker)` |
| Model Agent | `ModelAgent.train_ticker(ticker, dataset)` |
| Explainer Agent | `ExplainerAgent.explain_prediction(...)` |
| Risk Agent | `RiskAgent.decide(...)` |
| Report Agent | `ReportAgent.run(...)` |
