# Paper-to-Code Traceability

**Paper:** Explanation-First Agentic Forecaster for Stock Market
**DOI:** 10.1109/IEMENTECH202669403.2026.11434302

## Classification legend

| Label | Meaning |
|---|---|
| **PAPER-DEFINED** | Explicitly present in the publication |
| **AUTHOR-CONFIRMED** | Confirmed by the authors, not verbatim in the publication |
| **RECONSTRUCTION-ASSUMED** | Chosen by the reconstruction team where the paper is silent |
| **NOT YET RECOVERED** | Known to be required by the paper, but the original artefact has not been found |

## The universe is two separate claims

The publication supports **one** statement about its stock universe:

* the study is framed around a **NIFTY-50 / 50-stock universe** — this is
  **PAPER-DEFINED**.

It does **not** support a claim about the **exact 50 constituent symbols**. The
constituent list is **RECONSTRUCTION-ASSUMED** and remains
**NOT YET RECOVERED**. A fixed list does exist in this repository
(`configs/nifty50.yaml`), but it is a reconstruction-team choice — its own
header says NIFTY-50 membership is time-varying and the publication states no
as-of date — and it is **not** the proven original list.

Two further constraints are documented and unresolved:

* The paper's representative predictions for 2023-07-05 name
  `RELIANCE, TCS, INFY, HDFCBANK, ITC`. `TCS` is absent from
  `configs/nifty50_legacy_user_supplied.yaml`, so the user-supplied legacy
  universe **cannot** be the paper's evaluation universe.
* Because the paper's dataset ends around **2025-11-04**, a snapshot taken near
  data-collection time is a more plausible reconstruction target than a
  2000-01-03 snapshot. `configs/nifty50_paper_snapshot_2025_11_04.yaml` is that
  candidate, built from official NSE Indices evidence. It is a
  **reconstruction hypothesis, not a proven fact.**

See **`docs/PAPER_UNIVERSE_RECOVERY.md`** for the full audit, the local
evidence sweep, and the candidate policies. Do not read any row below as
evidence that the exact original constituent list has been recovered.

## Traceability table

| Paper component | Label | Implementation | Test |
|---|---|---|---|
| NIFTY-50 / 50-stock framing (the *count* and the *index*) | PAPER-DEFINED | `data/universe.py` | `test_universe.py` |
| Exact 50 constituent symbols | RECONSTRUCTION-ASSUMED / **NOT YET RECOVERED** | `configs/nifty50.yaml` (reconstruction), `configs/nifty50_paper_snapshot_2025_11_04.yaml` (snapshot candidate) — **not** the proven original list; see `docs/PAPER_UNIVERSE_RECOVERY.md` | `test_universe.py` (count only) |
| Intraday → daily OHLCV | PAPER-DEFINED | `data/resampling.py:resample_intraday_to_daily` | `test_daily_resampling.py` |
| Next-day direction target | PAPER-DEFINED | `features/engineer.py:build_feature_frame` | `test_target_next_day.py` |
| `target_date` = next trading date | PAPER-DEFINED | `data/agent.py:run_ticker` | `test_target_date_metadata.py` |
| Split-boundary label safety | RECONSTRUCTION-ASSUMED | `data/agent.py:run_ticker` | `test_split_boundary_leakage.py` |
| 30-day sequence | AUTHOR-CONFIRMED | `data/agent.py:run_ticker` | `test_sequence_alignment.py` |
| One model per stock | AUTHOR-CONFIRMED | `agents/model_agent.py:train_ticker` | `test_pipeline_integration.py` |
| Per-stock StandardScaler | RECONSTRUCTION-ASSUMED | `data/agent.py:run_ticker` | `test_per_stock_scalers` |
| OHLCV in the model | PAPER-DEFINED | `features/engineer.py:build_feature_frame` | `test_features.py` |
| log return | PAPER-DEFINED | `features/engineer.py:log_return` | `test_features.py` |
| realised volatility (log returns) | PAPER-DEFINED | `features/engineer.py:realized_volatility` | `test_features.py` |
| RSI-14 | PAPER-DEFINED | `features/engineer.py:rsi` | `test_features.py` |
| MACD line/signal/histogram | PAPER-DEFINED | `features/engineer.py:macd` | `test_features.py` |
| ATR-14 | PAPER-DEFINED | `features/engineer.py:atr` | `test_features.py` |
| Attention formulation | PAPER-DEFINED | `models/attention_lstm.py` | `test_models.py` |
| `Linear(hidden,1)` + BCE | PAPER-DEFINED | `models/attention_lstm.py`, `training/trainer.py` | `test_models.py` |
| 3 epochs max | AUTHOR-CONFIRMED | `training/trainer.py` | `test_training_3_epochs.py` |
| Adam lr 1e-3, wd 1e-4 | PAPER-DEFINED | `training/trainer.py` | `test_training_3_epochs.py` |
| batch size 64 | PAPER-DEFINED | `configs/paper.yaml` | — |
| patience 10 | PAPER-DEFINED | `training/trainer.py` | `test_training_3_epochs.py` |
| gradient clipping | RECONSTRUCTION-ASSUMED | `training/trainer.py` | — |
| 2 layers / hidden 64 / dropout .2 | RECONSTRUCTION-ASSUMED | `models/attention_lstm.py` | — |
| Temperature calibration | RECONSTRUCTION-ASSUMED | `calibration/temperature.py` | `test_calibration_applied.py` |
| Calibration applied at inference | PAPER-DEFINED | `agents/model_agent.py:predict_proba` | `test_calibration_applied.py` |
| SHAP implementation | RECONSTRUCTION-ASSUMED | `explainability/shap_explainer.py` | `test_shap_nonzero.py` |
| Attention evidence | PAPER-DEFINED | `explainability/attention.py` | — |
| LLM explanation usage | AUTHOR-CONFIRMED | `agents/explainer_agent.py:_llm_narrate` (lazy import) | — |
| ATR risk equations | PAPER-DEFINED | `risk/risk_agent.py:decide` | `test_atr_risk.py` |
| Confidence multipliers | PAPER-DEFINED | `risk/risk_agent.py:decide` | `test_atr_risk.py` |
| RRR / RiskScore | PAPER-DEFINED | `risk/risk_agent.py:decide` | `test_atr_risk.py` |
| Walk-forward folds | PAPER-DEFINED | `orchestration/walk_forward.py:PAPER_FOLDS` | `test_walk_forward_folds.py` |
| Cross-sectional Precision@3 | PAPER-DEFINED | `evaluation/metrics.py:precision_at_3_cross_sectional` | `test_precision_at_3_cross_sectional.py` |
| Baselines executed | PAPER-DEFINED | `agents/model_agent.py:train_baselines` | `test_baseline_execution.py` |
| Ablations executed | RECONSTRUCTION-ASSUMED | `agents/ablation_agent.py` | `test_ablation_execution.py` |
| Aggregate metrics | RECONSTRUCTION-ASSUMED | `orchestration/walk_forward.py` | `test_reproduction_outputs.py` |
| ECE bin count | RECONSTRUCTION-ASSUMED | `evaluation/metrics.py` | `test_metrics.py` |
| Five-agent architecture | PAPER-DEFINED | `orchestration/pipeline.py` | `test_pipeline_integration.py` |
| HTML/PDF reports | PAPER-DEFINED | `agents/report_agent.py` | `test_report_agent.py` |

## Five-agent workflow

| Agent | Entry point |
|---|---|
| Data | `DataAgent.run_ticker(ticker)` |
| Model | `ModelAgent.train_ticker(ticker, dataset)` |
| Explainer | `ExplainerAgent.explain_prediction(...)` |
| Risk | `RiskAgent.decide(...)` |
| Report | `ReportAgent.run(...)` |
