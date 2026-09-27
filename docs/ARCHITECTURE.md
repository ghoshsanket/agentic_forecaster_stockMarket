# Architecture

## Overview

A **five-agent pipeline** producing an explanation-first next-day direction
forecast for 50 NIFTY-50 stocks, with **one independent model per stock**.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    ┌──────────┐    ┌─────────────┐
│  Data Agent │ -> │ Model Agent │ -> │ Explainer Agent │ -> │Risk Agent│ -> │Report Agent │
└─────────────┘    └─────────────┘    └─────────────────┘    └──────────┘    └─────────────┘
```

Runtime policy: **Python 3.11**, **PyTorch**.

## Agents

| Agent | Module | Responsibility |
|---|---|---|
| **Data Agent** | `data/agent.py` | Discover raw 1-minute CSVs, resolve the NIFTY-50 universe, resample intraday → **daily** OHLCV (cached, source-hash validated), build the Phase-1 feature set, construct the next-day target and `target_date`, build 30-day sequences whose window **ends on the prediction-origin day**, and fit **one StandardScaler per stock per fold on training rows only**. |
| **Model Agent** | `agents/model_agent.py` | Train one Attention-LSTM per stock (binary logit, `BCEWithLogitsLoss`, max 3 epochs, gradient clipping) plus the four baselines, and fit temperature calibration on that stock's validation split. |
| **Explainer Agent** | `agents/explainer_agent.py` | Per-prediction SHAP (`GradientExplainer` over a binary-logit wrapper, Integrated Gradients fallback) + attention evidence + reason codes + narrative. |
| **Risk Agent** | `risk/risk_agent.py` | **ATR-based** stop-loss / take-profit from confidence bands, plus RRR and RiskScore. **No position sizing.** |
| **Report Agent** | `agents/report_agent.py` | Per-stock HTML + PDF reports. This is the **only** reporting system. |

## Model architecture

### Attention-LSTM (primary)

```
Input (B, 30, 12)                       # OHLCV + log_return + vol20 + RSI + MACD x3 + ATR
  -> LSTM(hidden=64, layers=2, dropout=0.2)
  -> additive (Bahdanau-style) attention over the 30 time steps
  -> dropout
  -> Linear(64, 1)                      # single binary logit
  -> p_up = sigmoid(logit)
```

Attention weights are returned with the logit so the Explainer Agent can
surface *which days* the model attended to.

### Baselines

Plain LSTM (same encoder, no attention), Random Forest, Logistic Regression,
and Majority Class — all on the same temporal split and the same per-stock
scaler.

## Calibration

Temperature scaling fits a single scalar `T` on validation logits by
minimising binary NLL. A scalar `T` preserves ranking (accuracy and
Precision@3 are unchanged) while changing Brier and ECE. Both raw and
calibrated probabilities are persisted and reported.

## SHAP compatibility

The model emits a 1-D logit; SHAP's gradient explainer expects a 2-D output.
`BinaryLogitWrapper` re-shapes the logit to `(batch, 1)` without changing its
value, so attribution semantics are preserved exactly. The background sample
is drawn from **training** sequences only.

## Storage categories

| Category | Location | In Git? |
|---|---|---|
| **A — Repository** | `src/`, `configs/`, `docs/`, `results/`, `figures/`, `reports/`, `artifacts/`, `scripts/`, `tests/` | Yes |
| **B — Runtime** | `$AGENTIC_RAW_DATA_ROOT`, `$AGENTIC_PROCESSED_DATA_ROOT`, `$AGENTIC_MODEL_ROOT`, `$AGENTIC_OUTPUT_ROOT` | No |
| **C — Exported** | Final artefacts copied B → A by `package-submission` / `reproduce-paper --export-final-results` | Yes |

See `docs/STORAGE_LAYOUT.md`.

## Reproduction output

`run_walk_forward` writes, under `$AGENTIC_OUTPUT_ROOT/reproduction/<run_id>/`:
`manifest.json`, `ticker_metrics.csv`, `aggregate_metrics.json`,
`predictions.csv.gz`, `calibration_metrics.csv`, `precision_at_3.csv`,
`p3_daily_selections.csv`, `baseline_metrics.csv`, `ablation_metrics.csv`,
`paper_comparison.csv`, `training_summary.json`, `figures/`, `reports/`.

## Technology stack

Python 3.11, PyTorch, scikit-learn, pandas, NumPy, SHAP (optional),
reportlab (PDF), Streamlit (app), uv (lockfile-managed environment).
