# Architecture

## Overview

The system is a **five-agent pipeline** that produces an explanation-first
forecast for each NIFTY 100 constituent.  Each agent is a separate,
independently testable component.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    ┌──────────┐    ┌─────────────┐
│  Data Agent │ -> │ Model Agent │ -> │ Explainer Agent │ -> │Risk Agent│ -> │Report Agent │
└─────────────┘    └─────────────┘    └─────────────────┘    └──────────┘    └─────────────┘
```

## Agents

| Agent | Module | Responsibility |
|---|---|---|
| **Data Agent** | `src/agentic_forecaster/data/agent.py` | Locate/download raw data, engineer features (RSI, MACD, ATR, volatility, ...), construct the 30-day target, build sequences, fit `StandardScaler` on train only. |
| **Model Agent** | `src/agentic_forecaster/agents/model_agent.py` | Train all five model families: Attention-LSTM, plain LSTM, Random Forest, Logistic Regression, Majority. |
| **Explainer Agent** | `src/agentic_forecaster/agents/explainer_agent.py` | SHAP feature attribution + attention evidence; LLM narration with deterministic fallback. |
| **Risk Agent** | `src/agentic_forecaster/risk/risk_agent.py` | Half-Kelly position sizing, volatility targeting, stop-loss / take-profit. |
| **Report Agent** | `src/agentic_forecaster/agents/report_agent.py` | Render per-ticker HTML/PDF reports combining forecast, explanation, risk and metrics. |

## Orchestration

`src/agentic_forecaster/orchestration/pipeline.py` wires the five agents
together.  The `Pipeline.run()` method executes them in order and returns a
`PipelineResult` containing the dataset, fitted models, explanation bundle,
metrics and generated reports.

## Storage categories

| Category | Location | In Git? |
|---|---|---|
| **A — Repository** | `src/`, `configs/`, `docs/`, `results/`, `figures/`, `reports/`, `artifacts/`, `scripts/`, `tests/` | Yes |
| **B — Runtime** | `$AGENTIC_RAW_DATA_ROOT`, `$AGENTIC_PROCESSED_DATA_ROOT`, `$AGENTIC_MODEL_ROOT`, `$AGENTIC_OUTPUT_ROOT` | No |
| **C — Exported** | Final artefacts copied from B into A by `scripts/package_submission.py` | Yes (after export) |

See `docs/STORAGE_LAYOUT.md` for the full directory map.

## Model architecture

### Attention-LSTM (primary)

```
Input (B, 30, F)
  -> LSTM(hidden=64, layers=2, dropout=0.2)
  -> additive attention over 30 time steps
  -> dropout
  -> Linear(64, 2)
```

Attention weights are returned alongside logits so the Explainer Agent can
surface *which days* the model attended to.

### Baselines

- **Plain LSTM**: same encoder, no attention, last hidden state.
- **Random Forest**: 300 trees, max_depth 8, flattened (30 x F) input.
- **Logistic Regression**: C=1.0, flattened input.
- **Majority**: predicts the training majority class.

## Calibration

Temperature scaling fits a single scalar `T` on validation logits by
minimising NLL.  A scalar `T` preserves ranking (accuracy, Precision@3
unchanged) while improving Brier and ECE.

## Technology stack

- Python 3.10+, PyTorch (torch), scikit-learn, pandas, NumPy
- SHAP (optional, with permutation fallback)
- Streamlit (demo app)
- reportlab (PDF reports)
- kaggle (dataset download)
