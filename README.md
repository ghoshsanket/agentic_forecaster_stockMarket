# Explanation-First Agentic Forecaster for Stock Market

Reference implementation of *Explanation-First Agentic Forecaster for Stock
Market* — [DOI: 10.1109/IEMENTECH202669403.2026.11434302](https://doi.org/10.1109/IEMENTECH202669403.2026.11434302).

A five-agent pipeline that forecasts next-day price direction for the NIFTY-50
universe and explains *why* each forecast was made before any capital is
committed.

## Agents

1. **Data Agent** — loads daily OHLCV for a ticker, engineers the Phase-1
   feature set (OHLCV + log_return + realized_volatility_20 + RSI-14 + MACD +
   ATR-14), builds the next-day direction target and 30-day sequences, and fits
   one `StandardScaler` per ticker per fold on TRAIN rows only.
2. **Model Agent** — trains **one independent Attention-LSTM per stock**
   (binary logit + `BCEWithLogitsLoss`, early stopping, gradient clipping).
3. **Explainer Agent** — per-prediction SHAP (GradientExplainer with
   permutation fallback) + attention evidence + reason codes; optional LLM
   narration with a deterministic fallback.
4. **Risk Agent** — ATR-based stop-loss / take-profit from confidence bands
   (HIGH/MEDIUM/LOW), plus RRR and RiskScore.
5. **Report Agent** — renders per-ticker HTML and PDF reports: forecast,
   indicators, SL/TP prices, SHAP, attention, reason codes and narrative.

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    ┌──────────┐    ┌─────────────┐
│  Data Agent │ -> │ Model Agent │ -> │ Explainer Agent │ -> │Risk Agent│ -> │Report Agent │
└─────────────┘    └─────────────┘    └─────────────────┘    └──────────┘    └─────────────┘
```

One model per stock: the orchestration layer calls
`DataAgent.run_ticker(ticker)` and `ModelAgent.train_ticker(ticker, dataset)`
once per ticker, so models, scalers and calibrators are never shared.

## Repository Layout

```
agentic-forecaster/
├── README.md
├── pyproject.toml               # package metadata + dependency groups
├── uv.lock                      # locked environment (validated in CI)
├── configs/                     # nifty50.yaml, demo.yaml, paper.yaml,
│                                # reproduction_search/ (yfinance variants)
├── src/agentic_forecaster/      # implementation
│   ├── agents/  data/  features/  models/  training/
│   ├── calibration/  explainability/  risk/  evaluation/
│   ├── orchestration/  recovery/  commands/
│   └── cli.py                   # command-line entry point
├── app/streamlit_app.py         # interactive demo (DEMO + REAL modes)
├── scripts/                     # dataset download, training, validation
├── tests/                       # unit + integration + fixtures
├── docs/                        # architecture, traceability, assumptions
└── .github/workflows/ci.yml     # ruff + pytest + smoke train
```

This repository contains **code only**. Datasets, model checkpoints and run
outputs are stored outside Git — see [data/README.md](data/README.md).

## Dataset

Data is downloaded from **Yahoo Finance** through the `yfinance` package.
No API key or account is required.

| | |
|---|---|
| **Universe** | 50 NIFTY-50 symbols (`configs/nifty50.yaml`), requested as `<SYMBOL>.NS` |
| **Bars** | Daily (`interval=1d`), 2000-01-01 → 2025-12-31 (`end` is exclusive) |
| **Columns** | `Date,Open,High,Low,Close,Volume` |
| **Variants** | `adjusted` (primary, `auto_adjust=True`), `unadjusted` (sensitivity) |
| **Location** | `$AGENTIC_YFINANCE_DAILY_ROOT` — **not** committed to Git |

```bash
# 1. Download (resumable; skips tickers that already validate)
uv run python scripts/download_yfinance_daily.py \
    --start 2000-01-01 --end 2026-01-01 --variant both --resume --all

# 2. Build metadata, coverage and SHA-256 manifests
uv run python scripts/build_yfinance_artifacts.py
```

Full layout, request parameters and data-quality notes:
[docs/YFINANCE_DAILY_DATASET.md](docs/YFINANCE_DAILY_DATASET.md) and
[data/README.md](data/README.md).

## Installation

Python **3.11**, managed with [uv](https://docs.astral.sh/uv/) and the
committed `uv.lock`:

```bash
uv sync --all-extras --frozen

uv run python -m agentic_forecaster --help
```

CI installs the same way, so the lockfile — not a fresh resolve — is what gets
validated.

## Configuration

| Config | Purpose |
|---|---|
| `configs/reproduction_search/yfinance_adjusted.yaml` | Daily Yahoo Finance data, adjusted — used in the examples below |
| `configs/reproduction_search/yfinance_unadjusted.yaml` | Same pipeline, unadjusted prices |
| `configs/nifty50.yaml` | 50-symbol universe |
| `configs/demo.yaml` | Synthetic end-to-end demo; needs no dataset or GPU |
| `configs/paper.yaml` | Original paper setup — 1-minute bars resampled to daily; needs its own raw dataset (`docs/DATASET_PROVENANCE.md`) |

`${AGENTIC_*}` placeholders inside the YAML files are expanded from the
environment at load time; every one has a workspace default, so no exports are
required.

## Data Preparation

Builds the cached daily series, feature frames and targets for one ticker:

```bash
uv run python -m agentic_forecaster prepare-data \
    --config configs/reproduction_search/yfinance_adjusted.yaml \
    --ticker RELIANCE
```

## Training

One independent model, scaler and calibrator per stock:

```bash
# Single ticker
uv run python -m agentic_forecaster train \
    --ticker RELIANCE --config configs/reproduction_search/yfinance_adjusted.yaml --device auto

# Every configured ticker, independently
uv run python -m agentic_forecaster train-all \
    --config configs/reproduction_search/yfinance_adjusted.yaml --device auto --baselines

# Equivalent script form
uv run python scripts/train_all.py --config configs/reproduction_search/yfinance_adjusted.yaml
```

Checkpoints are written to `$AGENTIC_MODEL_ROOT/trained/<ticker>/fold_0/`
(outside Git).

## Evaluation

```bash
# Evaluate a saved bundle
uv run python -m agentic_forecaster evaluate \
    --model <bundle_dir> --data <processed_dir>

# Walk-forward: both published folds, retraining every stock
uv run python -m agentic_forecaster walk-forward \
    --config configs/reproduction_search/yfinance_adjusted.yaml --device auto
```

Computed metrics: accuracy, Brier (raw + calibrated), ECE (raw + calibrated),
Precision@3 (cross-sectional UP/DOWN), F1 and ROC-AUC. They are written to
`$AGENTIC_OUTPUT_ROOT`, not to the repository.

## Explanations and Reports

```bash
# One prediction with SHAP + attention evidence
uv run python -m agentic_forecaster explain \
    --model <bundle_dir> --data <processed_dir> --index -1

# Per-ticker HTML/PDF report
uv run python -m agentic_forecaster report \
    --config configs/reproduction_search/yfinance_adjusted.yaml --ticker RELIANCE

# End-to-end smoke test: real data, SHAP, report, bundle reload
uv run python scripts/smoke_test_real.py --ticker RELIANCE \
    --config configs/reproduction_search/yfinance_adjusted.yaml

# Full pipeline run (same implementation as scripts/reproduce_paper.py)
uv run python -m agentic_forecaster reproduce-paper \
    --config configs/reproduction_search/yfinance_adjusted.yaml --device auto
```

## Interactive Demo

```bash
uv run streamlit run app/streamlit_app.py                 # DEMO (synthetic) mode
uv run streamlit run app/streamlit_app.py -- --mode real  # REAL mode (dataset + checkpoints)
```

## Tests and Lint

```bash
uv run ruff check .
uv run pytest -v

# CI smoke train on synthetic data — no dataset, GPU or LLM required
uv run python -m agentic_forecaster train --ticker SYN00 --config configs/demo.yaml --device cpu
```

One network test is opt-in: `AGENTIC_RUN_NETWORK_TESTS=1 uv run pytest tests/unit/test_yfinance_download.py`.

## Documentation

- `docs/ARCHITECTURE.md` — pipeline and model architecture
- `docs/PAPER_TRACEABILITY.md` — paper section → code mapping
- `docs/IMPLEMENTATION_ASSUMPTIONS.md` — every reconstruction assumption
- `docs/YFINANCE_DAILY_DATASET.md` — dataset construction and validation
- `docs/REPRODUCIBILITY.md` — how to rerun the full pipeline
- `docs/STORAGE_LAYOUT.md` — what lives in Git vs outside it

## Disclaimer

Research software only. It does not constitute investment advice. Past
performance does not guarantee future results.
