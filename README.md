# Explanation-First Agentic Forecaster for Stock Market

> **A software reconstruction / reference implementation of**
> *Explanation-First Agentic Forecaster for Stock Market*,
> **DOI: 10.1109/IEMENTECH202669403.2026.11434302**

## Overview

This repository is a complete software reconstruction of the paper
*Explanation-First Agentic Forecaster for Stock Market*. It implements a
**five-agent pipeline** that forecasts next-day price direction for 50 NIFTY-50
stocks and explains *why* each forecast was made, before any capital is
committed.

The five agents are:

1. **Data Agent** — locates the dataset, resamples intraday (1-minute) OHLCV
   to daily, engineers the Phase-1 feature set (OHLCV + log_return +
   realized_volatility_20 + RSI-14 + MACD + ATR-14), constructs the next-day
   direction target, builds 30-day sequences, and fits one `StandardScaler`
   per ticker per fold on TRAIN rows only.
2. **Model Agent** — trains **one independent Attention-LSTM per stock**
   (max 3 epochs, binary logit + BCEWithLogitsLoss, gradient clipping).
3. **Explainer Agent** — per-prediction SHAP (GradientExplainer with
   permutation fallback) + attention evidence + reason codes; optional LLM
   narration with grounding instructions and deterministic fallback.
4. **Risk Agent** — ATR-based stop-loss / take-profit using confidence bands
   (HIGH/MEDIUM/LOW) with lambda_SL/lambda_TP multipliers; RRR and RiskScore.
5. **Report Agent** — renders per-ticker **HTML** and **PDF** reports with
   forecast, indicators, SL/TP prices, RRR, RiskScore, SHAP, attention,
   reason codes, narrative, and reconstruction disclaimer.

## Paper

- **Title:** Explanation-First Agentic Forecaster for Stock Market
- **DOI:** [10.1109/IEMENTECH202669403.2026.11434302](https://doi.org/10.1109/IEMENTECH202669403.2026.11434302)

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐    ┌──────────┐    ┌─────────────┐
│  Data Agent │ -> │ Model Agent │ -> │ Explainer Agent │ -> │Risk Agent│ -> │Report Agent │
└─────────────┘    └─────────────┘    └─────────────────┘    └──────────┘    └─────────────┘
```

One-model-per-stock: the orchestration layer calls `DataAgent.run_ticker(ticker)`
and `ModelAgent.train_ticker(ticker, dataset)` once per ticker, guaranteeing
independent models and scalers.

## Repository Layout

```
agentic-forecaster/
├── README.md
├── pyproject.toml
├── .gitignore
├── .gitattributes
├── configs/                    # paper.yaml, demo.yaml, nifty50.yaml
├── src/agentic_forecaster/     # full implementation
├── app/streamlit_app.py        # Streamlit demo (DEMO + REAL modes)
├── scripts/                    # download, train, reproduce, smoke_test, validate
├── tests/                      # unit + integration + fixtures
├── docs/                       # architecture, traceability, assumptions, ...
├── results/                    # final metrics, predictions, baselines, ablations
├── figures/                    # reliability, training history, comparison, ...
├── reports/                    # representative example reports
├── artifacts/                  # manifests + submission manifest
└── .github/workflows/ci.yml
```

## Dataset

> **Raw dataset is not stored in Git due to size.**

- **Kaggle URL:** https://www.kaggle.com/datasets/debashis74017/algo-trading-data-nifty-100-data-with-indicators
- **Slug:** `debashis74017/algo-trading-data-nifty-100-data-with-indicators`
- **Content:** 1-minute OHLCV for NIFTY-50 constituents (2015-2026)

The dataset lives **outside** the repository under `$AGENTIC_RAW_DATA_ROOT`.

### How to authenticate

1. Create a Kaggle account and generate an API token at
   https://www.kaggle.com/settings/api.
2. Either place `kaggle.json` in `~/.kaggle/` or set `KAGGLE_USERNAME` and
   `KAGGLE_KEY` in the environment.

**Never commit credentials.** See `docs/DATASET_PROVENANCE.md`.

### How to download

```bash
python scripts/download_dataset.py
```

### How to inspect

```bash
python scripts/inspect_raw_dataset.py
```

## Installation

```bash
pip install -e '.[all]'
```

## Training

```bash
# Train a single stock
python -m agentic_forecaster train --ticker RELIANCE --config configs/paper.yaml

# Train all 50 stocks independently
python -m agentic_forecaster train-all --config configs/paper.yaml

# Or use the script directly
python scripts/train_all.py --config configs/paper.yaml
```

Checkpoints are written under `$AGENTIC_MODEL_ROOT/trained/<ticker>/fold_0/`.

## Evaluation

```bash
# Evaluate a trained model
python -m agentic_forecaster evaluate --model <bundle_dir> --data <processed_dir>

# Run paper walk-forward (2 folds, retrain per ticker)
python -m agentic_forecaster walk-forward --config configs/paper.yaml
```

Metrics: accuracy, Brier (raw + calibrated), ECE (raw + calibrated),
Precision@3 (cross-sectional UP/DOWN), F1, ROC-AUC.

## Reproduce the Paper

```bash
# 1. Download the dataset
python scripts/download_dataset.py

# 2. Run the full paper reproduction
python -m agentic_forecaster reproduce-paper --config configs/paper.yaml \
    --export-final-results

# 3. Validate the submission
python scripts/validate_submission.py
```

A quick smoke test on one real ticker:

```bash
python scripts/smoke_test_real.py --ticker RELIANCE
```

## Results

| Model | Accuracy | Brier | ECE | P@3 Up | P@3 Down |
|---|---|---|---|---|---|
| Attention-LSTM (calibrated) | see `results/paper_reproduction/` | | | | |
| Attention-LSTM (raw) | | | | | |
| Plain LSTM | | | | | |
| Random Forest | | | | | |

> Paper-reference values are transcribed from the publication and are
> **not** presented as newly produced results. See `results/README.md` and
> `docs/RESULTS.md`.

## Example Reports

Representative reports are committed under `reports/examples/`.
See `reports/README.md` for how to generate more.

## Models / Checkpoints

- **Code** defining all models is committed under `src/agentic_forecaster/models/`.
- **Checkpoints** are runtime-only (Category B) under `$AGENTIC_MODEL_ROOT`.
- **Manifests** with SHA-256 hashes are committed under `artifacts/manifests/`.
- **Git LFS** is configured in `.gitattributes`.
- **Recreate checkpoints:** `python -m agentic_forecaster train-all --config configs/paper.yaml`

## Paper-to-Code Traceability

See `docs/PAPER_TRACEABILITY.md` for a section-by-section mapping of the paper
to the implementing code, with PAPER-DEFINED vs RECONSTRUCTION-ASSUMED
classifications.

## Reconstruction Assumptions

See `docs/IMPLEMENTATION_ASSUMPTIONS.md` for every assumption made when
translating the paper into code.

## Limitations

- The exact ticker list and some hyperparameters are reconstructed from the
  paper's description; where the paper is ambiguous, sensible defaults are
  used and documented.
- The LLM narration layer requires an OpenAI-compatible endpoint; the
  deterministic fallback is the default and the only mode exercised in CI.
- SHAP's GradientExplainer may not support the attention mechanism directly;
  a permutation-based fallback is used and explicitly reported.
- Results on synthetic data are near-chance by design and are **not** paper
  reconstructions.

## Research Disclaimer

This is a software reconstruction for research purposes. It does not
constitute investment advice. Past performance does not guarantee future
results.
