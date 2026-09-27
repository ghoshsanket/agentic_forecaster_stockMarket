# Reproducibility

## Quick start

```bash
# 1. Install
pip install -e '.[all]'

# 2. Download the dataset (Category B, outside Git)
python scripts/download_dataset.py

# 3. Reproduce the paper
python scripts/reproduce_paper.py --config configs/paper.yaml \
    --export-final-results

# 4. Validate the submission
python scripts/validate_submission.py
```

## What `reproduce_paper.py` does

1. **Data Agent** — loads the raw Kaggle dataset, engineers 17 features,
   constructs the next-day direction target, builds 30-day sequences, fits
   `StandardScaler` on train only.
2. **Model Agent** — trains all five model families (Attention-LSTM, plain
   LSTM, Random Forest, Logistic Regression, Majority).
3. **Explainer Agent** — computes SHAP values + attention evidence for the
   Attention-LSTM.
4. **Risk Agent** — converts conviction into a sized, risk-limited position.
5. **Report Agent** — renders per-ticker HTML/PDF reports.

Heavy runtime artefacts are written under `$AGENTIC_OUTPUT_ROOT` and
`$AGENTIC_MODEL_ROOT` (Category B).  With `--export-final-results` the
canonical final result set is exported into the repository (Category A).

## Training all 50 models

```bash
python scripts/train_all.py --config configs/paper.yaml
```

This trains one Attention-LSTM per ticker and writes checkpoints under
`$AGENTIC_MODEL_ROOT` (Category B).

## Demo / smoke test (no dataset required)

```bash
python scripts/reproduce_paper.py --config configs/demo.yaml
```

Uses a small synthetic dataset to verify the full pipeline end-to-end.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `AGENTIC_RAW_DATA_ROOT` | `$RESEARCH_ROOT/dataset/agentic-forecaster/raw` | Raw Kaggle data (Category B) |
| `AGENTIC_PROCESSED_DATA_ROOT` | `$RESEARCH_ROOT/dataset/agentic-forecaster/processed` | Processed sequences (Category B) |
| `AGENTIC_MODEL_ROOT` | `$RESEARCH_ROOT/models/agentic-forecaster` | Trained checkpoints (Category B) |
| `AGENTIC_OUTPUT_ROOT` | `$RESEARCH_ROOT/outputs/agentic-forecaster` | Run outputs (Category B) |
| `AGENTIC_REPO_RESULTS_ROOT` | `<repo>/results` | Exported results (Category A) |
| `AGENTIC_REPO_REPORTS_ROOT` | `<repo>/reports` | Exported reports (Category A) |
| `AGENTIC_REPO_FIGURES_ROOT` | `<repo>/figures` | Exported figures (Category A) |
| `AGENTIC_REPO_ARTIFACTS_ROOT` | `<repo>/artifacts` | Exported manifests (Category A) |

## Determinism

- Global seed 42 is applied to Python, NumPy and PyTorch.
- `torch.backends.cudnn.deterministic = True`.
- Synthetic fixtures use fixed seeds.

## CI

`.github/workflows/ci.yml` runs the full test suite on every push using
only synthetic fixtures — no dataset, no CUDA, no LLM credentials.
