# Results

This directory stores all experimental outputs from the agentic-forecaster
reconstruction.

## Directory structure

```
results/
├── README.md                          # this file
├── paper_reproduction/                # main reproduction results
│   ├── aggregate_metrics.json         # mean metrics across all 50 tickers
│   ├── ticker_metrics.csv            # per-ticker metrics
│   ├── paper_comparison.csv           # paper reference vs reconstructed
│   ├── paper_reference.json           # values transcribed from the paper
│   ├── reconstructed_run.json        # values produced by this implementation
│   └── dataset_hashes.json            # SHA-256 of raw CSVs used
├── metrics/                           # full per-fold metric dumps
├── predictions/                       # representative prediction samples
├── baselines/                         # baseline model metrics
└── ablations/                         # ablation study results
```

## Paper reference vs reconstructed

- **Paper reference** (`paper_reference.json`): values transcribed from the
  publication. These are **not** presented as newly produced experimental
  results.
- **Reconstructed** (`reconstructed_run.json`): actual outputs of this
  implementation on the Kaggle dataset.
- **Comparison** (`paper_comparison.csv`): side-by-side join of the two.

## Regenerating results

```bash
python scripts/reproduce_paper.py --config configs/paper.yaml \
    --export-final-results
```

## Interpretation

Differences between reference and reconstructed values arise from:

- Dataset version and preprocessing choices
- Library versions (PyTorch, scikit-learn, SHAP)
- Random seeds and initialization
- Reconstruction assumptions (see `docs/IMPLEMENTATION_ASSUMPTIONS.md`)

Neither reference nor reproduced values constitute investment advice.
