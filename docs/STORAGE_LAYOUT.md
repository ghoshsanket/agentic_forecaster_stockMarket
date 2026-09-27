# Storage Layout

The project uses a three-category storage policy.

## Category A — Git repository (`projects/agentic-forecaster/`)

```
agentic-forecaster/
├── README.md
├── pyproject.toml
├── .gitignore
├── .gitattributes              # Git LFS tracking for *.pt, *.joblib, ...
├── configs/
│   ├── paper.yaml
│   ├── demo.yaml
│   └── nifty50.yaml
├── src/agentic_forecaster/
│   ├── agents/                 # Model, Explainer, Report agents
│   ├── data/                   # Data agent, dataset, downloader
│   ├── features/               # RSI, MACD, ATR, volatility, ...
│   ├── models/                 # Attention-LSTM, LSTM, RF, LR, Majority
│   ├── training/               # Trainer with early stopping
│   ├── calibration/            # Temperature scaling
│   ├── explainability/         # SHAP + attention evidence
│   ├── risk/                   # Risk agent
│   ├── evaluation/             # Metrics + walk-forward
│   ├── reporting/              # (report agent lives in agents/)
│   ├── orchestration/          # Pipeline
│   ├── packaging.py            # Category B -> A export
│   └── cli.py
├── app/streamlit_app.py
├── scripts/
│   ├── download_dataset.py
│   ├── inspect_raw_dataset.py
│   ├── train_all.py
│   ├── reproduce_paper.py
│   ├── package_submission.py
│   └── validate_submission.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── PAPER_TRACEABILITY.md
│   ├── IMPLEMENTATION_ASSUMPTIONS.md
│   ├── DATASET_PROVENANCE.md
│   ├── REPRODUCIBILITY.md
│   ├── STORAGE_LAYOUT.md
│   └── RESULTS.md
├── results/
│   ├── README.md
│   ├── paper_reproduction/
│   │   ├── aggregate_metrics.json
│   │   ├── ticker_metrics.csv
│   │   ├── paper_comparison.csv
│   │   └── dataset_hashes.json
│   ├── metrics/
│   ├── predictions/
│   ├── baselines/
│   └── ablations/
├── figures/
│   ├── reliability_diagram.png
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── paper_comparison.png
│   ├── attention_visualization.png
│   └── shap_summary.png
├── reports/
│   ├── README.md
│   ├── examples/
│   │   ├── RELIANCE_example.html
│   │   └── RELIANCE_example.pdf
│   └── html/
├── artifacts/
│   ├── README.md
│   ├── SUBMISSION_MANIFEST.md
│   ├── manifests/
│   │   └── models/
│   │       └── runtime_checkpoint_index.json
│   └── models/                 # Git LFS (if checkpoints are small enough)
└── .github/workflows/ci.yml
```

## Category B — Runtime (outside Git)

```
$RESEARCH_ROOT/
├── dataset/agentic-forecaster/
│   ├── raw/                    # Kaggle download (NOT in Git)
│   └── processed/              # Sequences + scaler (NOT in Git)
├── models/agentic-forecaster/  # Trained checkpoints (NOT in Git)
├── outputs/agentic-forecaster/ # Run outputs (NOT in Git)
├── cache/                      # pip, torch, HF, ... (NOT in Git)
├── environments/               # (NOT in Git)
└── secrets/                    # Credentials (NEVER in Git)
```

## Category C — Exported final artefacts

After a successful run, `scripts/package_submission.py` copies the canonical
final artefacts from Category B into Category A:

| From (B) | To (A) |
|---|---|
| `$AGENTIC_OUTPUT_ROOT/metrics.json` | `results/metrics.json` |
| `$AGENTIC_OUTPUT_ROOT/reports/*.html` | `reports/html/` |
| `$AGENTIC_OUTPUT_ROOT/figures/*.png` | `figures/` |
| `$AGENTIC_MODEL_ROOT/**/*.pt` | `artifacts/manifests/models/` (metadata + hashes) |

The exporter **never** copies the raw dataset, secrets, caches, environments
or temporary files, and prints exactly what was copied.
