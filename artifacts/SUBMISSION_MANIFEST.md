# Submission Manifest

## Paper

- **Title:** Explanation-First Agentic Forecaster for Stock Market
- **DOI:** 10.1109/IEMENTECH202669403.2026.11434302
- **URL:** https://doi.org/10.1109/IEMENTECH202669403.2026.11434302

## Repository

- **Root:** `projects/agentic-forecaster/`
- **Version:** 1.0.0
- **Commit:** (populated at submission time)
- **License:** MIT

## Code components

| Component | Location |
|---|---|
| Data Agent | `src/agentic_forecaster/data/agent.py` |
| Model Agent | `src/agentic_forecaster/agents/model_agent.py` |
| Explainer Agent | `src/agentic_forecaster/agents/explainer_agent.py` |
| Risk Agent | `src/agentic_forecaster/risk/risk_agent.py` |
| Report Agent | `src/agentic_forecaster/agents/report_agent.py` |
| Attention-LSTM | `src/agentic_forecaster/models/attention_lstm.py` |
| Plain LSTM | `src/agentic_forecaster/models/lstm.py` |
| Random Forest | `src/agentic_forecaster/models/baselines.py` |
| Logistic Regression | `src/agentic_forecaster/models/baselines.py` |
| Majority | `src/agentic_forecaster/models/baselines.py` |
| Feature engineering | `src/agentic_forecaster/features/engineer.py` |
| Training | `src/agentic_forecaster/training/trainer.py` |
| Calibration | `src/agentic_forecaster/calibration/temperature.py` |
| Explainability | `src/agentic_forecaster/explainability/` |
| Evaluation | `src/agentic_forecaster/evaluation/` |
| Orchestration | `src/agentic_forecaster/orchestration/pipeline.py` |

## Configurations

- `configs/paper.yaml` — full paper reproduction
- `configs/demo.yaml` — synthetic smoke test
- `configs/nifty50.yaml` — ticker universe

## Dataset

- **Source URL:** https://www.kaggle.com/datasets/debashis74017/algo-trading-data-nifty-100-data-with-indicators
- **Slug:** `debashis74017/algo-trading-data-nifty-100-data-with-indicators`
- **Exclusion reason:** very large; would make the repository impractically large
- **Hashes from author's run:** see `results/paper_reproduction/dataset_hashes.json`
- **Download:** `python scripts/download_dataset.py`

## Final metrics files

- `results/paper_reproduction/aggregate_metrics.json`
- `results/paper_reproduction/ticker_metrics.csv`
- `results/paper_reproduction/paper_comparison.csv`

## Final plots

- `figures/reliability_diagram.png`
- `figures/training_history.png`
- `figures/confusion_matrix.png`
- `figures/paper_comparison.png`
- `figures/attention_visualization.png`
- `figures/shap_summary.png`

## Representative reports

- `reports/examples/RELIANCE_2024-12-31.html`
- `reports/examples/RELIANCE_2024-12-31.pdf`

## Model artifacts

- **Included:** no (checkpoints are runtime-only due to size policy)
- **Git LFS:** configured in `.gitattributes` but no LFS objects committed
- **Checkpoint hashes:** `artifacts/manifests/models/runtime_checkpoint_index.json`
- **Recreation:** `python scripts/train_all.py --config configs/paper.yaml`

## Environment / reproduction

```bash
pip install -e '.[all]'
python scripts/download_dataset.py
python scripts/reproduce_paper.py --config configs/paper.yaml --export-final-results
python scripts/validate_submission.py
```
