# Figures

| Figure | Provenance | Source run |
|---|---|---|
| `reliability_diagram.png` | real data | 3-ticker real-data integration smoke run (`smoke_wf`, RELIANCE/TCS/INFY, both folds) |
| `confusion_matrix.png` | real data | same |
| `paper_comparison.png` | real data + paper reference | same |
| `precision_at_3.png` | real data | same |
| `attention_visualization.png` | real data | single RELIANCE prediction (`scripts/smoke_test_real.py`) |
| `shap_summary.png` | real data | single RELIANCE prediction |
| `training_history.png` | real data | RELIANCE fold_0, 3 epochs |

**These are pre-full-run artefacts.** They were produced by small real-data
integration runs (3 tickers / 1 ticker), not by the full NIFTY-50
reproduction. They demonstrate that figure generation works on real run
artefacts; they are not the final paper figures.

When the full reproduction runs, `--export-final-results` overwrites this
directory with figures from the complete run directory:

```bash
uv run python -m agentic_forecaster reproduce-paper \
    --config configs/paper.yaml --device auto --export-final-results
```

Each run also writes a `figures_manifest.json` recording the run id that
produced its figures, so provenance is never lost on overwrite.
