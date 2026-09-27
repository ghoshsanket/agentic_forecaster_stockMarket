# Baselines

Per-ticker baseline metrics (Plain LSTM, Random Forest, Logistic Regression,
Majority Class) for the full paper reproduction.

Populated by:

```bash
uv run python -m agentic_forecaster reproduce-paper \
    --config configs/paper.yaml --device auto --export-final-results
```

`baseline_metrics.csv` is written here by that export. Before the full run
completes this directory intentionally contains only this README; the
1-ticker real-data smoke-test evidence lives in
`../smoke_tests/reliance_two_fold/`.
