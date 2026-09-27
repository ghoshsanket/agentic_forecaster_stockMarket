# Predictions

Per-date, per-ticker calibrated predictions for the full paper reproduction
(`predictions.csv.gz`).

Written by:

```bash
uv run python -m agentic_forecaster reproduce-paper \
    --config configs/paper.yaml --device auto --export-final-results
```

Columns: `date`, `ticker`, `y`, `raw_p_up`, `calibrated_p_up`, `fold`,
`direction`, `confidence`, `origin_date`, `target_date`.

The file is gzipped because the full 49-ticker x 2-fold test set is large. The
1-ticker real-data smoke-test sample is in
`../smoke_tests/reliance_two_fold/`.
