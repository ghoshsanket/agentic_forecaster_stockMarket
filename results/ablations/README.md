# Ablations

Executed ablation variants for the full paper reproduction:

| Variant | Question |
|---|---|
| `attention_lstm` vs `plain_lstm` | Does attention help? |
| `attention_lstm_raw` vs `attention_lstm_calibrated` | Does temperature scaling help? |
| `ohlcv_only` vs `ohlcv_plus_technical` | Do the Phase-1 technical indicators help? |

Populated by:

```bash
uv run python -m agentic_forecaster reproduce-paper \
    --config configs/paper.yaml --device auto --export-final-results
```

`ablation_metrics.csv` is written here by that export. The primary model and
the plain-LSTM baseline are **reused** from the main run; only the OHLCV-only
Attention-LSTM is trained additionally, so each ticker/fold needs three neural
fits rather than five.
