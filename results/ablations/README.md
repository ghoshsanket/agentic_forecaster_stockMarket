# Ablations

Ablation study results.

The paper's ablations disable individual components to measure their
contribution.  To reproduce:

```bash
# Disable attention (plain LSTM only)
# Edit configs/paper.yaml: models.attention_lstm.enabled = false
python scripts/reproduce_paper.py --config configs/paper.yaml

# Disable SHAP (attention evidence only)
# Edit configs/paper.yaml: explainability.method = attention_only
python scripts/reproduce_paper.py --config configs/paper.yaml
```

| File | Provenance | Description |
|---|---|---|
| `synthetic_smoke_test_ablation.json` | `synthetic_smoke_test` | Demo ablation on synthetic data |
