# Reports

The Report Agent produces per-ticker HTML (and optionally PDF) reports.

## Committed examples

| File | Provenance | Description |
|---|---|---|
| `examples/RELIANCE_2024-12-31.html` | `synthetic_smoke_test` | Representative example (synthetic data) |
| `examples/RELIANCE_2024-12-31.pdf` | `synthetic_smoke_test` | PDF version of the above |

These examples are generated from the demo config on synthetic data and are
clearly watermarked.  They demonstrate the report format; they are **not**
paper-reconstruction results.

## Generating more reports

```bash
# Full reproduction (generates reports under $AGENTIC_OUTPUT_ROOT/reports/)
python scripts/reproduce_paper.py --config configs/paper.yaml

# Export representative reports into the repository
python scripts/package_submission.py
```

## Report contents

Each report includes:

- Ticker, prediction date, direction and conviction
- Risk level, position size, stop-loss and take-profit
- Explanation narrative (LLM or deterministic fallback)
- Top SHAP features
- Model metrics on the test split
