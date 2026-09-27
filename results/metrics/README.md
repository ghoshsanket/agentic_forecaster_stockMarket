# Metrics

This directory holds metric outputs from the reconstruction run.

| File | Provenance | Description |
|---|---|---|
| `synthetic_smoke_test_metrics.json` | `synthetic_smoke_test` | Metrics from the demo config on synthetic data |

Full per-fold metrics from the real reconstruction run are written to
`$AGENTIC_OUTPUT_ROOT/metrics/` (Category B) and exported into
`results/paper_reproduction/` by `scripts/package_submission.py`.
