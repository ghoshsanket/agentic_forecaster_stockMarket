# Real-Data Integration Smoke Tests

**Real-data integration smoke test only — not the full NIFTY-50 reproduction.**

## What this is

Artefacts from a deliberately small real-data run used to validate the
pipeline end-to-end before committing to the expensive full experiment:

- **run_id:** `testrun`
- **Ticker:** `RELIANCE` (1 stock)
- **Folds:** `fold_0` (test year 2022), `fold_1` (test year 2023)
- **ticker/fold runs:** 2

Each of these is a real, trained Attention-LSTM on real daily bars resampled
from the real 1-minute Kaggle dataset, with real temperature calibration,
real SHAP attribution, real ATR-based risk levels and real HTML/PDF reports.

## What this is NOT

- It is **not** the paper reproduction.
- It covers **1 of 49** available configured securities, not the full universe.
- Its accuracy (~0.44 / ~0.52) is **not** comparable to the paper's reported
  0.815 and must not be cited as a reconstruction result.

The authoritative reproduction results live in `../paper_reproduction/`, which
remains in a pre-full-run state until the full universe run completes.

## Contents

| File | Description |
|---|---|
| `ticker_metrics.csv` | Per ticker/fold metrics for RELIANCE (2 rows) |
| `aggregate_metrics.csv` → `aggregate_metrics.json` | Mean metrics over those 2 runs |
| `calibration_metrics.csv` | Temperature and raw/calibrated Brier + ECE |
| `precision_at_3.csv` | Cross-sectional P@3 (degenerate: only 1 ticker) |
| `p3_daily_selections.csv` | Daily top-3 selections (audit trail) |
| `baseline_metrics.csv` | Plain LSTM / RF / LR / Majority for RELIANCE |
| `ablation_metrics.csv` | Ablation variants for RELIANCE |
| `paper_comparison.csv` | Paper reference vs this 1-ticker run |
| `predictions.csv.gz` | Per-date predictions (2 folds) |
| `training_summary.json` | Training config + loss history |
| `ticker_status.json` | Per ticker/fold status |
| `run_manifest.json` | Run manifest |

## Reproducing this smoke test

```bash
source $RESEARCH_ROOT/scripts/research-env.sh
uv run python -m agentic_forecaster walk-forward \
    --config configs/paper.yaml --device auto \
    --tickers RELIANCE --run-id testrun
```

Note that with a single ticker the cross-sectional Precision@3 is degenerate
(every date selects the same single stock), which is why this run is kept
separate from the real result set.
