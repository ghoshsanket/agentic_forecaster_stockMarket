# Data

This repository contains **code only**. The dataset is too large for Git, so it
is downloaded separately and stored outside the repository.

## Source

- **Yahoo Finance** (NSE), fetched with the `yfinance` package — no API key or
  account required.
- **Script:** `scripts/download_yfinance_daily.py`
- **Universe:** 50 NIFTY-50 symbols from `configs/nifty50.yaml`, requested as
  `<SYMBOL>.NS` (`M&M` → `M&M.NS`, punctuation preserved).
- **Range:** `start=2000-01-01`, `end=2026-01-01` (`end` is exclusive; last bar
  is 2025-12-31), `interval=1d`.
- **Canonical columns:** `Date,Open,High,Low,Close,Volume`
  (`Adj Close`, `Dividends` and `Stock Splits` are never written to canonical
  output).
- **Variants:** `adjusted` (`auto_adjust=True`, primary) and `unadjusted`
  (`auto_adjust=False`, sensitivity).

## Location (outside Git)

Root: `$AGENTIC_YFINANCE_DAILY_ROOT`, defaulting to
`$RESEARCH_ROOT/dataset/yfinance_daily_2000_2025`.

```
yfinance_daily_2000_2025/
├── adjusted/                      # PRIMARY
│   ├── csv/                       # 50 CSVs + per-ticker manifests
│   ├── parquet/                   # 50 Parquet files
│   └── source_snapshots/          # raw yfinance frames, untouched
├── unadjusted/                    # SENSITIVITY
│   ├── csv/  parquet/  source_snapshots/
├── workbooks/
│   ├── adjusted/Nifty-50.xlsx     # 50 sheets, canonical columns
│   └── unadjusted/Nifty-50.xlsx
├── metadata/                      # coverage, data quality, download summary
├── manifests/manifest_sha256.csv  # sha256 of every delivered file
├── logs/download.log
└── cache/yfinance/                # yfinance tz/cookie cache (redirected here)
```

## Download

```bash
# Resumable: a ticker whose manifest matches and whose files validate is skipped
uv run python scripts/download_yfinance_daily.py \
    --start 2000-01-01 --end 2026-01-01 --variant both --resume --all

# One ticker / force re-download
uv run python scripts/download_yfinance_daily.py --ticker RELIANCE --force

# Metadata, coverage, data-quality report and SHA-256 manifests
uv run python scripts/build_yfinance_artifacts.py
```

Downloads are written to a temporary name and atomically renamed, so an
interrupted run can never be mistaken for a complete one.

## Processed Data

The Data Agent derives daily bars, features and targets from the raw CSVs and
caches them under the configured `data.processed_root`:

- `configs/reproduction_search/yfinance_adjusted.yaml` →
  `$AGENTIC_OUTPUT_ROOT/reproduction_recovery/processed/yfinance_daily_2000_2025/adjusted`
- `configs/paper.yaml` → `$AGENTIC_PROCESSED_DATA_ROOT`

Processed data is also outside Git.

## Environment Variables

Every variable has a default derived from `RESEARCH_ROOT`; exporting one
overrides the default.

| Variable | Default | Purpose |
|---|---|---|
| `RESEARCH_ROOT` | `<repo>/../..` | Parent of `dataset/`, `models/`, `output/` |
| `AGENTIC_DATA_ROOT` | `$RESEARCH_ROOT/dataset` | Parent data directory |
| `AGENTIC_YFINANCE_DAILY_ROOT` | `$AGENTIC_DATA_ROOT/yfinance_daily_2000_2025` | Raw Yahoo Finance dataset |
| `AGENTIC_RAW_DATA_ROOT` | `$AGENTIC_DATA_ROOT/agentic-forecaster/raw` | Raw data for minute-bar configs |
| `AGENTIC_PROCESSED_DATA_ROOT` | `$AGENTIC_DATA_ROOT/agentic-forecaster/processed` | Cached processed sequences |
| `AGENTIC_MODEL_ROOT` | `$RESEARCH_ROOT/models/agentic-forecaster` | Trained checkpoints |
| `AGENTIC_OUTPUT_ROOT` | `$RESEARCH_ROOT/output/agentic-forecaster` | Run outputs, metrics, reports |

## Inspection

```bash
# Coverage, data-quality and manifest reports written by the step above
ls "$AGENTIC_YFINANCE_DAILY_ROOT/metadata"

# Confirm the pipeline reads the raw CSVs end to end
uv run python -m agentic_forecaster prepare-data \
    --config configs/reproduction_search/yfinance_adjusted.yaml --ticker RELIANCE
```

Per-ticker spans and row counts are in `metadata/coverage.csv`; quality
findings are in `metadata/data_quality.json`.

## Git Policy

The dataset is ignored on purpose (`.gitignore`):

```
data/raw/  data/interim/  data/processed/  datasets/
yfinance_daily_2000_2025/  **/yfinance_daily_2000_2025/  *.xlsx
outputs/  runtime_models/
```

Only this README is tracked under `data/`. Dataset details:
[docs/YFINANCE_DAILY_DATASET.md](../docs/YFINANCE_DAILY_DATASET.md).
