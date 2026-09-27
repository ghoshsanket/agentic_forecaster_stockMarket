# Dataset Provenance

## Source

- **Kaggle URL:** https://www.kaggle.com/datasets/debashis74017/algo-trading-data-nifty-100-data-with-indicators
- **Slug:** `debashis74017/algo-trading-data-nifty-100-data-with-indicators`

## Download Status

**COMPLETE.** Dataset downloaded and extracted to `$AGENTIC_RAW_DATA_ROOT`.

## Actual Dataset Structure

- **File format:** CSV (one file per ticker)
- **File count:** 536 files
- **Total size:** 21.23 GB
- **Total rows:** 4,550 (sampled; actual row counts vary per file)
- **Ticker count:** 531 unique tickers
- **Date range:** 2015-02-02 09:15:00 to 2026-01-29 14:09:00
- **Frequency:** 1-minute intervals
- **OHLCV available:** Yes

## Paper Date Match

**PARTIAL MATCH.** The paper describes daily NIFTY-50 data from ~2000-01-03 to 2025-11-04. The actual dataset contains 1-minute data from 2015-02-02 to 2026-01-29.

## Key Differences from Paper

1. **Frequency:** Source is 1-minute; paper uses daily OHLCV. Phase-1 must resample to daily.
2. **Ticker universe:** 531 tickers available (NIFTY-100 constituents); paper uses NIFTY-50. Phase-1 must select 50.
3. **Date range:** Source starts 2015-02-02; paper references ~2000-01-03. Later start date.
4. **Indicators:** Source may include precomputed technical indicators; Phase-1 will recompute from OHLCV.

## Raw Data Location

```
$AGENTIC_RAW_DATA_ROOT
(default: $RESEARCH_ROOT/dataset/agentic-forecaster/raw/)
```

The raw dataset lives **outside** the Git repository.

## How to Download

```bash
source $RESEARCH_ROOT/scripts/research-env.sh
python scripts/download_dataset.py
```

## How to Inspect

```bash
python scripts/inspect_raw_dataset.py
```

Writes metadata to `$AGENTIC_DATA_ROOT/metadata/`.

## Metadata Files

- `$AGENTIC_DATA_ROOT/metadata/dataset_summary.json` — machine-readable summary
- `$AGENTIC_DATA_ROOT/metadata/discovered_tickers.txt` — sorted ticker list
- `$AGENTIC_DATA_ROOT/manifests/raw_manifest.json` — SHA-256 hashes

## Authentication

Kaggle API token stored at `$RESEARCH_ROOT/secrets/kaggle/access_token` (chmod 600). Token is loaded by `research-env.sh` and never printed or committed.
