# Dataset Provenance

## Source

- **Kaggle URL:** https://www.kaggle.com/datasets/debashis74017/algo-trading-data-nifty-100-data-with-indicators
- **Slug:** `debashis74017/algo-trading-data-nifty-100-data-with-indicators`

## Download Status

**COMPLETE.** Dataset downloaded and extracted to `$AGENTIC_RAW_DATA_ROOT`.

## Actual Dataset Structure (authoritative)

Regenerate this record at any time — it reads the first and last data line of
every raw CSV, which is cheap even for ~21 GB of 1-minute data:

```bash
uv run python scripts/verify_dataset_provenance.py
```

It rewrites `$AGENTIC_RAW_DATA_ROOT/../metadata/{dataset_summary.json,
discovered_tickers.txt}` and is the **authoritative** source for these fields.

- **File format:** CSV (one file per ticker)
- **File count:** 536 files
- **Unique symbols:** **531** (5 duplicated `*_minute_new.csv` files deduplicated
  by `discover_ticker_files`: `AREM`, `GVTD`, `JKBANK`, `MM`, `MMFIN`)
- **Total size (deduplicated):** 20,973,947,683 bytes (~20.97 GB)
- **Date range:** **2015-02-02 09:15:00 → 2026-04-08 15:29:00**
- **Frequency:** 1-minute
- **OHLCV available:** Yes
- **Phase-1 frequency:** daily (resampled)

### Reconciled against the earlier Prompt-A record

1. **End date.** The earlier record reported `2026-01-29`. The raw files
   actually extend to **2026-04-08 15:29:00** (verified by reading the last
   data line of every file). `results/ticker_availability.csv` agrees with
   this; per-ticker variation (e.g. `MM` ends `2026-01-23`) is a genuine
   property of that stock's file, not a contradiction.
2. **Symbol count.** Previously reported as 536 by the discovery scan; after
   normalising the duplicated `_minute_new` files the count is a stable
   **531**, matching the original record.

Both paper folds (test years 2022 and 2023) fall well inside the reconciled
range, so neither correction changes any reported result.

## Paper Date Match

**PARTIAL MATCH.** The paper describes daily NIFTY-50 data from ~2000-01-03 to
2025-11-04. The actual dataset contains 1-minute data from 2015-02-02 to
2026-04-08, which covers both paper walk-forward folds (2022 and 2023).

## Key Differences from Paper

1. **Frequency:** Source is 1-minute; paper uses daily OHLCV. Phase-1 resamples
   to daily (`data/resampling.py`).
2. **Ticker universe:** 531 symbols available; the paper uses NIFTY-50.
   `configs/nifty50.yaml` fixes a 50-symbol list and reports anything that
   cannot be resolved as unavailable (never substituting a different company).
3. **Date range:** Source starts 2015-02-02; the paper references ~2000-01-03.
   Both paper folds (train from 2016) are still covered.

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
