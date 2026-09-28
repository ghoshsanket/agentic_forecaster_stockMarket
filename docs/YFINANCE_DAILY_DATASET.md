# Yahoo Finance Daily Dataset (2000-2025)

**Status:** reconstructed and validated. Not a primary result of the paper
reproduction.

## Purpose

The original project dataset was a daily OHLCV workbook for the NIFTY-50,
sourced from Yahoo Finance. That workbook was lost. This document records the
reconstruction of an equivalent dataset from the same upstream source, plus an
unadjusted sensitivity variant, so that the effect of the `auto_adjust` choice
can be measured rather than assumed.

This dataset is **not** used by the primary reported result. The paper
reproduction still runs against the Kaggle dataset
(`configs/paper.yaml`). The two reproduction-search configs
(`configs/reproduction_search/yfinance_adjusted.yaml` and
`.../yfinance_unadjusted.yaml`) exist only to point the existing pipeline at
this data for a controlled data-source comparison. Nothing was trained as part
of building this dataset.

## Location

Root: `$AGENTIC_DATA_ROOT/yfinance_daily_2000_2025`, which resolves to
`$RESEARCH_ROOT/dataset/yfinance_daily_2000_2025`.

```
yfinance_daily_2000_2025/
├── adjusted/                       PRIMARY   (auto_adjust=True)
│   ├── csv/                        50 canonical CSVs + per-ticker manifest
│   ├── parquet/                    50 canonical Parquet files
│   └── source_snapshots/           50 raw yfinance frames, untouched
├── unadjusted/                     SENSITIVITY (auto_adjust=False)
│   ├── csv/  parquet/  source_snapshots/
├── workbooks/
│   ├── adjusted/Nifty-50.xlsx      50 sheets, Date,Open,High,Low,Close,Volume
│   └── unadjusted/Nifty-50.xlsx
├── metadata/
│   ├── coverage.csv                per-ticker, per-variant spans and row counts
│   ├── coverage_summary.json
│   ├── data_quality.json           OHLC consistency, non-positive prices
│   ├── corporate_action_jumps.csv  |1-day return| > 25%
│   ├── yfinance_vs_kaggle.csv      comparison against the Kaggle dataset
│   ├── yfinance_vs_kaggle.json
│   └── download_summary.json       request parameters, versions, provenance
├── manifests/manifest_sha256.csv   sha256 of every delivered file
├── logs/download.log
└── cache/yfinance/                 yfinance tz + cookie cache
```

This root sits beside the Kaggle data under `$AGENTIC_DATA_ROOT` and is
completely independent of it. Nothing under
`$AGENTIC_RAW_DATA_ROOT`, `$AGENTIC_PROCESSED_DATA_ROOT`, `$AGENTIC_MODEL_ROOT`
or `$AGENTIC_OUTPUT_ROOT` was read for writing or modified.

## Exact request parameters

Every ticker, in both variants, was requested with:

| Parameter | Value | Note |
|---|---|---|
| `start` | `2000-01-01` | |
| `end` | `2026-01-01` | **exclusive**; last bar is 2025-12-31 |
| `interval` | `1d` | |
| `auto_adjust` | `True` (adjusted) / `False` (unadjusted) | |
| `actions` | `False` | no dividend/split columns requested |
| `repair` | `False` | yfinance repair heuristics left off |
| `prepost` | `False` | regular trading hours only |
| `threads` | `False` | one request at a time, deterministic order |
| `progress` | `False` | |

`yf.set_tz_cache_location()` is called with
`$AGENTIC_YFINANCE_DAILY_ROOT/cache/yfinance` **before the first request**, so
yfinance's persistent cache never touches a global location. Verified: no
`~/.cache/py-yfinance`, `~/Library/Caches/py-yfinance` or `~/AppData` directory
was created.

Retries: up to 5 attempts with 2/5/10/20/40 second backoff, plus a 1 second
delay between symbols. Files are written to a temporary name and atomically
renamed, so an interrupted run cannot be mistaken for a complete one.

## Column contract

Canonical CSV, Parquet and workbook columns are exactly:

```
Date,Open,High,Low,Close,Volume
```

* `Date` is a timezone-naive calendar date, ascending, deduplicated.
* `Adj Close`, `Dividends`, `Stock Splits` and `Capital Gains` are **never**
  written to canonical output. The unadjusted variant keeps them only in
  `source_snapshots/`, which stores the immediate yfinance frame unmodified.
* No index column. Nothing is forward-filled, interpolated or invented.

## Universe and symbol mapping

50 securities, recovered from `configs/nifty50.yaml` and cross-checked against
`results/ticker_availability.csv` and the repository manifests.

**Provenance label: `RECONSTRUCTED_FIXED_NIFTY50_UNIVERSE`.** This is *not* the
exact original lost universe. NIFTY-50 membership is time-varying, the
publication names no as-of date, and the two surviving sources are themselves
reconstructions. The list is a fixed, documented, current-constituent set, and
is labelled as such rather than being presented as the original.

Mapping is the NSE form `<SYMBOL>.NS`. Punctuation is preserved verbatim, so
`M&M -> M&M.NS` and `BAJAJ-AUTO -> BAJAJ-AUTO.NS`. Symbols that are already
explicit Yahoo symbols (containing a dot, a `^` prefix or an `=` suffix) are
not suffixed.

`configs/nifty50.yaml` also carries a Kaggle vendor alias `M&M -> MM`. That
alias is **Kaggle-specific and was deliberately not applied to Yahoo**, where
`M&M.NS` is the correct symbol.

### ZOMATO -> ETERNAL (same-security rename)

`ZOMATO.NS` returns no data. This is a rename, not a delisting:

* NSE circular CML67420: "name and symbol of Zomato Limited will be changed
  w.e.f. April 09, 2025" (name change approved by the Ministry of Corporate
  Affairs). Zomato Ltd became **Eternal Ltd** on both NSE and BSE.
* Yahoo reports `longName = "Eternal Limited"`, exchange NSE, currency INR,
  sector Consumer Cyclical, industry Internet Retail — matching Zomato.
* The series is continuous: 1099 bars from 2021-07-23 (IPO era) through
  2025-12-31, with no gap or split across 2025-04-09 (that day closes
  -1.77%, an ordinary session).

This is the same legal entity and ISIN, so the mapping is authorised and is
recorded in the affected per-ticker manifests under `same_security_rename`.
No other substitution was made; anything unresolvable would have been reported
as unavailable rather than replaced.

## Coverage

All 50 securities downloaded successfully in both variants.

| Variant | Complete | Rows | Files |
|---|---|---|---|
| adjusted (primary) | 50/50 | 287,277 | 50 CSV + 50 Parquet + 50 snapshots |
| unadjusted (sensitivity) | 50/50 | 287,277 | 50 CSV + 50 Parquet + 50 snapshots |

Row counts are equal because `auto_adjust` rescales prices; it does not change
which dates exist. Per-ticker spans are in `metadata/coverage.csv`; the earliest
bar is 2000-01-03 and the latest is 2025-12-31. Securities that listed later
start later (ZOMATO/ETERNAL 2021-07-23, JSWSTEEL 2003-05-08, UPL 2002-07-01).

## Data quality findings

Full detail in `metadata/data_quality.json`. Two findings are real and are
reported rather than repaired, because the instruction is to preserve vendor
values and their quirks.

**1. `ADANIENT` adjusted prices are negative for the first 47 bars (2002-07-01
onward).** Yahoo's back-adjustment multiplies the early series by a factor
derived from cumulative dividends; for ADANIENT's 2002 price that product goes
negative. The unadjusted series for the same dates is itself implausible
(~1.36 INR against a ~1,400 INR price today), indicating a vendor split-basis
problem in ADANIENT's early Yahoo history. No value was altered. Consumers of
the adjusted variant should treat pre-2002-07 ADANIENT history as unusable.
This is 47 rows in 1 of 50 tickers; the other 49 have no non-positive prices.

**2. 157 rows across both variants have a |1-day return| > 25%** — genuine
corporate actions, listed in `metadata/corporate_action_jumps.csv`. These are
diagnostics only; no series was split-adjusted further.

The adjusted variant additionally shows small OHLC ordering breaches
(`High` marginally below `Close`, etc.) on 30 of 50 tickers, at magnitudes at
the 1e-16 level. These are float artefacts of the back-adjustment factor and
are reported, not corrected. The unadjusted variant has none.

## Workbook verification

Both workbooks were read back with openpyxl and compared row by row against the
canonical Parquet, checking header, row count, date and all five value columns
per cell.

| Workbook | Sheets | Rows verified | Round-trip |
|---|---|---|---|
| `workbooks/adjusted/Nifty-50.xlsx` | 50 | 287,277 | pass |
| `workbooks/unadjusted/Nifty-50.xlsx` | 50 | 287,277 | pass |

`M&M` is written to sheet `M-M` because `&` is not permitted in an Excel sheet
name; the CSV/Parquet filenames keep the `M&M` spelling.

## Reproducing

```bash
cd "$AGENTIC_PROJECT_ROOT"
source "$RESEARCH_ROOT/scripts/research-env.sh"
uv run --frozen python scripts/download_yfinance_daily.py \
    --start 2000-01-01 --end 2026-01-01 --variant both --resume --all
uv run --frozen python scripts/build_yfinance_artifacts.py
uv run --frozen python scripts/compare_yfinance_vs_kaggle.py
```

`--resume` skips any ticker whose manifest records the same parameters and
whose files validate. Use `--force` to re-download, or `--ticker SYMBOL` to
refresh one security.

Offline tests (no network):

```bash
uv run --frozen python -m pytest tests/unit/test_yfinance_download.py -q
```

One test is marked `network` and is skipped unless
`AGENTIC_RUN_NETWORK_TESTS=1`.

## Scope limits

* This dataset was built and validated only. No features, targets, models,
  calibration or metrics were computed, and no model was trained.
* Version recorded in `metadata/download_summary.json`: yfinance 1.7.0.
* Vendor data can change retroactively. `manifests/manifest_sha256.csv` pins
  the exact bytes delivered here.
* This large dataset lives outside Git by design and is not committed.
