# Legacy Yahoo Finance daily dataset (USER-SUPPLIED legacy NIFTY-50)

**Status:** reconstructed and validated. **No model was trained.**
**Universe:** `USER_SUPPLIED_LEGACY_NIFTY50_UNIVERSE` (50 user-supplied labels).
**Window:** 2000-01-01 → 2025-12-31 (`end=2026-01-01`, exclusive).

## Independence from the modern dataset

This dataset is a **second, completely separate** reconstruction. The earlier
modern-universe dataset at `$AGENTIC_DATA_ROOT/yfinance_daily_2000_2005` /
`yfinance_daily_2000_2025` was **not modified, not overwritten and not read for
writing**. Its aggregate content fingerprint was captured before this task
began and re-verified afterwards.

| | Modern dataset | This dataset |
|---|---|---|
| Root | `yfinance_daily_2000_2005` → `$AGENTIC_DATA_ROOT/yfinance_daily_2000_2025` | `$AGENTIC_DATA_ROOT/yfinance_legacy_nifty50_2000_2025` |
| Universe | `RECONSTRUCTED_FIXED_NIFTY50_UNIVERSE` (today's NIFTY-50) | `USER_SUPPLIED_LEGACY_NIFTY50_UNIVERSE` |
| Config | `configs/nifty50.yaml` | `configs/nifty50_legacy_user_supplied.yaml` |
| Configs | `reproduction_search/yfinance_*.yaml` | `reproduction_search/legacy_yfinance_*.yaml` |

## Location

Root: `$AGENTIC_DATA_ROOT/yfinance_legacy_nifty50_2000_2025`.

```
yfinance_legacy_nifty50_2000_2025/
├── adjusted/                        PRIMARY   (auto_adjust=True)
│   ├── csv/                         41 canonical CSVs + manifests
│   ├── parquet/                     41 canonical Parquet files
│   └── source_snapshots/            41 raw yfinance frames, untouched
├── unadjusted/                      ORIGINAL-RECOLLECTION CANDIDATE (auto_adjust=False)
│   ├── csv/  parquet/  source_snapshots/
├── workbooks/
│   ├── adjusted/Nifty-50-Legacy-Historical-Securities.xlsx
│   ├── adjusted/Nifty-50.xlsx
│   ├── unadjusted/Nifty-50-Legacy-Historical-Securities.xlsx
│   └── unadjusted/Nifty-50.xlsx
├── metadata/
│   ├── legacy_coverage.csv          per-security outcome and date span
│   ├── security_lineage.csv         machine-readable identity registry
│   ├── successor_lineage_plan.csv   successors, for a FUTURE experiment only
│   ├── modern_vs_legacy_universe.csv
│   ├── yahoo_symbol_probe.csv       every symbol probed, with what Yahoo returned
│   ├── sheet_name_map.csv           legacy_label -> file/sheet names
│   ├── adjusted_vs_unadjusted.csv
│   ├── data_quality.json
│   └── coverage_summary.json
├── manifests/
│   ├── legacy_yfinance_manifest.json
│   └── manifest_sha256.csv          332 files hashed
├── corporate_history/<label>.md     50 per-security evidence records
├── cache/yfinance/                  yfinance tz + cookie cache
├── logs/download.log
└── tmp/
```

All caches stay under `RESEARCH_ROOT`. No global yfinance cache was created.

## Request parameters

`start="2000-01-01"`, `end="2026-01-01"` (**exclusive**), `interval="1d"`,
`actions=False`, `progress=False`, `repair=False`, `prepost=False`,
`threads=False`, both `auto_adjust=True` and `auto_adjust=False`.
`yf.set_tz_cache_location()` is called with
`$AGENTIC_YFINANCE_LEGACY_ROOT/cache/yfinance` before the first request.
Retries: 5 attempts, 2/5/10/20/40 s backoff, 1 s inter-symbol delay, atomic
temp-file renames, fully resumable.

## The central rule: a successor is not the same security

Two different situations, treated differently:

* **Same legal security, renamed ticker/name** → the history is continuous and
  is used (e.g. `HINDLEVER` → `HINDUNILVR.NS`).
* **Merged into / acquired by / demerged from a different legal person** → the
  history **ends**. The successor's prices are **never** appended, nothing is
  forward-filled, no last close is repeated, and no synthetic pre-listing data
  is created.

The rule is enforced in three independent places: the registry sets
`stitch_successors_into_canonical: false` (the loader *raises* if this is ever
set to true), every downloaded manifest records
`successor_prices_appended: false`, and 52 tests in
`tests/unit/test_legacy_security_lineage.py` assert each specific mapping.

## Coverage

| | |
|---|---|
| Requested | 50 |
| Retrieved in both variants | **41** |
| Unavailable from Yahoo (series correctly empty) | **9** |
| Rows (adjusted = unadjusted) | **254,834** |
| Securities covering the full window from 2000-01-03 | 23 of 41 |
| Earliest / latest | 2000-01-03 / 2025-12-31 |

The 9 unavailable securities were each merged into or acquired by a different
issuer before or during the window, and Yahoo retains no history for them on
either NSE or BSE (probed across 4-6 symbols each; see
`metadata/yahoo_symbol_probe.csv`): **BURROUGHS, COCHINREFN, HDFC, IBP, ICICI,
POND'S, RANBAXY, RHONE-POUL, SATYAMCOMP**.

**Coverage limitation:** 13 of the 41 retrieved securities begin 2002-07-01
rather than 2000-01-03, because Yahoo's NSE history for those symbols does not
reach back to 2000 (ACC, BAJAJ-AUTO, COLPAL, EPL, GE SHIPPING, GLAXO, GRASIM,
GUTRLCEMENT, IFCI, L&T, MADRASCEM, NOCIL, VSTILL; plus KNOLLPHARM, NESTLEIND
and P&G from 2002-08-12). This is a vendor limit, reported rather than filled.

## Canonical data contract

Exactly `Date,Open,High,Low,Close,Volume`. No `Adj Close`, `Dividends`,
`Stock Splits`, no index column. Timezone-naive ascending dates, deduplicated.
`Adj Close` appears only in the unadjusted **source snapshots**.

## Filenames and sheet names

Filesystem-safe stems (`BRITISH_OXYGEN_BOC.csv`, `L_AND_T.csv`, `P_AND_G.csv`,
`PONDS.csv`); the exact `legacy_label` is preserved in all metadata and in
`metadata/sheet_name_map.csv`.

All 50 supplied labels are legal Excel sheet names, so each workbook uses the
**exact label verbatim** as the sheet name — including `L&T`, `M&M`, `P&G`,
`POND'S` and `BRITISH OXYGEN (BOC)`.

## Workbooks

| Workbook | Sheets | Round-trip |
|---|---|---|
| `workbooks/adjusted/Nifty-50-Legacy-Historical-Securities.xlsx` | 50 | pass |
| `workbooks/unadjusted/Nifty-50-Legacy-Historical-Securities.xlsx` | 50 | pass |
| `workbooks/adjusted/Nifty-50.xlsx` | 50 | pass |
| `workbooks/unadjusted/Nifty-50.xlsx` | 50 | pass |

Each is read back with openpyxl and compared row by row against the canonical
Parquet: header, row count, date and all five value columns. The 9 unavailable
securities have **empty sheets containing only the header row** — no invented
rows, and no commentary inside the OHLCV table.

## Reproducing

```bash
cd "$AGENTIC_PROJECT_ROOT"
source "$RESEARCH_ROOT/scripts/research-env.sh"

# 1. forensic symbol probe (writes metadata/yahoo_symbol_probe.csv)
uv run --frozen python scripts/probe_legacy_yahoo_symbols.py

# 2. download both variants (resumable)
uv run --frozen python scripts/download_yfinance_daily.py \
    --universe-config configs/nifty50_legacy_user_supplied.yaml \
    --lineage-config  configs/legacy_security_lineage.yaml \
    --dataset-root    "$AGENTIC_YFINANCE_LEGACY_ROOT" \
    --start 2000-01-01 --end 2026-01-01 --variant both --resume --all

# 3. workbooks, coverage, lineage metadata, manifests
uv run --frozen python scripts/build_legacy_yfinance_artifacts.py

# 4. regenerate the lineage document from the registry
uv run --frozen python scripts/generate_legacy_lineage_doc.py
```

Offline tests:

```bash
uv run --frozen python -m pytest tests/unit/test_legacy_security_lineage.py -q
```

## Scope limits

* **No model was trained.** No features, targets, sequences, Attention-LSTM,
  calibration or metrics were computed. No paper result was overwritten.
* `configs/reproduction_search/legacy_yfinance_{unadjusted,adjusted}.yaml`
  were **created but deliberately not run**, pending inspection of this data.
* The unadjusted variant is the **ORIGINAL-RECOLLECTION CANDIDATE** and the
  adjusted variant the **CORPORATE-ACTION-ADJUSTED SENSITIVITY**. Neither is
  claimed to be the proven original.
* The `POND'S` date anomaly is documented, not corrected — see
  `docs/LEGACY_UNIVERSE_HISTORY.md`.
* `docs/LEGACY_SECURITY_LINEAGE.md` is generated from the registry and records
  `successor stitched = NO` for all 50 securities.
* `metadata/successor_lineage_plan.csv` records possible successor experiments
  that were **not** performed.
* This large dataset lives outside Git and is not committed.
