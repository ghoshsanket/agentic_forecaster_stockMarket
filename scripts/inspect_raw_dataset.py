#!/usr/bin/env python3
"""Inspect the raw Kaggle dataset without modifying it."""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import pandas as pd

from agentic_forecaster import DATASET_SLUG, DATASET_URL
from agentic_forecaster.config import get_env_roots
from agentic_forecaster.utils import atomic_json_dump, setup_logging

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet", ".feather"}
SAMPLE_ROWS = 10000
CHUNK_SIZE = 100000


def discover_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_frequency(date_series: pd.Series) -> str:
    if len(date_series) < 2:
        return "unknown"
    diffs = date_series.diff().dropna()
    if len(diffs) == 0:
        return "unknown"
    median_diff = diffs.median()
    if median_diff <= pd.Timedelta(minutes=1):
        return "1-minute"
    elif median_diff <= pd.Timedelta(minutes=5):
        return "5-minute"
    elif median_diff <= pd.Timedelta(minutes=15):
        return "15-minute"
    elif median_diff <= pd.Timedelta(minutes=30):
        return "30-minute"
    elif median_diff <= pd.Timedelta(hours=1):
        return "hourly"
    elif median_diff <= pd.Timedelta(days=1):
        return "daily"
    else:
        return f"other ({median_diff})"


def inspect_file(path: Path) -> dict:
    ext = path.suffix.lower()
    info: dict = {
        "filename": path.name,
        "relative_path": str(path),
        "extension": ext,
        "size_bytes": path.stat().st_size,
        "sha256": compute_sha256(path),
    }

    try:
        if ext == ".csv":
            df_sample = pd.read_csv(path, nrows=SAMPLE_ROWS)
            row_count = sum(1 for _ in pd.read_csv(path, chunksize=CHUNK_SIZE))
        elif ext in {".xlsx", ".xls"}:
            df_sample = pd.read_excel(path, nrows=SAMPLE_ROWS)
            row_count = len(pd.read_excel(path))
        elif ext == ".parquet":
            df_sample = pd.read_parquet(path).head(SAMPLE_ROWS)
            row_count = len(pd.read_parquet(path))
        elif ext == ".feather":
            df_sample = pd.read_feather(path).head(SAMPLE_ROWS)
            row_count = len(pd.read_feather(path))
        else:
            return info

        info["rows"] = row_count
        info["columns"] = list(df_sample.columns)
        info["dtypes"] = {c: str(t) for c, t in df_sample.dtypes.items()}

        date_cols = [c for c in df_sample.columns if any(k in c.lower() for k in ["date", "time", "timestamp"])]
        info["date_columns"] = date_cols
        for col in date_cols:
            try:
                dates = pd.to_datetime(df_sample[col], errors="coerce")
                info[f"{col}_min"] = str(dates.min())
                info[f"{col}_max"] = str(dates.max())
                info[f"{col}_frequency"] = detect_frequency(dates)
            except (OSError, ValueError, TypeError):
                pass

        ticker_cols = [c for c in df_sample.columns if any(k in c.lower() for k in ["ticker", "symbol", "stock", "name"])]
        info["ticker_columns"] = ticker_cols

        ohlcv_candidates = {
            "open": [c for c in df_sample.columns if "open" in c.lower()],
            "high": [c for c in df_sample.columns if "high" in c.lower()],
            "low": [c for c in df_sample.columns if "low" in c.lower()],
            "close": [c for c in df_sample.columns if "close" in c.lower()],
            "volume": [c for c in df_sample.columns if any(k in c.lower() for k in ["volume", "vol"])],
        }
        info["ohlcv_columns"] = {k: v for k, v in ohlcv_candidates.items() if v}
        info["ohlcv_available"] = all(len(v) > 0 for v in ohlcv_candidates.values())

        indicator_keywords = ["rsi", "macd", "atr", "bollinger", "bb_", "ma_", "sma", "ema", "obv", "volatility", "stochastic", "cci", "williams", "adx"]
        info["indicator_columns"] = [c for c in df_sample.columns if any(k in c.lower() for k in indicator_keywords)]

        info["missing_counts"] = {c: int(df_sample[c].isna().sum()) for c in df_sample.columns if df_sample[c].isna().sum() > 0}

        ticker_from_name = path.stem.split("_")[0]
        info["_ticker_from_filename"] = ticker_from_name

    except (OSError, ValueError, TypeError, KeyError) as e:
        info["error"] = str(e)

    return info


def main() -> int:
    setup_logging()
    roots = get_env_roots()
    default_raw = Path(roots["AGENTIC_RAW_DATA_ROOT"])
    default_data_root = default_raw.parent
    default_meta = default_data_root / "metadata"

    parser = argparse.ArgumentParser(description="Inspect raw dataset")
    parser.add_argument("--raw-dir", type=Path, default=default_raw)
    parser.add_argument("--metadata-dir", type=Path, default=default_meta)
    args = parser.parse_args()

    raw_dir = args.raw_dir
    meta_dir = args.metadata_dir
    meta_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(raw_dir)
    if not files:
        logger.warning("No data files found in %s", raw_dir)
        print(f"No data files found in {raw_dir}")
        return 1

    logger.info("Inspecting %d files...", len(files))
    file_infos = []
    for i, f in enumerate(files, 1):
        logger.info("  [%d/%d] %s", i, len(files), f.name)
        file_infos.append(inspect_file(f))

    all_tickers = set()
    for info in file_infos:
        ticker = info.get("_ticker_from_filename")
        if ticker:
            all_tickers.add(ticker)

    tickers_sorted = sorted(all_tickers)
    (meta_dir / "discovered_tickers.txt").write_text("\n".join(tickers_sorted) + "\n")

    all_dates = []
    for info in file_infos:
        for key, val in info.items():
            if key.endswith(("_min", "_max")):
                try:
                    all_dates.append(pd.to_datetime(val))
                except (OSError, ValueError, TypeError):
                    pass

    earliest = min(all_dates) if all_dates else None
    latest = max(all_dates) if all_dates else None

    frequencies = set()
    for info in file_infos:
        for key, val in info.items():
            if key.endswith("_frequency"):
                frequencies.add(val)

    total_rows = sum(info.get("rows", 0) for info in file_infos)
    total_size = sum(info.get("size_bytes", 0) for info in file_infos)

    years_available = {}
    if earliest and latest:
        for year in range(earliest.year, latest.year + 1):
            year_start = pd.Timestamp(f"{year}-01-01")
            year_end = pd.Timestamp(f"{year}-12-31")
            if earliest <= year_end and latest >= year_start:
                years_available[str(year)] = True

    walk_forward_years = {str(y): years_available.get(str(y), False) for y in range(2016, 2024)}

    paper_start = pd.Timestamp("2000-01-03")
    paper_end = pd.Timestamp("2025-11-04")
    if earliest and latest:
        if earliest <= paper_end and latest >= paper_start:
            overlap_start = max(earliest, paper_start)
            overlap_end = min(latest, paper_end)
            if overlap_start <= overlap_end:
                paper_match = "PARTIAL MATCH"
            else:
                paper_match = "NO MATCH"
        else:
            paper_match = "NO MATCH"
    else:
        paper_match = "UNKNOWN"

    summary = {
        "dataset_slug": DATASET_SLUG,
        "source_url": DATASET_URL,
        "file_count": len(files),
        "file_formats": {f.suffix.lower() for f in files},
        "total_size_bytes": total_size,
        "total_rows": total_rows,
        "ticker_count": len(tickers_sorted),
        "tickers": tickers_sorted,
        "datetime_columns": {k for info in file_infos for k in info.get("date_columns", [])},
        "ticker_columns": {k for info in file_infos for k in info.get("ticker_columns", [])},
        "earliest_timestamp": str(earliest) if earliest else None,
        "latest_timestamp": str(latest) if latest else None,
        "detected_frequency": list(frequencies),
        "ohlcv_available": any(info.get("ohlcv_available", False) for info in file_infos),
        "ohlcv_column_mapping": {k: v for info in file_infos for k, v in info.get("ohlcv_columns", {}).items()},
        "indicator_columns": {c for info in file_infos for c in info.get("indicator_columns", [])},
        "yearly_availability_2016_2023": walk_forward_years,
        "paper_date_match_status": paper_match,
        "paper_described_start": str(paper_start),
        "paper_described_end": str(paper_end),
        "downloaded_start": str(earliest) if earliest else None,
        "downloaded_end": str(latest) if latest else None,
        "notes": "Dataset inspection complete. Source data is minute-level; Phase-1 will resample to daily.",
    }

    atomic_json_dump(summary, meta_dir / "dataset_summary.json")

    manifest = {
        "dataset_slug": DATASET_SLUG,
        "source_url": DATASET_URL,
        "files": [
            {
                "relative_path": info["relative_path"],
                "size_bytes": info["size_bytes"],
                "sha256": info["sha256"],
                "extension": info["extension"],
            }
            for info in file_infos
        ],
    }
    atomic_json_dump(manifest, default_data_root / "manifests" / "raw_manifest.json")

    print(f"Metadata written to {meta_dir}")
    print(f"Files inspected: {len(files)}")
    print(f"Tickers found: {len(tickers_sorted)}")
    print(f"Total rows: {total_rows}")
    print(f"Total size: {total_size / 1e9:.2f} GB")
    print(f"Frequencies: {list(frequencies)}")
    print(f"OHLCV available: {summary['ohlcv_available']}")
    print(f"Date range: {earliest} to {latest}")
    print(f"Paper date match: {paper_match}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
