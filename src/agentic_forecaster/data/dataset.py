"""Dataset discovery, loading and synthetic fixture generation.

The raw Kaggle dataset is NOT stored in Git.  These helpers locate it under
``$AGENTIC_foreCASTER_RAW`` (Category B) and load per-ticker CSV files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("agentic_forecaster.data")

REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


_SUFFIXES_TO_STRIP = ("_MINUTE", "_1MIN", "_1M", "_INTRADAY", "_MIN")


def normalize_symbol(stem: str) -> str:
    """Normalize a raw CSV stem into a ticker symbol.

    Handles the dataset's duplicated ``*_minute_new.csv`` files by stripping
    the ``_NEW`` marker and then the frequency suffix, so
    ``MM_minute_new`` and ``MM_minute`` both map to ``MM``.
    """
    ticker = stem.upper()
    ticker = ticker.removesuffix("_NEW")
    for suffix in _SUFFIXES_TO_STRIP:
        if ticker.endswith(suffix):
            ticker = ticker[: -len(suffix)]
            break
    return ticker


def discover_ticker_files(raw_root: str | Path) -> dict[str, Path]:
    """Discover per-ticker CSV files under the raw dataset root.

    Expected layout (Kaggle download)::

        <raw_root>/
            RELIANCE_minute.csv
            TCS_minute.csv
            ...

    Files whose stem normalises to the same symbol (e.g. ``MM_minute.csv``
    and ``MM_minute_new.csv``) are deduplicated; the canonical (non-``_new``)
    file wins.  Returns a mapping ``{ticker: path}``.
    """
    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(
            f"Raw dataset directory not found: {raw_root}\n"
            "Run `python scripts/download_dataset.py` first. "
            "See docs/DATASET_PROVENANCE.md."
        )
    candidates: dict[str, list[Path]] = {}
    for csv_path in sorted(raw_root.glob("*.csv")):
        ticker = normalize_symbol(csv_path.stem)
        candidates.setdefault(ticker, []).append(csv_path)

    files: dict[str, Path] = {}
    for ticker, paths in candidates.items():
        # Prefer a file without the duplicated "_new" marker, then the largest.
        def rank(p: Path) -> tuple[int, int]:
            return (1 if "_new" in p.stem.lower() else 0, -p.stat().st_size)
        files[ticker] = min(paths, key=rank)

    if not files:
        raise FileNotFoundError(
            f"No per-ticker CSV files found under {raw_root}. "
            "Expected one CSV per NIFTY constituent."
        )
    deduped = sum(len(v) - 1 for v in candidates.values() if len(v) > 1)
    if deduped:
        logger.info("Deduplicated %d duplicate symbol file(s)", deduped)
    logger.info("Discovered %d ticker files under %s", len(files), raw_root)
    return files


def load_ticker_frame(path: str | Path) -> pd.DataFrame:
    """Load and normalise a single ticker CSV.

    Normalisation:
      - lowercase column names
      - parse ``date`` column and sort ascending
      - coerce OHLCV to float
      - drop rows with missing close
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"{path}: missing 'date' column")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def make_synthetic_dataset(
    n_tickers: int = 3,
    n_days: int = 600,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate a small deterministic synthetic OHLCV dataset.

    Used by the demo config and by CI tests.  NOT a substitute for the real
    dataset; exists only to exercise the pipeline.
    """
    rng = np.random.default_rng(seed)
    tickers = [f"SYN{i:02d}" for i in range(n_tickers)]
    frames: dict[str, pd.DataFrame] = {}
    for t in tickers:
        dates = pd.bdate_range("2020-01-01", periods=n_days)
        drift = rng.normal(0.0003, 0.0002)
        vol = rng.uniform(0.012, 0.025)
        log_returns = rng.normal(drift, vol, n_days)
        close = 100.0 * np.exp(np.cumsum(log_returns))
        open_ = close * (1 + rng.normal(0, 0.003, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n_days)))
        volume = rng.integers(1_000_000, 10_000_000, n_days).astype(float)
        frames[t] = pd.DataFrame(
            {
                "date": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
    return frames
