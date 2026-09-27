"""Kaggle dataset downloader.

Uses the ``kaggle`` Python package (``kagglehub``/``kaggle.api``) to download
::

    debashis74017/algo-trading-data-nifty-100-data-with-indicators

Authentication is read from ``$KAGGLE_USERNAME`` / ``$KAGGLE_KEY`` or
``~/.kaggle/kaggle.json``.  Credentials are never written to the repository.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agentic_forecaster import DATASET_SLUG

logger = logging.getLogger("agentic_forecaster.data.downloader")


def download_dataset(raw_root: str | Path, slug: str = DATASET_SLUG) -> Path:
    """Download and extract the Kaggle dataset into ``raw_root``."""
    raw_root = Path(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "The 'kaggle' package is required. Install with: "
            "pip install kaggle  (or: pip install -e '.[data]')"
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise RuntimeError(
            "Kaggle authentication failed. Set KAGGLE_USERNAME and KAGGLE_KEY "
            "in the environment, or place kaggle.json in ~/.kaggle/. "
            "See docs/DATASET_PROVENANCE.md — never commit credentials."
        ) from exc

    logger.info("Downloading %s -> %s", slug, raw_root)
    api.dataset_download_files(slug, path=str(raw_root), unzip=True)
    logger.info("Download complete: %s", raw_root)
    return raw_root


def inspect_dataset(raw_root: str | Path) -> dict:
    """Return a lightweight inventory of the raw dataset."""
    from agentic_forecaster.data.dataset import discover_ticker_files, load_ticker_frame

    files = discover_ticker_files(raw_root)
    inventory = {"slug": DATASET_SLUG, "n_tickers": len(files), "tickers": {}}
    for ticker, path in sorted(files.items()):
        df = load_ticker_frame(path)
        inventory["tickers"][ticker] = {
            "rows": len(df),
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
        }
    return inventory
