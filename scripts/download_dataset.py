#!/usr/bin/env python3
"""Download the Kaggle dataset for the paper reconstruction."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from agentic_forecaster import DATASET_SLUG
from agentic_forecaster.config import get_env_roots
from agentic_forecaster.utils import setup_logging

logger = logging.getLogger(__name__)


def check_kaggle_auth() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    kaggle_dir = Path.home() / ".kaggle"
    return (kaggle_dir / "kaggle.json").exists()


def download_dataset(out_dir: Path, force: bool = False) -> Path:
    if not check_kaggle_auth():
        logger.error("Kaggle authentication not configured.")
        logger.error("Set KAGGLE_API_TOKEN or run scripts/setup-kaggle-token.sh")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("*.csv")) + list(out_dir.glob("*.zip"))
    if existing and not force:
        logger.info("Dataset files already present in %s (use --force to re-download)", out_dir)
        return out_dir

    logger.info("Downloading %s to %s", DATASET_SLUG, out_dir)
    cmd = [
        "kaggle", "datasets", "download",
        "-d", DATASET_SLUG,
        "-p", str(out_dir),
    ]
    if force:
        cmd.append("--force")

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("Kaggle download failed: %s", result.stderr)
        sys.exit(1)

    logger.info("Download complete: %s", out_dir)
    return out_dir


def extract_archives(raw_dir: Path) -> None:
    for archive in raw_dir.glob("*.zip"):
        logger.info("Extracting %s", archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_dir)
        logger.info("Extracted %s", archive.name)


def main() -> int:
    setup_logging()
    roots = get_env_roots()
    default_out = Path(roots["AGENTIC_RAW_DATA_ROOT"])

    parser = argparse.ArgumentParser(description="Download Kaggle dataset")
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--extract", action="store_true", default=True)
    args = parser.parse_args()

    out = download_dataset(args.out, force=args.force)
    if args.extract:
        extract_archives(out)
    print(f"Raw data directory: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
