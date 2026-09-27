#!/usr/bin/env python3
"""Verify the 50-ticker universe against downloaded raw files.

Generates:
    results/ticker_availability.csv

with columns:
    requested_ticker, discovered_symbol, available, source_file,
    source_start, source_end, fold_0_available, fold_1_available
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from agentic_forecaster.config import get_env_roots, load_config
from agentic_forecaster.data.dataset import discover_ticker_files, load_ticker_frame
from agentic_forecaster.utils import setup_logging

PAPER_FOLDS = [
    {"fold": "fold_0", "test_start": "2022-01-01", "test_end": "2022-12-31"},
    {"fold": "fold_1", "test_start": "2023-01-01", "test_end": "2023-12-31"},
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--out", default="results/ticker_availability.csv")
    args = parser.parse_args()
    setup_logging()

    config = load_config(args.config)
    roots = get_env_roots()
    raw_root = Path(roots["AGENTIC_RAW_DATA_ROOT"])

    ticker_files = discover_ticker_files(raw_root)
    discovered = {k.upper(): v for k, v in ticker_files.items()}

    ticker_path = config["data"]["tickers"]
    if not Path(ticker_path).is_absolute():
        ticker_path = Path(config.get("_config_dir", ".")) / ticker_path
    import yaml
    with open(ticker_path) as f:
        requested = [t.upper() for t in yaml.safe_load(f).get("tickers", [])]

    rows = []
    for req in requested:
        available = req in discovered
        row = {
            "requested_ticker": req,
            "discovered_symbol": req if available else "",
            "available": available,
            "source_file": str(discovered[req]) if available else "",
            "source_start": "",
            "source_end": "",
            "fold_0_available": False,
            "fold_1_available": False,
        }
        if available:
            df = load_ticker_frame(discovered[req])
            row["source_start"] = str(df["date"].min().date())
            row["source_end"] = str(df["date"].max().date())
            for fold in PAPER_FOLDS:
                fold_key = f"{fold['fold']}_available"
                fold_start = pd.Timestamp(fold["test_start"])
                fold_end = pd.Timestamp(fold["test_end"])
                dates = pd.to_datetime(df["date"])
                mask = (dates >= fold_start) & (dates <= fold_end)
                row[fold_key] = int(mask.sum()) > 0
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_available = sum(1 for r in rows if r["available"])
    print(f"Wrote {out_path}")
    print(f"Available: {n_available}/{len(requested)}")
    for r in rows:
        if not r["available"]:
            print(f"  MISSING: {r['requested_ticker']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
