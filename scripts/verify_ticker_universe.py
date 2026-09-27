#!/usr/bin/env python3
"""Verify the NIFTY-50 universe against the downloaded raw dataset.

Generates ``results/ticker_availability.csv`` with:
    requested_ticker, discovered_symbol, available, source_file,
    source_start, source_end, fold_0_available, fold_1_available, reason

Applies the reconstruction policy in ``configs/nifty50.yaml``: genuine
symbol-format aliases are honoured; a missing security is reported as
unavailable and never substituted with a different company.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from agentic_forecaster.config import load_config
from agentic_forecaster.data.dataset import discover_ticker_files
from agentic_forecaster.data.universe import load_universe_config, resolve_universe
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
    raw_root = Path(config["data"]["raw_root"])
    discovered = discover_ticker_files(raw_root)

    ticker_path = Path(config["data"]["tickers"])
    if not ticker_path.is_absolute():
        ticker_path = Path(config.get("_config_dir", ".")) / ticker_path
    universe = resolve_universe(load_universe_config(ticker_path), discovered)

    rows = []
    for symbol in universe.requested:
        dataset_symbol = universe.available.get(symbol)
        row = {
            "requested_ticker": symbol,
            "discovered_symbol": dataset_symbol or "",
            "available": dataset_symbol is not None,
            "source_file": "",
            "source_start": "",
            "source_end": "",
            "fold_0_available": False,
            "fold_1_available": False,
            "reason": universe.unavailable.get(symbol, ""),
        }
        if dataset_symbol:
            path = discovered[dataset_symbol]
            row["source_file"] = path.name
            with open(path) as f:
                f.readline()
                first = f.readline().split(",")[0].strip()
                last = ""
                for line in f:
                    if line.strip():
                        last = line.split(",")[0].strip()
            row["source_start"] = first[:10]
            row["source_end"] = last[:10]
            for fold in PAPER_FOLDS:
                # A fold is usable when the test window has intraday coverage.
                row[f"{fold['fold']}_available"] = last[:10] >= fold["test_start"]
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")
    print(f"Available: {universe.n_available}/{universe.n_requested}")
    for symbol, reason in universe.unavailable.items():
        print(f"  UNAVAILABLE: {symbol} — {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
