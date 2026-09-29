#!/usr/bin/env python3
"""Probe Yahoo Finance for historical symbols of the legacy universe.

This is a FORENSIC step that runs BEFORE any canonical download.  For every
legacy label it tries a list of historically plausible Yahoo symbols and
records what Yahoo actually returns: row count, first/last date, company name,
quote type, currency and exchange.  The output drives
``configs/legacy_security_lineage.yaml``; nothing is accepted or rejected on
the basis of name similarity alone.

NSE (.NS) is the primary exchange because this is a NIFTY universe.  BSE (.BO)
may be probed as a secondary fallback, but BSE-only availability is recorded as
``BSE_AVAILABLE_ONLY`` and is NEVER merged into an NSE series.

Writes ``metadata/yahoo_symbol_probe.csv``.

Usage:
    python scripts/probe_legacy_yahoo_symbols.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

PROBE_START = "1995-01-01"   # probe earlier than requested download to see true start
PROBE_END = "2026-01-01"

PROBE_COLUMNS = [
    "legacy_label", "candidate_symbol", "exchange", "returned_rows",
    "first_date", "last_date", "yahoo_long_name", "yahoo_short_name",
    "quote_type", "currency", "exchange_name", "accepted", "reason",
]


def legacy_root() -> Path:
    import os
    return Path(os.environ.get(
        "AGENTIC_YFINANCE_LEGACY_ROOT",
        str(Path(os.environ["AGENTIC_DATA_ROOT"]) / "yfinance_legacy_nifty50_2000_2025")))


def candidate_symbols(label: str) -> list[tuple[str, str]]:
    """(symbol, exchange) candidates for a legacy label, primary NSE first."""
    base = {
        "ACC": ["ACC"], "BAJAJ-AUTO": ["BAJAJ-AUTO"], "BPCL": ["BPCL"],
        "BRITANNIA": ["BRITANNIA"], "BRITISH OXYGEN (BOC)": ["BOCINDIA", "LINDEINDIA", "LINDE"],
        "BSES": ["BSES"], "BURROUGHS": ["BURROUGHS", "BURROUGHSWELL"],
        "CIPLA": ["CIPLA"], "COCHINREFN": ["COCHINREFN", "KOCHIREF", "KERALAKERALA"],
        "COLPAL": ["COLPAL"], "DRREDDY": ["DRREDDY"], "EPL": ["EPL", "ESSEPRO"],
        "GAIL": ["GAIL"], "GE SHIPPING": ["GESHIP"], "GLAXO": ["GLAXO", "GSK"],
        "GRASIM": ["GRASIM"], "GUTRLCEMENT": ["AMBUJACEM", "GUJRATCEM", "AMBUJA"],
        "HDFC": ["HDFC", "HDFCBANK"], "HDFCBANK": ["HDFCBANK"],
        "HINDALCO": ["HINDALCO", "NOVALIS"], "HINDLEVER": ["HINDUNILVR", "HINDLEVER"],
        "HPCL": ["HINDPETRO", "HPCL"], "IBP": ["IBP", "IBPIND", "IBPIL"],
        "ICICI": ["ICICI", "ICICIBANK"], "IDBI": ["IDBI"],
        "IFCI": ["IFCI"], "INDIACEM": ["INDIACEM"],
        "INFOSYSTCH": ["INFY", "INFOSYSTCH", "INFY-OR", "INFOSYS"],
        "IOC": ["IOC"], "ITC": ["ITC"], "KNOLLPHARM": ["ABBOTINDIA", "KNOLLPHARM", "ABBOT"],
        "L&T": ["LT", "L&T"], "M&M": ["M&M", "MM"], "MADRASCEM": ["RAMCOCEM", "MADRASCEM", "RAMCO"],
        "MTNL": ["MTNL"], "NESTLEIND": ["NESTLEIND"], "NIIT": ["NIITLTD", "NIIT", "NIITTECH"],
        "NOCIL": ["NOCIL"], "ONGC": ["ONGC"],
        "P&G": ["PGHH", "P&G", "PGHHN"], "POND'S": ["PONDS", "POND", "HINDUNILVR"],
        "RANBAXY": ["RANBAXY", "RANBAXYLAB", "SUNPHARMA"],
        "RELIANCE": ["RELIANCE", "RCOM"], "RHONE-POUL": ["RHONEPOL", "RHONEP", "RPIL", "SANOFI", "PIRAMALFIN"],
        "SATYAMCOMP": ["SATYAMCOMP", "SAYAM", "TECHM", "MAHINDSAT"],
        "SBI": ["SBIN", "SBI"], "TATACHEM": ["TATACHEM"],
        "TELCO": ["TATAMOTORS", "TELCO", "TATAMTRDVR", "TAMOEHIND"],
        "TISCO": ["TATASTEEL", "TISCO"], "VSTILL": ["VSTIND", "VSTILL", "VST"],
    }.get(label, [label])
    out: list[tuple[str, str]] = []
    for b in base:
        out.append((f"{b}.NS", "NSE"))
    return out


def probe(symbol: str) -> dict:
    import yfinance as yf
    rec = {"returned_rows": 0, "first_date": "", "last_date": "",
           "yahoo_long_name": "", "yahoo_short_name": "", "quote_type": "",
           "currency": "", "exchange_name": ""}
    try:
        df = yf.download(symbol, start=PROBE_START, end=PROBE_END, interval="1d",
                         auto_adjust=False, actions=False, progress=False,
                         threads=False, repair=False, prepost=False,
                         multi_level_index=False)
    except Exception as exc:
        rec["reason"] = f"error:{type(exc).__name__}"
        return rec
    if not isinstance(df, pd.DataFrame) or df.empty:
        rec["reason"] = "no_data"
        return rec
    rec["returned_rows"] = len(df)
    rec["first_date"] = str(df.index.min().date())
    rec["last_date"] = str(df.index.max().date())
    try:
        info = yf.Ticker(symbol).get_info() or {}
        rec["yahoo_long_name"] = str(info.get("longName") or "")
        rec["yahoo_short_name"] = str(info.get("shortName") or "")
        rec["quote_type"] = str(info.get("quoteType") or "")
        rec["currency"] = str(info.get("currency") or "")
        rec["exchange_name"] = str(info.get("exchangeName") or info.get("fullExchangeName") or "")
    except Exception as exc:
        # Company metadata is supplementary; a failure here must not
        # invalidate an otherwise good price series.
        rec.setdefault("reason_detail", type(exc).__name__)
    rec["reason"] = "has_data"
    return rec


def main() -> int:
    root = legacy_root()
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    labels = yaml.safe_load(
        (REPO_ROOT / "configs" / "nifty50_legacy_user_supplied.yaml").read_text())["tickers"]

    import yfinance as yf
    yf.set_tz_cache_location(str(root / "cache" / "yfinance"))

    rows = []
    for i, label in enumerate(labels, 1):
        seen = []
        for symbol, exch in candidate_symbols(label):
            rec = probe(symbol)
            row = {"legacy_label": label, "candidate_symbol": symbol, "exchange": exch,
                   **rec, "accepted": "PENDING_REVIEW"}
            rows.append(row)
            seen.append(f"{symbol}:{rec['returned_rows']}")
            print(f"[{i:2d}/50] {label:20s} {symbol:16s} rows={rec['returned_rows']:>6} "
                  f"{rec['first_date']}..{rec['last_date']} {rec['reason']}")
            if rec["returned_rows"] > 0:
                break  # first hit for this label is the primary candidate
            time.sleep(0.4)
        time.sleep(0.4)

    out = root / "metadata" / "yahoo_symbol_probe.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PROBE_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in PROBE_COLUMNS})
    print(f"\nwrote {out}  ({len(rows)} probe rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
