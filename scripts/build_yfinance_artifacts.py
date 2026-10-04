#!/usr/bin/env python3
"""Build and validate the Yahoo Finance daily dataset artifacts.

Consumes the canonical Parquet files written by ``download_yfinance_daily.py``
and produces, under the same dataset root:

    workbooks/adjusted/Nifty-50.xlsx
    workbooks/unadjusted/Nifty-50.xlsx
    metadata/coverage.csv            per-ticker, per-variant row counts and spans
    metadata/coverage_summary.json
    metadata/data_quality.json       OHLC consistency, nulls, non-positive prices
    metadata/corporate_action_jumps.csv
    metadata/download_summary.json
    manifests/manifest_sha256.csv    sha256 of every delivered file

Every workbook is read back with openpyxl and compared cell-for-cell against
the canonical data before it is accepted.  No values are modified here: the
raw vendor series is reported as-is, including its quirks.

Usage:
    python scripts/build_yfinance_artifacts.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "download_yfinance_daily", REPO_ROOT / "scripts" / "download_yfinance_daily.py"
)
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)

CANONICAL = dl.CANONICAL_COLUMNS
VARIANTS = dl.VARIANTS


def _read(variant: str, ticker: str, root: Path) -> pd.DataFrame:
    return pd.read_parquet(root / variant / "parquet" / f"{ticker}.parquet")


def _tickers(root: Path) -> list[str]:
    files = sorted((root / "adjusted" / "parquet").glob("*.parquet"))
    return [f.stem for f in files]


# ------------------------------------------------------------------ workbook

def build_workbook(root: Path, variant: str, tickers: list[str]) -> dict:
    """One sheet per ticker; Date,Open,High,Low,Close,Volume and no index."""
    from openpyxl import Workbook, load_workbook

    path = root / "workbooks" / variant / "Nifty-50.xlsx"
    wb = Workbook()
    wb.remove(wb.active)

    expected: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = _read(variant, ticker, root)
        expected[dl.excel_sheet_name(ticker)] = df
        ws = wb.create_sheet(title=dl.excel_sheet_name(ticker))
        ws.append(CANONICAL)
        for rec in df.itertuples(index=False):
            ws.append([rec.Date.to_pydatetime(), float(rec.Open), float(rec.High),
                       float(rec.Low), float(rec.Close), int(rec.Volume)])
    wb.save(path)

    # ---- round-trip verification (read back with openpyxl) ----
    check = load_workbook(path, read_only=True, data_only=True)
    problems: list[str] = []
    if sorted(check.sheetnames) != sorted(expected):
        problems.append(f"sheet set mismatch: {sorted(check.sheetnames)}")
    for sheet, df in expected.items():
        if sheet not in check.sheetnames:
            problems.append(f"{sheet}: missing sheet")
            continue
        rows = check[sheet].iter_rows(values_only=True)
        header = next(rows, None)
        if list(header or []) != CANONICAL:
            problems.append(f"{sheet}: header {header} != {CANONICAL}")
            continue
        body = list(rows)
        if len(body) != len(df):
            problems.append(f"{sheet}: {len(body)} rows != {len(df)}")
            continue
        for i, (rec, got) in enumerate(zip(df.itertuples(index=False), body)):
            exp = (rec.Date.to_pydatetime(), float(rec.Open), float(rec.High),
                   float(rec.Low), float(rec.Close), int(rec.Volume))
            if got[0] != exp[0]:
                problems.append(f"{sheet}[{i}]: date {got[0]} != {exp[0]}")
                break
            for j in range(1, 6):
                if abs(float(got[j]) - exp[j]) > 1e-9:
                    problems.append(f"{sheet}[{i}]: col {CANONICAL[j]} "
                                    f"{got[j]} != {exp[j]}")
                    break
            else:
                continue
            break
    check.close()

    return {
        "path": str(path.relative_to(root)),
        "sheets": len(expected),
        "bytes": path.stat().st_size,
        "round_trip_ok": not problems,
        "round_trip_problems": problems[:20],
    }


# ------------------------------------------------------------------- reports

def build_coverage(root: Path, tickers: list[str]) -> tuple[pd.DataFrame, dict]:
    rows = []
    for variant in VARIANTS:
        for ticker in tickers:
            df = _read(variant, ticker, root)
            manifest = json.loads(
                (root / variant / "csv" / f"{ticker}.manifest.json").read_text())
            rows.append({
                "canonical_ticker": ticker,
                "yahoo_symbol": dl.yahoo_symbol(ticker),
                "variant": variant,
                "auto_adjust": variant == "adjusted",
                "rows": len(df),
                "first_date": str(df["Date"].min().date()),
                "last_date": str(df["Date"].max().date()),
                "status": manifest.get("status"),
                "renamed_from": (manifest.get("same_security_rename") or {}).get(
                    "canonical_name", ""),
                "corporate_action_jumps": manifest.get("corporate_action_jumps", 0),
            })
    coverage = pd.DataFrame(rows)
    summary = {
        "requested_tickers": len(tickers),
        "complete_adjusted": int((coverage[coverage.variant == "adjusted"]
                                 .status == "complete").sum()),
        "complete_unadjusted": int((coverage[coverage.variant == "unadjusted"]
                                   .status == "complete").sum()),
        "total_rows_adjusted": int(coverage[coverage.variant == "adjusted"].rows.sum()),
        "total_rows_unadjusted": int(coverage[coverage.variant == "unadjusted"].rows.sum()),
        "earliest_date": coverage.first_date.min(),
        "latest_date": coverage.last_date.max(),
        "failed": coverage[coverage.status != "complete"][
            ["canonical_ticker", "variant", "status"]].to_dict("records"),
    }
    return coverage, summary


def build_quality(root: Path, tickers: list[str]) -> dict:
    out: dict = {
        "definition": {
            "ohlc_consistency": "High >= max(Open,Close) and Low <= min(Open,Close)",
            "non_positive_price": "any of Open/High/Low/Close <= 0",
            "note": "Vendor values are reported as-is; nothing is repaired.",
        },
        "by_variant": {},
    }
    for variant in VARIANTS:
        offenders: list[dict] = []
        nonpos: list[dict] = []
        zero_vol_total = 0
        for ticker in tickers:
            df = _read(variant, ticker, root)
            v = df.dropna(subset=["Open", "High", "Low", "Close"])
            bad = ((v["High"] < v["Low"]) | (v["High"] < v["Open"])
                   | (v["High"] < v["Close"]) | (v["Low"] > v["Open"])
                   | (v["Low"] > v["Close"]))
            if bad.any():
                sub = v[bad]
                offenders.append({
                    "ticker": ticker, "rows": int(bad.sum()),
                    "first": str(sub["Date"].min().date()),
                    "last": str(sub["Date"].max().date()),
                })
            neg = df[(df["Open"] <= 0) | (df["High"] <= 0)
                     | (df["Low"] <= 0) | (df["Close"] <= 0)]
            if len(neg):
                nonpos.append({
                    "ticker": ticker, "rows": len(neg),
                    "first": str(neg["Date"].min().date()),
                    "last": str(neg["Date"].max().date()),
                })
            zero_vol_total += int((df["Volume"] == 0).sum())
        out["by_variant"][variant] = {
            "ohlc_inconsistent_tickers": offenders,
            "non_positive_price_tickers": nonpos,
            "zero_volume_rows": zero_vol_total,
        }
    return out


def build_corporate_actions(root: Path, tickers: list[str]) -> pd.DataFrame:
    frames = []
    for variant in VARIANTS:
        for ticker in tickers:
            df = _read(variant, ticker, root)
            ret = df["Close"].pct_change()
            hit = ret.abs() > dl.JUMP_THRESHOLD
            if not hit.any():
                continue
            sub = df.loc[hit, ["Date", "Close", "Volume"]].copy()
            sub.insert(0, "variant", variant)
            sub.insert(0, "ticker", ticker)
            sub["return"] = ret[hit].values
            sub["abs_return"] = sub["return"].abs()
            frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["ticker", "variant", "Date", "Close",
                                     "Volume", "return", "abs_return"])
    return pd.concat(frames).sort_values("abs_return", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ manifest

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        for sub in ("csv", "parquet", "source_snapshots"):
            for f in sorted((root / variant / sub).iterdir()):
                if f.is_file():
                    rows.append({"path": str(f.relative_to(root)),
                                 "bytes": f.stat().st_size, "sha256": sha256(f)})
        f = root / "workbooks" / variant / "Nifty-50.xlsx"
        if f.is_file():
            rows.append({"path": str(f.relative_to(root)),
                         "bytes": f.stat().st_size, "sha256": sha256(f)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- main

def main() -> int:
    root = dl.dataset_root()
    dl.ensure_layout(root)
    tickers = _tickers(root)
    print(f"dataset root : {root}")
    print(f"tickers      : {len(tickers)}")

    coverage, cov_summary = build_coverage(root, tickers)
    coverage.to_csv(root / "metadata" / "coverage.csv", index=False)
    (root / "metadata" / "coverage_summary.json").write_text(
        json.dumps(cov_summary, indent=2))

    quality = build_quality(root, tickers)
    (root / "metadata" / "data_quality.json").write_text(json.dumps(quality, indent=2))

    actions = build_corporate_actions(root, tickers)
    actions.to_csv(root / "metadata" / "corporate_action_jumps.csv", index=False)

    workbooks = {v: build_workbook(root, v, tickers) for v in VARIANTS}
    for variant, info in workbooks.items():
        print(f"workbook {variant:11}: sheets={info['sheets']} "
              f"round_trip_ok={info['round_trip_ok']} bytes={info['bytes']:,}")
        if info["round_trip_problems"]:
            print("   PROBLEMS:", info["round_trip_problems"][:3])

    manifest = build_manifest(root)
    manifest.to_csv(root / "manifests" / "manifest_sha256.csv", index=False)

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "request": {
            "start": "2000-01-01", "end": "2026-01-01", "end_is_exclusive": True,
            "interval": "1d", "actions": False, "repair": False, "prepost": False,
            "threads": False,
            "adjusted": {"auto_adjust": True, "role": "PRIMARY"},
            "unadjusted": {"auto_adjust": False, "role": "SENSITIVITY"},
        },
        "coverage": cov_summary,
        "workbooks": workbooks,
        "corporate_action_jumps_rows": len(actions),
        "files_hashed": len(manifest),
        "total_bytes": int(manifest.bytes.sum()),
    }
    try:
        import yfinance
        summary["yfinance_version"] = yfinance.__version__
    except Exception as exc:
        summary["yfinance_version"] = f"unavailable ({type(exc).__name__})"
    (root / "metadata" / "download_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"coverage     : {cov_summary['complete_adjusted']}/50 adjusted, "
          f"{cov_summary['complete_unadjusted']}/50 unadjusted, "
          f"{cov_summary['total_rows_adjusted']:,} rows")
    print(f"actions      : {len(actions)} jumps > {dl.JUMP_THRESHOLD:.0%}")
    print(f"manifest     : {len(manifest)} files, {manifest.bytes.sum():,} bytes")
    return 0 if all(w["round_trip_ok"] for w in workbooks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
