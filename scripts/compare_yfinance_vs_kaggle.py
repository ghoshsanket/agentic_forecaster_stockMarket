#!/usr/bin/env python3
"""Compare the Yahoo Finance reconstruction against the existing Kaggle data.

The Kaggle source used by the earlier Phase-1 work is *minute* level, so the
comparison resamples it to daily with the project's own
``resample_intraday_to_daily`` rather than a fresh aggregation.  That keeps the
two sides derived by the same logic and makes any difference attributable to
the data, not to the resampler.

Two comparisons are produced for each security:

    adjusted   Yahoo auto_adjust=True   vs  Kaggle daily (unadjusted OHLCV)
    unadjusted Yahoo auto_adjust=False  vs  Kaggle daily (unadjusted OHLCV)

The adjusted/unadjusted gap inside Yahoo is reported as a third block, because
it is the quantity that actually depends on the reconstruction choice.

Nothing in the Kaggle dataset is modified.

Usage:
    python scripts/compare_yfinance_vs_kaggle.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.data.resampling import resample_intraday_to_daily
from agentic_forecaster.data.universe import load_universe_config

_spec = importlib.util.spec_from_file_location(
    "download_yfinance_daily", REPO_ROOT / "scripts" / "download_yfinance_daily.py"
)
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)

# Representative, liquid, long-history names (required set first).
FOCUS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC",
         "SBIN", "LT", "HINDUNILVR", "ICICIBANK", "AXISBANK", "TITAN", "MARUTI"]

# Kaggle vendor spelling for the same security (Kaggle used the MM alias).
KAGGLE_ALIAS = {"M&M": "MM"}


def kaggle_file(raw_root: Path, ticker: str) -> Path | None:
    name = KAGGLE_ALIAS.get(ticker, ticker)
    for candidate in (f"{name}_minute.csv", f"{name}_daily.csv"):
        p = raw_root / candidate
        if p.is_file():
            return p
    return None


def kaggle_daily(raw_root: Path, ticker: str) -> pd.DataFrame | None:
    path = kaggle_file(raw_root, ticker)
    if path is None:
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    daily = resample_intraday_to_daily(df, date_col="date")
    daily = daily.rename(columns={c: c.capitalize() for c in daily.columns})
    if "Date" not in daily.columns:
        for cand in ("Date", "Datetime"):
            if cand in daily.columns:
                daily = daily.rename(columns={cand: "Date"})
    daily["Date"] = pd.to_datetime(daily["Date"]).dt.normalize()
    return daily[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna(
        subset=["Close"]).reset_index(drop=True)


def yahoo_daily(root: Path, ticker: str, variant: str) -> pd.DataFrame | None:
    p = root / variant / "parquet" / f"{ticker}.parquet"
    return pd.read_parquet(p) if p.is_file() else None


def _align(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    m = a.merge(b, on="Date", how="inner", suffixes=("_a", "_b"))
    return m.dropna(subset=["Close_a", "Close_b"])


def _stats(m: pd.DataFrame, price_cols=("Open", "High", "Low", "Close")) -> dict:
    if m.empty:
        return {"overlapping_days": 0}
    out: dict = {"overlapping_days": len(m)}
    out["max_abs_close_diff"] = float((m["Close_a"] - m["Close_b"]).abs().max())
    out["mean_abs_close_diff"] = float((m["Close_a"] - m["Close_b"]).abs().mean())
    denom = m["Close_b"].abs().replace(0, np.nan)
    rel = ((m["Close_a"] - m["Close_b"]).abs() / denom).dropna()
    out["max_rel_close_diff_pct"] = float(rel.max() * 100) if len(rel) else None
    out["mean_rel_close_diff_pct"] = float(rel.mean() * 100) if len(rel) else None
    exact = 0.0
    for c in price_cols:
        exact += float((m[f"{c}_a"] - m[f"{c}_b"]).abs().max())
    out["max_abs_diff_across_ohlc"] = exact
    va, vb = m["Volume_a"], m["Volume_b"]
    out["volume_exact_match_pct"] = float((va == vb).mean() * 100)
    return out


def main() -> int:
    root = dl.dataset_root()
    raw_root = Path(__import__("os").environ["AGENTIC_RAW_DATA_ROOT"])
    universe = load_universe_config(REPO_ROOT / "configs" / "nifty50.yaml").requested
    focus = [t for t in FOCUS if t in universe]

    comparisons: list[dict] = []
    for ticker in focus:
        kg = kaggle_daily(raw_root, ticker)
        if kg is None:
            comparisons.append({"ticker": ticker, "status": "no_kaggle_file"})
            continue
        entry: dict = {"ticker": ticker, "status": "compared",
                       "kaggle_rows": len(kg),
                       "kaggle_span": f"{kg['Date'].min().date()}..{kg['Date'].max().date()}"}
        y_adj = yahoo_daily(root, ticker, "adjusted")
        y_un = yahoo_daily(root, ticker, "unadjusted")
        for variant, frame in (("adjusted", y_adj), ("unadjusted", y_un)):
            if frame is None:
                entry[f"yahoo_{variant}_vs_kaggle"] = {"status": "missing"}
                continue
            m = _align(frame, kg)
            entry[f"yahoo_{variant}"] = {
                "rows": len(frame),
                "span": f"{frame['Date'].min().date()}..{frame['Date'].max().date()}"}
            entry[f"yahoo_{variant}_vs_kaggle"] = _stats(m)
        if y_adj is not None and y_un is not None:
            mu = _align(y_adj, y_un)
            entry["yahoo_adjusted_vs_unadjusted"] = {
                ** _stats(mu),
                "interpretation": "difference is the dividend/split adjustment; "
                                  "unadjusted equals the vendor raw close",
            }
        comparisons.append(entry)

    out_dir = root / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = []
    for c in comparisons:
        for key in ("yahoo_adjusted_vs_kaggle", "yahoo_unadjusted_vs_kaggle",
                    "yahoo_adjusted_vs_unadjusted"):
            stats = c.get(key) or {}
            flat.append({"ticker": c["ticker"], "comparison": key,
                         **{k: v for k, v in stats.items()
                            if k != "interpretation"}})
    pd.DataFrame(flat).to_csv(out_dir / "yfinance_vs_kaggle.csv", index=False)

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "kaggle_source": str(raw_root),
        "kaggle_level": "minute, resampled to daily with the project resampler",
        "yahoo_root": str(root),
        "focus_tickers": focus,
        "comparisons": comparisons,
    }
    (out_dir / "yfinance_vs_kaggle.json").write_text(json.dumps(report, indent=2))

    print(f"{'ticker':11} {'overlap':>8} {'adj close diff':>15} {'unadj close diff':>18} "
          f"{'adj-vs-unadj':>14}")
    for c in comparisons:
        a = c.get("yahoo_adjusted_vs_kaggle", {})
        u = c.get("yahoo_unadjusted_vs_kaggle", {})
        d = c.get("yahoo_adjusted_vs_unadjusted", {})
        print(f"{c['ticker']:11} {a.get('overlapping_days', 0):8d} "
              f"{_f(a.get('max_abs_close_diff')):>15} "
              f"{_f(u.get('max_abs_close_diff')):>18} "
              f"{_f(d.get('max_abs_close_diff')):>14}")
    print(f"\nwrote {out_dir / 'yfinance_vs_kaggle.csv'}")
    return 0


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
