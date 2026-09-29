#!/usr/bin/env python3
"""Download daily NIFTY-50 OHLCV from Yahoo Finance via yfinance.

Reconstructs the original project dataset organisation:

    yfinance_daily_2000_2025/
        adjusted/    {csv,parquet,source_snapshots}   auto_adjust=True   (PRIMARY)
        unadjusted/  {csv,parquet,source_snapshots}   auto_adjust=False  (sensitivity)
        workbooks/{adjusted,unadjusted}/Nifty-50.xlsx
        metadata/  manifests/  logs/  cache/  tmp/

Canonical output columns are EXACTLY: Date,Open,High,Low,Close,Volume
`Adj Close`, Dividends and Stock Splits are never written to the canonical
CSV/Parquet/workbook; the unadjusted variant keeps them only in the immutable
source snapshot.

The script is resumable: a ticker whose manifest records the same requested
parameters and whose file validates is skipped unless --force is given.
Partial downloads are written to a temporary name and atomically renamed, so
an interrupted run can never be mistaken for a complete one.

Usage
-----
    python scripts/download_yfinance_daily.py \
        --start 2000-01-01 --end 2026-01-01 --variant both --resume --all

No features, targets or models are computed here.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

CANONICAL_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
VARIANTS = ("adjusted", "unadjusted")
AUTO_ADJUST = {"adjusted": True, "unadjusted": False}
# Columns that must never reach the canonical output.
NON_CANONICAL = ("Adj Close", "Dividends", "Stock Splits", "Capital Gains")

MAX_ATTEMPTS = 5
BACKOFF_SECONDS = (2, 5, 10, 20, 40)
INTER_SYMBOL_DELAY = 1.0
JUMP_THRESHOLD = 0.25  # 25% absolute one-day return

# Same-security ticker renames, with the evidence that authorises each mapping.
# A rename is NOT a substitution: the legal entity, ISIN and price history are
# continuous.  Anything not listed here is treated as genuinely unavailable and
# is never replaced by a different company.
#
# ZOMATO -> ETERNAL:  NSE circular CML67420, "name and symbol of Zomato Limited
# will be changed w.e.f. April 09, 2025" (name change approved by the Ministry of
# Corporate Affairs).  Zomato Ltd became Eternal Ltd on both NSE and BSE; the
# Yahoo series is continuous, starts 2021-07-23 (IPO era) and Yahoo reports
# longName "Eternal Limited" on NSE/INR, same Consumer Cyclical / Internet Retail
# profile.  Daily price action across 2025-04-09 is continuous (-1.77%, no split).
RENAMED_TICKERS = {
    "ZOMATO": {
        "yahoo_symbol": "ETERNAL.NS",
        "canonical_name": "Eternal Limited (formerly Zomato Limited)",
        "reason": "Same-security rename; NSE circular CML67420 effective 2025-04-09.",
    },
}

logger = logging.getLogger("yfinance_daily")


# ---------------------------------------------------------------- paths

def dataset_root(override: str | None = None) -> Path:
    """Resolve the dataset root.

    Precedence: explicit ``--dataset-root`` > ``AGENTIC_YFINANCE_DAILY_ROOT``
    (the modern-universe root) > ``$AGENTIC_DATA_ROOT/yfinance_daily_2000_2025``.
    """
    if override:
        return Path(override)
    if os.environ.get("AGENTIC_YFINANCE_DAILY_ROOT"):
        return Path(os.environ["AGENTIC_YFINANCE_DAILY_ROOT"])
    research_root = os.environ.get("RESEARCH_ROOT")
    data_root = os.environ.get("AGENTIC_DATA_ROOT") or (
        str(Path(research_root) / "dataset") if research_root
        else str(REPO_ROOT.parents[1] / "dataset")
    )
    return Path(data_root) / "yfinance_daily_2000_2025"


def ensure_layout(root: Path) -> None:
    for variant in VARIANTS:
        for sub in ("csv", "parquet", "source_snapshots"):
            (root / variant / sub).mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        (root / "workbooks" / variant).mkdir(parents=True, exist_ok=True)
    for sub in ("metadata", "manifests", "logs", "tmp"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    # yfinance's persistent cache must stay inside the Research workspace.
    (root / "cache" / "yfinance").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- symbols

def load_universe(repo_root: Path, config_path: str | None = None) -> list[str]:
    """Load the security list from a universe config.

    Default (``None``) preserves the original modern-universe behaviour and
    reads ``configs/nifty50.yaml``.  Labels are returned exactly as spelled in
    the config, because the legacy universe must preserve its original
    human-readable labels.
    """
    path = Path(config_path) if config_path else repo_root / "configs" / "nifty50.yaml"
    if not path.is_absolute():
        candidate = repo_root / path
        path = candidate if candidate.exists() else path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [str(t) for t in raw.get("tickers", [])]


def load_lineage(repo_root: Path, lineage_path: str | None) -> dict[str, dict]:
    """Load the corporate-identity registry, if one is configured.

    Returns a mapping of legacy_label -> entry.  An empty dict means no
    registry, in which case the modern ``RENAMED_TICKERS`` behaviour applies.
    """
    if not lineage_path:
        return {}
    path = Path(lineage_path)
    if not path.is_absolute():
        candidate = repo_root / path
        path = candidate if candidate.exists() else path
    if not path.is_file():
        raise FileNotFoundError(f"lineage registry not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("stitch_successors_into_canonical"):
        raise ValueError(
            "lineage registry sets stitch_successors_into_canonical=true; "
            "successor prices must never be appended to a historical series")
    return {str(s["legacy_label"]): s for s in raw.get("securities", [])}


def safe_filename(label: str) -> str:
    """Filesystem-safe canonical name that preserves the label as closely as
    possible.  ``BRITISH OXYGEN (BOC)`` -> ``BRITISH_OXYGEN_BOC``;
    ``L&T`` -> ``L_AND_T``; ``P&G`` -> ``P_AND_G``; ``POND'S`` -> ``PONDS``."""
    name = label.strip()
    name = name.replace("&", "_AND_")
    name = name.replace("'", "").replace("(", "").replace(")", "")
    name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    name = name.replace(".", "_").replace(",", "_").replace(":", "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_") or "UNNAMED"


def yahoo_symbol(canonical: str, lineage: dict[str, dict] | None = None) -> str:
    """Resolve a canonical project ticker to its Yahoo Finance symbol.

    Precedence:

    0. a corporate-identity registry entry (configs/legacy_security_lineage.yaml)
       -- authoritative for the legacy universe, and the ONLY source that may
       map a legacy label onto a differently-named listed company
    1. an authorised same-security rename (``RENAMED_TICKERS``)
    2. a symbol that is already a Yahoo symbol (has a dot, a ``^`` index prefix
       or an ``=`` cross suffix)
    3. otherwise the NSE form ``<SYMBOL>.NS``

    A registry entry whose ``primary_download_policy`` is ``do_not_download``
    yields an empty string: that legacy security has no legitimate same-security
    Yahoo history and MUST NOT be substituted.

    Punctuation is preserved verbatim: ``M&M`` -> ``M&M.NS``,
    ``BAJAJ-AUTO`` -> ``BAJAJ-AUTO.NS``.
    """
    if lineage:
        entry = lineage.get(canonical)
        if entry is not None:
            if entry.get("primary_download_policy") == "do_not_download":
                return ""
            return str(entry.get("primary_yahoo_candidate") or "")
    rename = RENAMED_TICKERS.get(canonical)
    if rename is not None:
        return rename["yahoo_symbol"]
    if "." in canonical or canonical.startswith("^") or "=" in canonical:
        return canonical
    return f"{canonical}.NS"


# Excel sheet names are capped at 31 characters and cannot contain []:*?/ or a
# backslash.
_EXCEL_ILLEGAL = set("[]:*?/\\")


def excel_sheet_name(ticker: str) -> str:
    """Sheet naming for the MODERN universe (unchanged historical behaviour)."""
    name = ticker.replace("&", "-").replace("/", "-")
    return name[:31]


def legacy_sheet_name(label: str) -> str:
    """Sheet name for a legacy label, preserving the EXACT label when Excel allows.

    Every one of the 50 user-supplied labels is already a legal Excel sheet
    name (max 31 chars, no ``[]:*?/\\``), so the labels are used verbatim and
    ``metadata/sheet_name_map.csv`` records the mapping.  Sanitisation is kept
    only as a defensive fallback.
    """
    if len(label) <= 31 and not (_EXCEL_ILLEGAL & set(label)):
        return label
    return safe_filename(label)[:31]


# ------------------------------------------------------- canonicalisation

def canonicalize(df: pd.DataFrame, *, tz_strip: bool = True) -> pd.DataFrame:
    """Reduce a yfinance frame to exactly the canonical OHLCV columns.

    * index -> tz-naive ascending ``Date`` column
    * drops Adj Close / Dividends / Stock Splits / Capital Gains
    * drops duplicate dates (keeping the first observation)
    * never forward-fills, never invents rows
    """
    out = df.copy()

    if isinstance(out.index, pd.DatetimeIndex):
        if tz_strip and out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        out = out.reset_index()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"])
        if getattr(out["Date"].dt, "tz", None) is not None:
            out["Date"] = out["Date"].dt.tz_localize(None)

    for col in NON_CANONICAL:
        if col in out.columns:
            out = out.drop(columns=[col])

    missing = [c for c in CANONICAL_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"missing expected columns {missing}; got {list(out.columns)}")

    out = out[CANONICAL_COLUMNS]
    out = out.sort_values("Date").reset_index(drop=True)
    out = out.drop_duplicates(subset=["Date"], keep="first").reset_index(drop=True)

    for col in ("Open", "High", "Low", "Close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce")
    return out


def validate_frame(df: pd.DataFrame, start: str, end: str) -> list[str]:
    """Return a list of quality problems; empty means the frame is acceptable."""
    issues: list[str] = []
    if df.empty:
        return ["empty_frame"]
    if not df["Date"].is_monotonic_increasing:
        issues.append("date_not_monotonic")
    if df["Date"].duplicated().any():
        issues.append("duplicate_dates")
    if df["Date"].min() < pd.Timestamp(start):
        issues.append(f"date_before_start:{df['Date'].min().date()}")
    # end is exclusive in the request, so the last row must be strictly before it.
    if df["Date"].max() >= pd.Timestamp(end):
        issues.append(f"date_on_or_after_end:{df['Date'].max().date()}")
    for col in CANONICAL_COLUMNS[1:]:
        if df[col].isna().any():
            issues.append(f"null_in_{col}")
    if (df["Volume"] < 0).any():
        issues.append("negative_volume")
    valid = df.dropna(subset=["Open", "High", "Low", "Close"])
    if (valid["High"] < valid["Low"]).any():
        issues.append("high_below_low")
    if (valid["High"] < valid["Open"]).any():
        issues.append("high_below_open")
    if (valid["High"] < valid["Close"]).any():
        issues.append("high_below_close")
    if (valid["Low"] > valid["Open"]).any():
        issues.append("low_above_open")
    if (valid["Low"] > valid["Close"]).any():
        issues.append("low_above_close")
    return issues


def corporate_action_jumps(df: pd.DataFrame) -> pd.DataFrame:
    """Flag |one-day return| > 25% without modifying the data."""
    ret = df["Close"].pct_change()
    hits = df.loc[ret.abs() > JUMP_THRESHOLD, ["Date", "Close"]].copy()
    if not hits.empty:
        hits["abs_return"] = ret[ret.abs() > JUMP_THRESHOLD].abs().values
    return hits


# --------------------------------------------------------------- download

def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _read_saved(variant: str, ticker: str, root: Path) -> pd.DataFrame | None:
    path = root / variant / "parquet" / f"{safe_filename(ticker)}.parquet"
    if not path.is_file():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _saved_matches(root: Path, ticker: str, variant: str, start: str, end: str) -> bool:
    """True when a completed file for these exact parameters already exists."""
    stem = safe_filename(ticker)
    manifest = root / variant / "csv" / f"{stem}.manifest.json"
    parquet = root / variant / "parquet" / f"{stem}.parquet"
    csv_file = root / variant / "csv" / f"{stem}.csv"
    if not (manifest.is_file() and parquet.is_file() and csv_file.is_file()):
        return False
    try:
        meta = json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return False
    if meta.get("status") != "complete":
        return False
    if meta.get("start") != start or meta.get("end") != end:
        return False
    if meta.get("auto_adjust") is not AUTO_ADJUST[variant]:
        return False
    return meta.get("interval") == "1d"


def fetch_symbol(symbol: str, *, start: str, end: str, auto_adjust: bool):
    """Download one symbol with conservative exponential backoff.

    Returns the raw yfinance DataFrame.  An empty frame is never success.
    """
    import yfinance as yf

    last_error: str = "unknown"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=auto_adjust,
                actions=False,
                progress=False,
                threads=False,
                repair=False,
                prepost=False,
                multi_level_index=False,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
            last_error = "empty_dataframe"
            logger.warning("[%s] attempt %d/%d returned empty data", symbol, attempt, MAX_ATTEMPTS)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("[%s] attempt %d/%d failed: %s", symbol, attempt, MAX_ATTEMPTS, exc)
        if attempt < MAX_ATTEMPTS:
            time.sleep(BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)])
    raise RuntimeError(f"download failed after {MAX_ATTEMPTS} attempts: {last_error}")


def download_ticker(ticker: str, symbol: str, variant: str, root: Path,
                    start: str, end: str, force: bool, resume: bool,
                    lineage: dict[str, dict] | None = None) -> dict:
    """Download and persist one ticker for one variant. Returns a status dict."""
    stem = safe_filename(ticker)
    csv_path = root / variant / "csv" / f"{stem}.csv"
    manifest_path = root / variant / "csv" / f"{stem}.manifest.json"
    snapshot_path = root / variant / "source_snapshots" / f"{stem}.parquet"
    parquet_path = root / variant / "parquet" / f"{stem}.parquet"

    if resume and not force and _saved_matches(root, ticker, variant, start, end):
        df = _read_saved(variant, ticker, root)
        return {
            "status": "skipped_cached",
            "rows": len(df) if df is not None else 0,
            "first": str(df["Date"].min().date()) if df is not None and not df.empty else None,
            "last": str(df["Date"].max().date()) if df is not None and not df.empty else None,
            "issues": [],
        }

    record: dict = {
        "canonical_ticker": ticker,
        "yahoo_symbol": symbol,
        "variant": variant,
        "auto_adjust": AUTO_ADJUST[variant],
        "interval": "1d",
        "start": start,
        "end": end,
        "actions": False,
        "repair": False,
        "prepost": False,
        "threads": False,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
    }
    rename = RENAMED_TICKERS.get(ticker)
    if rename is not None:
        record["same_security_rename"] = {
            "canonical_name": rename["canonical_name"],
            "reason": rename["reason"],
        }
    lineage_entry = (lineage or {}).get(ticker)
    if lineage_entry is not None:
        record["legacy_label"] = ticker
        record["canonical_filename_stem"] = stem
        record["historical_company_name"] = lineage_entry.get("historical_company_name")
        record["current_or_final_company_name"] = lineage_entry.get("current_or_final_company_name")
        record["event_type"] = lineage_entry.get("event_type")
        record["same_legal_security"] = lineage_entry.get("same_legal_security")
        record["successor_security"] = lineage_entry.get("successor_security")
        record["event_date"] = lineage_entry.get("event_date")
        record["confidence"] = lineage_entry.get("confidence")
        # Explicitly record that no successor price was ever appended.
        record["successor_prices_appended"] = False

    try:
        raw = fetch_symbol(symbol, start=start, end=end,
                           auto_adjust=AUTO_ADJUST[variant])
    except Exception as exc:
        record.update(status="failed", reason=str(exc))
        manifest_path.write_text(json.dumps(record, indent=2))
        return {"status": "failed", "reason": str(exc), "rows": 0,
                "first": None, "last": None, "issues": []}

    # Immutable provenance snapshot: the immediate yfinance frame, untouched.
    try:
        snap_tmp = snapshot_path.with_suffix(f".parquet.tmp{os.getpid()}")
        raw.to_parquet(snap_tmp)
        os.replace(snap_tmp, snapshot_path)
    except Exception as exc:
        logger.warning("[%s/%s] source snapshot failed: %s", ticker, variant, exc)

    canon = canonicalize(raw)
    issues = validate_frame(canon, start, end)
    if "empty_frame" in issues:
        record.update(status="failed", reason="empty_after_canonicalization")
        manifest_path.write_text(json.dumps(record, indent=2))
        return {"status": "failed", "reason": "empty_after_canonicalization",
                "rows": 0, "first": None, "last": None, "issues": issues}

    _atomic_write_csv(canon, csv_path)
    _atomic_write_parquet(canon, parquet_path)

    jumps = corporate_action_jumps(canon)
    record.update(
        status="complete",
        rows=len(canon),
        first=str(canon["Date"].min().date()),
        last=str(canon["Date"].max().date()),
        issues=issues,
        corporate_action_jumps=len(jumps),
        snapshot=str(snapshot_path.relative_to(root)),
    )
    manifest_path.write_text(json.dumps(record, indent=2))

    return {
        "status": "complete",
        "rows": len(canon),
        "first": record["first"],
        "last": record["last"],
        "issues": issues,
    }


# ------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2026-01-01",
                        help="yfinance end boundary is EXCLUSIVE")
    parser.add_argument("--variant", choices=(*VARIANTS, "both"), default="both")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ticker", default=None, help="single canonical ticker")
    parser.add_argument("--all", action="store_true", help="all configured securities")
    parser.add_argument(
        "--universe-config", default=None,
        help="universe YAML to read (default: configs/nifty50.yaml, the modern "
             "reconstructed universe). Use configs/nifty50_legacy_user_supplied.yaml "
             "for the legacy universe.")
    parser.add_argument(
        "--lineage-config", default=None,
        help="corporate-identity registry YAML. Required for universes whose "
             "labels are not tickers. Enables lineage-aware symbol resolution "
             "and suppresses do_not_download securities.")
    parser.add_argument(
        "--dataset-root", default=None,
        help="override the dataset root (default: $AGENTIC_YFINANCE_DAILY_ROOT). "
             "Point this at $AGENTIC_YFINANCE_LEGACY_ROOT for the legacy run so "
             "the existing modern dataset is never touched.")
    args = parser.parse_args()

    root = dataset_root(args.dataset_root)
    ensure_layout(root)
    lineage = load_lineage(REPO_ROOT, args.lineage_config)
    if args.lineage_config:
        logger.info("lineage registry: %s (%d securities)", args.lineage_config, len(lineage))

    # Redirect the yfinance persistent cache BEFORE any request.
    import yfinance as yf
    cache_dir = root / "cache" / "yfinance"
    yf.set_tz_cache_location(str(cache_dir))

    log_path = root / "logs" / "download.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    logger.info("yfinance %s | root=%s", yf.__version__, root)
    logger.info("cache location -> %s", cache_dir)
    logger.info("request: start=%s end=%s (exclusive) interval=1d variant=%s",
                args.start, args.end, args.variant)

    universe = load_universe(REPO_ROOT, args.universe_config)
    logger.info("universe config=%s | %d securities",
                args.universe_config or "configs/nifty50.yaml", len(universe))
    tickers = [args.ticker] if args.ticker else universe

    variants = list(VARIANTS) if args.variant == "both" else [args.variant]

    results: dict[str, dict] = {}
    for variant in variants:
        for i, ticker in enumerate(tickers, 1):
            symbol = yahoo_symbol(ticker, lineage or None)
            if not symbol:
                # No legitimate same-security Yahoo history.  Record and move on.
                # NOTHING is written, nothing is substituted, nothing invented.
                results.setdefault(ticker, {})[variant] = {
                    "status": "unavailable_from_yahoo", "reason":
                        "lineage registry marks this security do_not_download; no "
                        "same-security Yahoo history exists and no successor is "
                        "appended", "rows": 0, "first": None, "last": None,
                    "issues": [],
                }
                logger.info("[%d/%d] %-20s -> UNAVAILABLE_FROM_YAHOO (no substitution)",
                            i, len(tickers), ticker)
                continue
            res = download_ticker(ticker, symbol, variant, root,
                                 args.start, args.end,
                                 force=args.force, resume=args.resume,
                                 lineage=lineage or None)
            results.setdefault(ticker, {})[variant] = res
            logger.info("[%d/%d] %-20s file=%-22s yahoo=%-18s adj=%-8s unadj=%-8s "
                        "rows=%s %s..%s",
                        i, len(tickers), ticker, safe_filename(ticker), symbol,
                        results[ticker].get("adjusted", {}).get("status", "-"),
                        results[ticker].get("unadjusted", {}).get("status", "-"),
                        res.get("rows"), res.get("first"), res.get("last"))
            time.sleep(INTER_SYMBOL_DELAY)

    # Persist the raw outcome for downstream steps.
    (root / "metadata" / "download_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    logger.info("download results -> %s", root / "metadata" / "download_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
