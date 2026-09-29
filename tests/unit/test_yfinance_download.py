"""Offline tests for the Yahoo Finance daily reconstruction downloader.

These tests never touch the network.  They exercise the pure transformations
(canonicalisation, validation, corporate-action diagnostics, symbol mapping)
and the resume/atomic-write bookkeeping using temporary directories.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "download_yfinance_daily.py"


def _load():
    spec = importlib.util.spec_from_file_location("download_yfinance_daily", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dl = _load()


def _fake_yf(adj_close: bool = False, dup_date: bool = False) -> pd.DataFrame:
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    if dup_date:
        dates.append("2020-01-03")
    idx = pd.DatetimeIndex(dates, name="Date")
    data = {
        "Open": [100.0, 101.0, 102.0, 102.0][: len(dates)],
        "High": [105.0, 106.0, 107.0, 107.0][: len(dates)],
        "Low": [95.0, 96.0, 97.0, 97.0][: len(dates)],
        "Close": [101.0, 103.0, 104.0, 104.0][: len(dates)],
        "Volume": [1000, 2000, 3000, 3000][: len(dates)],
    }
    if adj_close:
        data["Adj Close"] = [99.0, 101.0, 102.0, 102.0][: len(dates)]
    return pd.DataFrame(data, index=idx)


# ------------------------------------------------------------------ symbols

def test_yahoo_symbol_adds_ns_suffix():
    assert dl.yahoo_symbol("RELIANCE") == "RELIANCE.NS"


def test_yahoo_symbol_preserves_punctuation():
    # Do not mangle '&' or '-': these are the real Yahoo NSE symbols.
    assert dl.yahoo_symbol("M&M") == "M&M.NS"
    assert dl.yahoo_symbol("BAJAJ-AUTO") == "BAJAJ-AUTO.NS"


def test_yahoo_symbol_leaves_explicit_symbol():
    # Index / cross symbols are already explicit Yahoo symbols.
    assert dl.yahoo_symbol("^NSEI") == "^NSEI"
    assert dl.yahoo_symbol("EURUSD=X") == "EURUSD=X"
    assert dl.yahoo_symbol("RELIANCE.NS") == "RELIANCE.NS"


def test_excel_sheet_name_is_excel_safe():
    assert dl.excel_sheet_name("M&M") == "M-M"
    assert len(dl.excel_sheet_name("A" * 40)) == 31


# ----------------------------------------------------------- canonical form

def test_canonicalize_exact_columns_and_order():
    out = dl.canonicalize(_fake_yf())
    assert list(out.columns) == dl.CANONICAL_COLUMNS


def test_canonicalize_drops_adj_close():
    out = dl.canonicalize(_fake_yf(adj_close=True))
    assert "Adj Close" not in out.columns
    assert list(out.columns) == dl.CANONICAL_COLUMNS


def test_canonicalize_drops_dividends_and_splits():
    df = _fake_yf()
    df["Dividends"] = 1.0
    df["Stock Splits"] = 0.0
    out = dl.canonicalize(df)
    assert "Dividends" not in out.columns
    assert "Stock Splits" not in out.columns
    assert list(out.columns) == dl.CANONICAL_COLUMNS


def test_canonicalize_sorts_and_dedupes():
    out = dl.canonicalize(_fake_yf(dup_date=True))
    assert out["Date"].is_monotonic_increasing
    assert not out["Date"].duplicated().any()
    assert len(out) == 3


def test_canonicalize_strips_timezone():
    df = _fake_yf()
    df.index = df.index.tz_localize("Asia/Kolkata")
    out = dl.canonicalize(df)
    assert out["Date"].dt.tz is None


def test_canonicalize_raises_on_missing_column():
    df = _fake_yf().drop(columns=["Volume"])
    with pytest.raises(ValueError):
        dl.canonicalize(df)


# ------------------------------------------------------------- validation

def test_validate_frame_clean():
    assert dl.validate_frame(dl.canonicalize(_fake_yf()), "2019-01-01", "2021-01-01") == []


def test_validate_frame_flags_empty():
    assert "empty_frame" in dl.validate_frame(pd.DataFrame(), "2019-01-01", "2021-01-01")


def test_validate_frame_flags_end_boundary_violation():
    # end is exclusive, so a row exactly on the boundary is a violation.
    df = pd.DataFrame(
        {"Date": [pd.Timestamp("2021-01-01")], "Open": [1.0], "High": [1.0],
         "Low": [1.0], "Close": [1.0], "Volume": [1]}
    )
    issues = dl.validate_frame(df, "2019-01-01", "2021-01-01")
    assert any("date_on_or_after_end" in i for i in issues)


def test_validate_frame_flags_high_below_low():
    df = pd.DataFrame(
        {"Date": [pd.Timestamp("2020-01-01")], "Open": [1.0], "High": [0.5],
         "Low": [2.0], "Close": [1.0], "Volume": [1]}
    )
    # a 1.5-unit breach is material, so it must still be reported
    issues = dl.validate_frame(df, "2019-01-01", "2021-01-01")
    assert any(i.startswith("high_below_low") for i in issues)


def test_validate_frame_flags_negative_volume():
    df = pd.DataFrame(
        {"Date": [pd.Timestamp("2020-01-01")], "Open": [1.0], "High": [1.0],
         "Low": [1.0], "Close": [1.0], "Volume": [-1]}
    )
    assert "negative_volume" in dl.validate_frame(df, "2019-01-01", "2021-01-01")


# ------------------------------------------------- corporate action jumps

def test_corporate_action_jumps_detects_large_move():
    df = pd.DataFrame(
        {"Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
         "Open": [1.0, 1.0, 2.0], "High": [1.0, 1.0, 2.0],
         "Low": [1.0, 1.0, 2.0], "Close": [100.0, 100.0, 200.0],
         "Volume": [1, 1, 1]}
    )
    jumps = dl.corporate_action_jumps(df)
    assert len(jumps) == 1
    assert jumps["Date"].iloc[0] == pd.Timestamp("2020-01-03")


def test_corporate_action_jumps_ignores_normal_moves():
    df = pd.DataFrame(
        {"Date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
         "Open": [1.0, 1.0], "High": [1.0, 1.0], "Low": [1.0, 1.0],
         "Close": [100.0, 101.0], "Volume": [1, 1]}
    )
    assert dl.corporate_action_jumps(df).empty


# ------------------------------------------------- resume / atomic writing

def test_saved_matches_requires_complete_manifest(tmp_path):
    root = tmp_path
    for variant in dl.VARIANTS:
        dl.ensure_layout(root)
    csv_path = root / "adjusted" / "csv" / "TCS.csv"
    pq_path = root / "adjusted" / "parquet" / "TCS.parquet"
    manifest = root / "adjusted" / "csv" / "TCS.manifest.json"
    csv_path.write_text("a,b\n1,2\n")
    pq_path.write_bytes(b"parquet")
    manifest.write_text(json.dumps({
        "status": "complete", "start": "2000-01-01", "end": "2026-01-01",
        "auto_adjust": True, "interval": "1d",
    }))
    assert dl._saved_matches(root, "TCS", "adjusted", "2000-01-01", "2026-01-01")
    # Different window / variant must not be treated as cached.
    assert not dl._saved_matches(root, "TCS", "adjusted", "2015-01-01", "2026-01-01")
    assert not dl._saved_matches(root, "TCS", "unadjusted", "2000-01-01", "2026-01-01")


def test_saved_matches_rejects_incomplete(tmp_path):
    root = tmp_path
    dl.ensure_layout(root)
    (root / "adjusted" / "csv" / "TCS.csv").write_text("a\n1\n")
    (root / "adjusted" / "parquet" / "TCS.parquet").write_bytes(b"x")
    (root / "adjusted" / "csv" / "TCS.manifest.json").write_text(
        json.dumps({"status": "failed", "start": "2000-01-01", "end": "2026-01-01",
                    "auto_adjust": True, "interval": "1d"}))
    assert not dl._saved_matches(root, "TCS", "adjusted", "2000-01-01", "2026-01-01")


def test_atomic_write_leaves_no_tmp(tmp_path):
    target = tmp_path / "x.csv"
    dl._atomic_write_csv(pd.DataFrame({"a": [1]}), target)
    assert target.is_file()
    assert not list(tmp_path.glob("*.tmp*"))


def test_resume_skips_completed_ticker(tmp_path, monkeypatch):
    root = tmp_path
    dl.ensure_layout(root)
    canon = dl.canonicalize(_fake_yf())
    dl._atomic_write_csv(canon, root / "adjusted" / "csv" / "TCS.csv")
    dl._atomic_write_parquet(canon, root / "adjusted" / "parquet" / "TCS.parquet")
    (root / "adjusted" / "csv" / "TCS.manifest.json").write_text(json.dumps({
        "status": "complete", "start": "2000-01-01", "end": "2026-01-01",
        "auto_adjust": True, "interval": "1d",
    }))

    def _boom(*a, **k):  # pragma: no cover - should never be called
        raise AssertionError("network fetch called during resume")

    monkeypatch.setattr(dl, "fetch_symbol", _boom)
    res = dl.download_ticker("TCS", "TCS.NS", "adjusted", root,
                             "2000-01-01", "2026-01-01", force=False, resume=True)
    assert res["status"] == "skipped_cached"
    assert res["rows"] == len(canon)


def test_download_ticker_writes_all_artifacts(tmp_path, monkeypatch):
    """Exercises the real write path: snapshot, csv, parquet, manifest.

    Regression guard: this reaches manifest construction, so a stale datetime
    import (ruff UP017 rewriting the import but not the usage) fails here
    instead of only in a live download.
    """
    root = tmp_path
    dl.ensure_layout(root)
    seen: dict = {}

    def _fake_fetch(symbol, *, start, end, auto_adjust):
        seen.update(symbol=symbol, start=start, end=end, auto_adjust=auto_adjust)
        return _fake_yf(adj_close=not auto_adjust)

    monkeypatch.setattr(dl, "fetch_symbol", _fake_fetch)
    res = dl.download_ticker("TCS", "TCS.NS", "adjusted", root,
                             "2000-01-01", "2026-01-01", force=True, resume=False)

    assert res["status"] == "complete", res
    assert res["rows"] == 3
    assert res["issues"] == []
    assert seen == {"symbol": "TCS.NS", "start": "2000-01-01",
                     "end": "2026-01-01", "auto_adjust": True}

    for rel in ("adjusted/csv/TCS.csv", "adjusted/parquet/TCS.parquet",
                "adjusted/source_snapshots/TCS.parquet",
                "adjusted/csv/TCS.manifest.json"):
        assert (root / rel).is_file(), rel

    csv_df = pd.read_csv(root / "adjusted" / "csv" / "TCS.csv")
    assert list(csv_df.columns) == dl.CANONICAL_COLUMNS
    assert not list((root / "adjusted").rglob("*.tmp*"))

    meta = json.loads((root / "adjusted" / "csv" / "TCS.manifest.json").read_text())
    assert meta["status"] == "complete"
    assert meta["auto_adjust"] is True
    assert meta["interval"] == "1d"
    assert meta["actions"] is False and meta["repair"] is False
    assert meta["prepost"] is False and meta["threads"] is False
    assert "downloaded_at_utc" in meta


def test_rename_mapping_recorded_in_manifest(tmp_path, monkeypatch):
    """A same-security rename must be recorded, not silently applied."""
    root = tmp_path
    dl.ensure_layout(root)
    monkeypatch.setattr(dl, "fetch_symbol",
                        lambda *a, **k: _fake_yf())
    res = dl.download_ticker("ZOMATO", dl.yahoo_symbol("ZOMATO"), "adjusted", root,
                             "2000-01-01", "2026-01-01", force=True, resume=False)
    assert res["status"] == "complete"
    meta = json.loads((root / "adjusted" / "csv" / "ZOMATO.manifest.json").read_text())
    assert meta["yahoo_symbol"] == "ETERNAL.NS"
    assert "same_security_rename" in meta
    assert "Eternal" in meta["same_security_rename"]["canonical_name"]


# ------------------------------------------------------------------ layout

def test_ensure_layout_creates_all_required_dirs(tmp_path):
    root = tmp_path
    dl.ensure_layout(root)
    for rel in ("adjusted/csv", "adjusted/parquet", "adjusted/source_snapshots",
                "unadjusted/csv", "unadjusted/parquet", "unadjusted/source_snapshots",
                "workbooks/adjusted", "workbooks/unadjusted",
                "metadata", "manifests", "logs", "cache/yfinance", "tmp"):
        assert (root / rel).is_dir(), rel


def test_dataset_root_prefers_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTIC_YFINANCE_DAILY_ROOT", str(tmp_path / "yf"))
    assert dl.dataset_root() == tmp_path / "yf"


def test_dataset_root_derives_from_data_root(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENTIC_YFINANCE_DAILY_ROOT", raising=False)
    monkeypatch.setenv("AGENTIC_DATA_ROOT", str(tmp_path))
    assert dl.dataset_root() == tmp_path / "yfinance_daily_2000_2025"


# ----------------------------------------------------------------- network

@pytest.mark.network
def test_live_yahoo_reachable():
    """Opt in with AGENTIC_RUN_NETWORK_TESTS=1."""
    if os.environ.get("AGENTIC_RUN_NETWORK_TESTS") != "1":
        pytest.skip("set AGENTIC_RUN_NETWORK_TESTS=1 to run network tests")
    df = dl.fetch_symbol("RELIANCE.NS", start="2000-01-01", end="2026-01-01",
                         auto_adjust=True)
    canon = dl.canonicalize(df)
    assert not canon.empty
    assert list(canon.columns) == dl.CANONICAL_COLUMNS
    assert canon["Date"].min() >= pd.Timestamp("2000-01-01")
    assert canon["Date"].max() < pd.Timestamp("2026-01-01")
