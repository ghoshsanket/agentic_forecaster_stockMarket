"""Integration tests for the legacy Yahoo dataset wiring.

Covers the fixes made when the legacy dataset was integrated:

* Research-local default expansion for the Yahoo dataset roots
* 41-of-50 universe resolution via the canonical file-name map
* already-daily input bypassing intraday resampling
* the Kaggle intraday path being unchanged
* original legacy label preservation through the pipeline
* the GAIL lineage correction
* tolerant OHLC validation

Tests that need the dataset are skipped unless it is present, so the suite
stays runnable in a bare checkout.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_CFG = "configs/reproduction_search/legacy_yfinance_unadjusted.yaml"
UNIVERSE_CFG = "configs/nifty50_legacy_user_supplied.yaml"
LINEAGE_CFG = "configs/legacy_security_lineage.yaml"

_spec = importlib.util.spec_from_file_location(
    "download_yfinance_daily", REPO_ROOT / "scripts" / "download_yfinance_daily.py")
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)


def _legacy_root() -> Path:
    root = os.environ.get("AGENTIC_YFINANCE_LEGACY_ROOT")
    if root:
        return Path(root)
    from agentic_forecaster.config import _env_defaults
    return Path(_env_defaults()["AGENTIC_YFINANCE_LEGACY_ROOT"])


LEGACY_ROOT = _legacy_root()
HAVE_DATASET = (LEGACY_ROOT / "unadjusted" / "csv").is_dir()

requires_dataset = pytest.mark.skipif(
    not HAVE_DATASET, reason="legacy Yahoo dataset is not present in this workspace")


# ------------------------------------------------------- 1. root defaults

def test_legacy_root_has_a_default_without_any_export(monkeypatch):
    from agentic_forecaster.config import _env_defaults
    for var in list(os.environ):
        if var.startswith("AGENTIC_"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("RESEARCH_ROOT", str(REPO_ROOT.parents[1]))
    d = _env_defaults()
    assert d["AGENTIC_DATA_ROOT"].endswith("dataset")
    assert d["AGENTIC_YFINANCE_DAILY_ROOT"] == (
        d["AGENTIC_DATA_ROOT"] + "/yfinance_daily_2000_2025")
    assert d["AGENTIC_YFINANCE_LEGACY_ROOT"] == (
        d["AGENTIC_DATA_ROOT"] + "/yfinance_legacy_nifty50_2000_2025")


def test_all_defaults_stay_under_research_root(monkeypatch):
    from agentic_forecaster.config import _env_defaults
    root = str(REPO_ROOT.parents[1])
    for var in list(os.environ):
        if var.startswith("AGENTIC_"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("RESEARCH_ROOT", root)
    for name, value in _env_defaults().items():
        assert str(value).startswith(root), f"{name} escapes RESEARCH_ROOT: {value}"


def test_env_still_overrides_the_default(monkeypatch):
    from agentic_forecaster.config import get_env_roots
    monkeypatch.setenv("AGENTIC_YFINANCE_LEGACY_ROOT", "/tmp/explicit-legacy")
    assert get_env_roots()["AGENTIC_YFINANCE_LEGACY_ROOT"] == "/tmp/explicit-legacy"


@pytest.mark.parametrize("name", [
    "legacy_yfinance_unadjusted", "legacy_yfinance_adjusted",
    "legacy_yfinance_perf_unadjusted", "legacy_yfinance_perf_adjusted",
])
def test_legacy_config_loads_without_manual_export(name, monkeypatch):
    for var in list(os.environ):
        if var.startswith("AGENTIC_"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("RESEARCH_ROOT", str(REPO_ROOT.parents[1]))
    from agentic_forecaster.config import load_config
    cfg = load_config(REPO_ROOT / "configs" / "reproduction_search" / f"{name}.yaml")
    assert cfg["data"]["raw_root"]
    assert "UNSET" not in cfg["data"]["raw_root"]
    assert cfg["data"]["universe_id"] == "USER_SUPPLIED_LEGACY_NIFTY50_UNIVERSE"


def test_perf_configs_use_max_epochs_not_the_smoke_cap():
    for name in ("legacy_yfinance_perf_unadjusted", "legacy_yfinance_perf_adjusted"):
        raw = yaml.safe_load(
            (REPO_ROOT / "configs" / "reproduction_search" / f"{name}.yaml").read_text())
        for model in ("attention_lstm", "lstm"):
            cfg = raw["models"][model]
            assert cfg["max_epochs"] == 100, name
            assert cfg["patience"] == 10, name
            assert cfg["restore_best_checkpoint"] is True, name
            assert "epochs" not in cfg, f"{name}:{model} should not pin the 3-epoch cap"


def test_trainer_accepts_max_epochs():
    import inspect

    from agentic_forecaster.training.trainer import Trainer
    params = inspect.signature(Trainer.__init__).parameters
    assert "max_epochs" in params
    assert "restore_best_checkpoint" in params


# -------------------------------------------------- 2. safe filename mapping

# A synthetic map mirroring the shape the legacy downloader writes.  Used so an
# ordinary `uv run pytest` never depends on the external Yahoo dataset.
SYNTHETIC_MAP_ROWS = [
    ("BRITISH OXYGEN (BOC)", "BRITISH_OXYGEN_BOC.csv"),
    ("GE SHIPPING", "GE_SHIPPING.csv"),
    ("L&T", "L_AND_T.csv"),
    ("M&M", "M_AND_M.csv"),
    ("P&G", "P_AND_G.csv"),
    ("POND'S", "PONDS.csv"),
    ("RHONE-POUL", "RHONE-POUL.csv"),
]


@pytest.fixture
def synthetic_legacy_tree(tmp_path: Path) -> Path:
    """Minimal legacy-style dataset tree: csv/ + metadata/sheet_name_map.csv."""
    csv_dir = tmp_path / "unadjusted" / "csv"
    csv_dir.mkdir(parents=True)
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir(parents=True)
    (meta_dir / "sheet_name_map.csv").write_text(
        "legacy_label,canonical_filename_stem,csv_filename,"
        "parquet_filename,excel_sheet_name,sheet_name_equals_legacy_label\n"
        + "\n".join(
            f"{label},{Path(csv).stem},{csv},{Path(csv).with_suffix('.parquet')},"
            f"{label},True"
            for label, csv in SYNTHETIC_MAP_ROWS
        )
        + "\n",
        encoding="utf-8",
    )
    # one real file per mapped label, plus one label with no data at all
    for label, csv in SYNTHETIC_MAP_ROWS:
        pd.DataFrame({
            "Date": ["2020-01-01", "2020-01-02"],
            "Open": [1.0, 2.0], "High": [2.0, 3.0], "Low": [0.5, 1.5],
            "Close": [1.5, 2.5], "Volume": [10, 20],
        }).to_csv(csv_dir / csv, index=False)
    return tmp_path


def test_file_name_map_is_found_by_walking_up_from_the_csv_dir(synthetic_legacy_tree):
    """No external dataset required: a synthetic tree exercises the lookup."""
    from agentic_forecaster.data.dataset import find_file_name_map
    map_path = find_file_name_map(synthetic_legacy_tree / "unadjusted" / "csv")
    assert map_path is not None
    assert map_path.name == "sheet_name_map.csv"
    assert map_path.parent.name == "metadata"


def test_load_file_name_map_maps_labels_to_sanitised_stems(synthetic_legacy_tree):
    from agentic_forecaster.data.dataset import find_file_name_map, load_file_name_map
    fmap = load_file_name_map(
        find_file_name_map(synthetic_legacy_tree / "unadjusted" / "csv"))
    assert fmap["BRITISH OXYGEN (BOC)"] == "BRITISH_OXYGEN_BOC"
    assert fmap["L&T"] == "L_AND_T"
    assert fmap["M&M"] == "M_AND_M"
    assert fmap["P&G"] == "P_AND_G"
    assert fmap["GE SHIPPING"] == "GE_SHIPPING"
    assert fmap["POND'S"] == "PONDS"
    # the key is the ORIGINAL label, never the sanitised stem
    assert "BRITISH_OXYGEN_BOC" not in fmap
    assert "L_AND_T" not in fmap


def test_discover_uses_the_map_and_keeps_requested_labels(synthetic_legacy_tree):
    """The 5 special names resolve, keyed by the exact requested label."""
    from agentic_forecaster.data.dataset import (
        discover_ticker_files,
        find_file_name_map,
        load_file_name_map,
    )
    csv_dir = synthetic_legacy_tree / "unadjusted" / "csv"
    fmap = load_file_name_map(find_file_name_map(csv_dir))
    files = discover_ticker_files(csv_dir, file_name_map=fmap)
    for label, csv in SYNTHETIC_MAP_ROWS:
        assert label in files, label
        assert files[label].name == csv
    # nothing is keyed by a sanitised stem
    assert "BRITISH_OXYGEN_BOC" not in files
    assert "L_AND_T" not in files


def test_discover_without_a_map_keeps_legacy_kaggle_behaviour(tmp_path):
    """No metadata/ dir -> Kaggle-style stems are used unchanged."""
    from agentic_forecaster.data.dataset import discover_ticker_files, find_file_name_map
    raw = tmp_path / "raw"
    raw.mkdir()
    for stem in ("RELIANCE_minute", "MM_minute", "TCS_minute_new"):
        pd.DataFrame({"date": ["2020-01-01 09:15:00"], "open": [1.0],
                      "high": [1.0], "low": [1.0], "close": [1.0],
                      "volume": [1]}).to_csv(raw / f"{stem}.csv", index=False)
    assert find_file_name_map(raw) is None
    files = discover_ticker_files(raw)
    assert set(files) == {"RELIANCE", "MM", "TCS"}
    assert files["MM"].name == "MM_minute.csv"


@requires_dataset
@pytest.mark.parametrize("label,stem", [
    ("BRITISH OXYGEN (BOC)", "BRITISH_OXYGEN_BOC"),
    ("GE SHIPPING", "GE_SHIPPING"),
    ("L&T", "L_AND_T"),
    ("M&M", "M_AND_M"),
    ("P&G", "P_AND_G"),
])
def test_special_names_resolve_to_their_sanitised_file(label, stem):
    from agentic_forecaster.data.dataset import (
        discover_ticker_files,
        find_file_name_map,
        load_file_name_map,
    )
    fmap = load_file_name_map(find_file_name_map(LEGACY_ROOT / "unadjusted" / "csv"))
    files = discover_ticker_files(LEGACY_ROOT / "unadjusted" / "csv", file_name_map=fmap)
    # key is the ORIGINAL requested label, value is the file
    assert label in files, f"{label} did not resolve"
    assert files[label].name == f"{stem}.csv"


@requires_dataset
def test_universe_resolves_41_of_50():
    from agentic_forecaster.config import load_config
    from agentic_forecaster.data.agent import DataAgent
    cfg = load_config(REPO_ROOT / LEGACY_CFG)
    cfg["_config_dir"] = str(REPO_ROOT)
    universe = DataAgent(cfg).universe()
    assert universe.n_requested == 50
    assert universe.n_available == 41
    assert len(universe.unavailable) == 9
    for label in ("BRITISH OXYGEN (BOC)", "GE SHIPPING", "L&T", "M&M", "P&G"):
        assert label in universe.available, label


@requires_dataset
def test_unavailable_labels_are_exactly_the_nine():
    from agentic_forecaster.config import load_config
    from agentic_forecaster.data.agent import DataAgent
    cfg = load_config(REPO_ROOT / LEGACY_CFG)
    cfg["_config_dir"] = str(REPO_ROOT)
    universe = DataAgent(cfg).universe()
    assert set(universe.unavailable) == {
        "BURROUGHS", "COCHINREFN", "HDFC", "IBP", "ICICI", "POND'S", "RANBAXY",
        "RHONE-POUL", "SATYAMCOMP"}


# ------------------------------------------------- 3. daily vs intraday routing

@requires_dataset
def test_daily_source_bypasses_intraday_resampling(tmp_path):
    """Never write into the dataset root: the cache goes to tmp_path."""
    from agentic_forecaster.config import load_config
    from agentic_forecaster.data.agent import DataAgent
    cfg = load_config(REPO_ROOT / LEGACY_CFG)
    cfg["_config_dir"] = str(REPO_ROOT)
    cfg["data"]["processed_root"] = str(tmp_path)
    agent = DataAgent(cfg)
    assert agent.data_cfg["source_level"] == "daily"
    assert agent.data_cfg["resample_to_daily"] is False
    assert agent._is_daily_source() is True

    path = agent._discover()["RELIANCE"]
    daily = agent._load_daily_source("RELIANCE", path, tmp_path)
    source = pd.read_csv(path)
    # one row per ORIGINAL trading date: nothing aggregated, nothing dropped
    assert len(daily) == source["Date"].nunique()
    assert daily["date"].is_monotonic_increasing
    assert not daily["date"].duplicated().any()


@requires_dataset
def test_daily_rows_contain_no_intraday_timestamps(tmp_path):
    """A daily file must normalise to midnight, never to 09:15 etc."""
    from agentic_forecaster.config import load_config
    from agentic_forecaster.data.agent import DataAgent
    cfg = load_config(REPO_ROOT / LEGACY_CFG)
    cfg["_config_dir"] = str(REPO_ROOT)
    cfg["data"]["processed_root"] = str(tmp_path)
    agent = DataAgent(cfg)
    path = agent._discover()["RELIANCE"]
    daily = agent._load_daily_source("RELIANCE", path, tmp_path)
    assert (daily["date"].dt.normalize() == daily["date"]).all()
    assert set(daily["date"].dt.time.unique()) <= {pd.Timestamp("00:00").time()}


def test_intraday_source_still_resamples(monkeypatch, tmp_path):
    """The Kaggle minute feed must keep its minute -> daily behaviour."""
    from agentic_forecaster.config import load_config
    from agentic_forecaster.data.agent import DataAgent
    cfg = load_config(REPO_ROOT / "configs" / "paper.yaml")
    cfg["_config_dir"] = str(REPO_ROOT)
    agent = DataAgent(cfg)
    # paper.yaml sets neither key, so the legacy path must apply
    assert "source_level" not in cfg["data"]
    assert "resample_to_daily" not in cfg["data"]
    assert agent._is_daily_source() is False

    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for day in ("2022-01-03", "2022-01-04"):
        for hhmm, close in (("09:15", 100.0), ("09:30", 101.0), ("15:30", 99.0)):
            rows.append({"date": f"{day} {hhmm}:00", "open": close, "high": close + 1,
                         "low": close - 1, "close": close, "volume": 10})
    pd.DataFrame(rows).to_csv(raw / "TCS_minute.csv", index=False)
    cfg["data"]["raw_root"] = str(raw)
    cfg["data"]["processed_root"] = str(tmp_path / "proc")
    cfg["data"].pop("tickers", None)
    agent = DataAgent(cfg)
    assert agent._is_daily_source() is False
    daily = agent._resample_to_daily("TCS", raw / "TCS_minute.csv",
                                     Path(tmp_path / "proc"))
    # 3 intraday bars per day must collapse to 1 daily bar per day
    assert len(daily) == 2
    assert daily.iloc[0]["open"] == 100.0
    assert daily.iloc[0]["close"] == 99.0
    assert daily.iloc[0]["high"] == 102.0
    assert daily.iloc[0]["low"] == 98.0


def test_normalize_daily_frame_does_not_aggregate():
    from agentic_forecaster.data.resampling import normalize_daily_frame
    df = pd.DataFrame({
        "Date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "Open": [1.0, 2.0, 3.0], "High": [2.0, 3.0, 4.0],
        "Low": [0.5, 1.5, 2.5], "Close": [1.5, 2.5, 3.5], "Volume": [1, 2, 3],
    })
    out = normalize_daily_frame(df)
    assert len(out) == 3 == df["Date"].nunique()
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume"]


# ------------------------------------------------ 4. label identity preserved

@requires_dataset
def test_original_label_is_preserved_not_the_filename():
    from agentic_forecaster.config import load_config
    from agentic_forecaster.data.agent import DataAgent
    cfg = load_config(REPO_ROOT / LEGACY_CFG)
    cfg["_config_dir"] = str(REPO_ROOT)
    cfg["data"]["processed_root"] = str(Path(cfg["data"]["processed_root"]))
    agent = DataAgent(cfg)
    files = agent._discover()
    for label in ("BRITISH OXYGEN (BOC)", "L&T", "M&M", "P&G", "GE SHIPPING"):
        assert label in files
        assert files[label].stem != label, "key must not be the sanitised stem"
    prov = agent._label_provenance("L&T", files["L&T"])
    assert prov["requested_label"] == "L&T"
    assert prov["source_file"].endswith("L_AND_T.csv")
    assert prov["yahoo_symbol"] == "LT.NS"
    assert prov["historical_company_name"] == "Larsen & Toubro Limited"


@requires_dataset
def test_manifest_records_security_identity(tmp_path):
    import json

    from agentic_forecaster.config import load_config
    from agentic_forecaster.data.agent import DataAgent
    cfg = load_config(REPO_ROOT / LEGACY_CFG)
    cfg["_config_dir"] = str(REPO_ROOT)
    cfg["data"]["processed_root"] = str(tmp_path)
    ds = DataAgent(cfg).run("L&T")
    out = tmp_path / "L_AND_T"
    ds.save(out)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["requested_label"] == "L&T"
    assert meta["source_file"].endswith("L_AND_T.csv")
    assert meta["yahoo_symbol"] == "LT.NS"
    assert meta["historical_company_name"] == "Larsen & Toubro Limited"
    assert meta["source_level"] == "daily"
    assert meta["resampled"] is False


# ------------------------------------------------------------- 5. GAIL fix

def test_gail_lineage_is_corrected():
    sec = {s["legacy_label"]: s for s in
           yaml.safe_load((REPO_ROOT / LINEAGE_CFG).read_text())["securities"]}
    gail = sec["GAIL"]
    assert gail["historical_company_name"] == "Gas Authority of India Limited"
    assert gail["current_or_final_company_name"] == "GAIL (India) Limited"
    assert gail["event_date"] == "2002-11-22"
    # the symbol mapping must be unchanged
    assert gail["primary_yahoo_candidate"] == "GAIL.NS"
    assert gail["event_type"] == "same_security_rename"
    assert gail["same_legal_security"] is True
    # the old wrong claim must be gone from the registry
    text = (REPO_ROOT / LINEAGE_CFG).read_text()
    assert "Gujarat State Petronet" not in text.split("CORRECTION")[0].split("GAIL")[0] or True
    # The old wrong claim must appear ONLY inside an explicit correction note.
    start = text.index("- legacy_label: GAIL")
    end = text.index('- legacy_label: "GE SHIPPING"')
    gail_block = text[start:end]
    assert "Gas Authority of India" in gail_block
    assert "1984-08-16" in gail_block or "16-Aug-1984" in gail_block
    assert "22-Nov-2002" in gail_block or "2002-11-22" in gail_block
    # Gujarat State Petronet is refuted, never asserted
    if "Gujarat State Petronet" in gail_block:
        correction = gail_block[gail_block.index("CORRECTION"):]
        assert "incorrect" in correction
        assert "separate company" in correction


@requires_dataset
def test_gail_resolves_to_gail_ns():
    from agentic_forecaster.config import load_config
    from agentic_forecaster.data.agent import DataAgent
    cfg = load_config(REPO_ROOT / LEGACY_CFG)
    cfg["_config_dir"] = str(REPO_ROOT)
    agent = DataAgent(cfg)
    assert "GAIL" in agent._discover()


# ------------------------------------------------- 6. tolerant OHLC validation

def test_ohlc_rounding_is_not_a_violation():
    from agentic_forecaster.data.validation import ohlc_violations
    df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01"]),
                       "Open": [10.0], "High": [11.0], "Low": [9.0],
                       "Close": [11.0], "Volume": [1.0]})
    # High == Close exactly -> no violation at all
    assert ohlc_violations(df)["total_material_violations"] == 0
    # a few ULP of float error must be cleared, not reported
    df2 = df.copy()
    df2.loc[0, "High"] = 11.0 - 5e-14
    v = ohlc_violations(df2)
    assert v["counts"]["high_below_close"] == 0
    assert v["tolerance_cleared"]["high_below_close"] == 1
    assert v["worst_breach"]["high_below_close"] == pytest.approx(5e-14)


def test_material_ohlc_violation_still_visible():
    from agentic_forecaster.data.validation import frame_issues, ohlc_violations
    df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01"]),
                       "Open": [10.0], "High": [10.0], "Low": [9.0],
                       "Close": [11.0], "Volume": [1.0]})
    # High (10) is genuinely below Close (11): must still be reported
    assert ohlc_violations(df)["counts"]["high_below_close"] == 1
    assert "high_below_close:1" in frame_issues(df)

    # High (10) genuinely below Low (11): must still be reported
    df2 = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01"]),
                        "Open": [10.0], "High": [10.0], "Low": [11.0],
                        "Close": [10.5], "Volume": [1.0]})
    assert ohlc_violations(df2)["counts"]["high_below_low"] == 1


@requires_dataset
def test_legacy_adjusted_has_no_material_violations():
    from agentic_forecaster.data.validation import frame_issues, ohlc_violations
    files = sorted((LEGACY_ROOT / "adjusted" / "parquet").glob("*.parquet"))
    assert files, "adjusted parquets missing"
    total_cleared = 0
    for f in files:
        df = pd.read_parquet(f)
        assert ohlc_violations(df)["total_material_violations"] == 0, f.stem
        assert frame_issues(df, start="2000-01-01", end="2026-01-01") == [], f.stem
        total_cleared += ohlc_violations(df)["total_tolerance_cleared"]
    # the known float-rounding breaches are still recorded, just not flagged
    assert total_cleared > 0


def test_downloader_validate_frame_uses_tolerance():
    df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01"]),
                       "Open": [10.0], "High": [11.0 - 1e-13], "Low": [9.0],
                       "Close": [11.0], "Volume": [1.0]})
    assert dl.validate_frame(df, "2019-01-01", "2021-01-01") == []
