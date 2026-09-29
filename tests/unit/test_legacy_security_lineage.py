"""Corporate-identity and no-stitching tests for the LEGACY NIFTY-50 universe.

These tests are the guard rail for the dataset's central scientific claim: a
SUCCESSOR COMPANY IS NOT AUTOMATICALLY THE SAME SECURITY.  They assert both
directions:

  * the verified same-security renames DO resolve to their current symbol, and
  * the merged/acquired securities DO NOT resolve to their successor, and are
    never silently substituted.

No network access is required.  The lineage registry is a static, reviewed
artefact; these tests pin its content and the downloader's use of it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_CFG = REPO_ROOT / "configs" / "nifty50_legacy_user_supplied.yaml"
LINEAGE_CFG = REPO_ROOT / "configs" / "legacy_security_lineage.yaml"

_spec = importlib.util.spec_from_file_location(
    "download_yfinance_daily", REPO_ROOT / "scripts" / "download_yfinance_daily.py")
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)

UNIVERSE = yaml.safe_load(UNIVERSE_CFG.read_text())
LABELS: list[str] = UNIVERSE["tickers"]
LINEAGE = {s["legacy_label"]: s for s in yaml.safe_load(LINEAGE_CFG.read_text())["securities"]}

ALLOWED_EVENT_TYPES = {
    "unchanged", "same_security_rename", "statutory_continuation", "demerger",
    "merger_into_other_company", "acquired_and_merged", "delisted", "unresolved",
}


def sym(label: str) -> str:
    return dl.yahoo_symbol(label, LINEAGE)


# ------------------------------------------------------- universe integrity

def test_universe_has_exactly_50_labels():
    assert len(LABELS) == 50


def test_universe_id_and_window():
    assert UNIVERSE["universe_id"] == "USER_SUPPLIED_LEGACY_NIFTY50_UNIVERSE"
    assert UNIVERSE["requested_download_start"] == "2000-01-01"
    assert UNIVERSE["requested_download_end_exclusive"] == "2026-01-01"


def test_legacy_labels_preserved_exactly_and_in_order():
    """The user's list is authoritative and must not be normalised."""
    expected = [
        "ACC", "BAJAJ-AUTO", "BPCL", "BRITANNIA", "BRITISH OXYGEN (BOC)", "BSES",
        "BURROUGHS", "CIPLA", "COCHINREFN", "COLPAL", "DRREDDY", "EPL", "GAIL",
        "GE SHIPPING", "GLAXO", "GRASIM", "GUTRLCEMENT", "HDFC", "HDFCBANK",
        "HINDALCO", "HINDLEVER", "HPCL", "IBP", "ICICI", "IDBI", "IFCI", "INDIACEM",
        "INFOSYSTCH", "IOC", "ITC", "KNOLLPHARM", "L&T", "M&M", "MADRASCEM", "MTNL",
        "NESTLEIND", "NIIT", "NOCIL", "ONGC", "P&G", "POND'S", "RANBAXY", "RELIANCE",
        "RHONE-POUL", "SATYAMCOMP", "SBI", "TATACHEM", "TELCO", "TISCO", "VSTILL",
    ]
    assert LABELS == expected


def test_lineage_covers_every_label_exactly_once():
    assert set(LINEAGE) == set(LABELS)
    assert len(LINEAGE) == 50


def test_event_types_are_in_the_controlled_vocabulary():
    for label, e in LINEAGE.items():
        assert e["event_type"] in ALLOWED_EVENT_TYPES, label


def test_registry_forbids_successor_stitching():
    assert yaml.safe_load(LINEAGE_CFG.read_text())["stitch_successors_into_canonical"] is False


def test_do_not_download_labels_have_no_primary_symbol():
    for label, e in LINEAGE.items():
        if e["primary_download_policy"] == "do_not_download":
            assert e["primary_yahoo_candidate"] == "", label
            assert sym(label) == "", label


# ------------------------------------------------- verified same-security renames

@pytest.mark.parametrize("label,expected", [
    ("BRITISH OXYGEN (BOC)", "LINDEINDIA.NS"),
    ("BSES", "RELINFRA.NS"),
    ("HINDLEVER", "HINDUNILVR.NS"),
    ("INFOSYSTCH", "INFY.NS"),
    ("KNOLLPHARM", "ABBOTINDIA.NS"),
    ("GUTRLCEMENT", "AMBUJACEM.NS"),
    ("MADRASCEM", "RAMCOCEM.NS"),
    ("TELCO", "TMPV.NS"),
    ("TISCO", "TATASTEEL.NS"),
    ("HPCL", "HINDPETRO.NS"),
    ("L&T", "LT.NS"),
    ("SBI", "SBIN.NS"),
])
def test_verified_mappings_resolve(label, expected):
    """Same legal security -> the verified current symbol is used."""
    assert sym(label) == expected
    assert LINEAGE[label]["primary_download_policy"].startswith("download_primary")


def test_ticker_format_labels_are_same_entity_renames_not_mergers():
    for label in ("HPCL", "L&T", "IDBI"):
        assert LINEAGE[label]["event_type"] in {
            "unchanged", "same_security_rename", "statutory_continuation"}


def test_knollpharm_is_a_round_trip_rename_not_a_merger():
    """Abbott India was renamed TO Knoll, then renamed BACK. Never a merger."""
    e = LINEAGE["KNOLLPHARM"]
    assert e["event_type"] == "same_security_rename"
    assert e["same_legal_security"] is True


# ------------------------------------------------------ NO-STITCH regression

@pytest.mark.parametrize("label,forbidden", [
    ("BURROUGHS", "GLAXO"),
    ("COCHINREFN", "BPCL"),
    ("IBP", "IOC"),
    ("ICICI", "ICICIBANK"),
    ("HDFC", "HDFCBANK"),
    ("RANBAXY", "SUNPHARMA"),
    ("SATYAMCOMP", "TECHM"),
    ("POND'S", "HINDUNILVR"),
    ("RHONE-POUL", "SANOFI"),
])
def test_successor_is_never_substituted(label, forbidden):
    """A merged/acquired security must NOT resolve to its successor."""
    e = LINEAGE[label]
    assert e["primary_download_policy"] == "do_not_download"
    assert e["primary_yahoo_candidate"] == ""
    assert e["same_legal_security"] is False
    # The successor may be RECORDED for a future experiment, but must never be
    # the download target, and must not appear in the probed candidates.
    assert forbidden not in [c.upper() for c in (e.get("additional_yahoo_candidates") or [])]
    assert forbidden not in sym(label).upper()
    # the successor is still documented
    assert e["successor_security"]


def test_hdfc_and_hdfcbank_are_distinct_securities():
    """The 1977 HDFC Ltd and the 1994 HDFC Bank were separate listings."""
    assert sym("HDFC") == ""
    assert sym("HDFCBANK") == "HDFCBANK.NS"
    assert LINEAGE["HDFC"]["successor_security"] == "HDFCBANK.NS (HDFC Bank Ltd)"


def test_hindunilvr_is_its_own_security_not_a_ponds_stand_in():
    assert sym("HINDLEVER") == "HINDUNILVR.NS"
    assert sym("POND'S") == ""


def test_glaxo_is_its_own_security_not_a_burroughs_stand_in():
    assert sym("GLAXO") == "GLAXO.NS"
    assert sym("BURROUGHS") == ""


# ------------------------------------------------ BAJAJ-AUTO demerger regression

def test_bajaj_auto_is_a_demerger_not_a_rename():
    """Requirement 24: the 2008 Bajaj event must NOT be a same_security_rename."""
    e = LINEAGE["BAJAJ-AUTO"]
    assert e["event_type"] == "demerger"
    assert e["event_type"] != "same_security_rename"


def test_bajaj_auto_has_false_legal_continuity():
    """The listed scrip continued, but it is NOT one unchanged legal security."""
    e = LINEAGE["BAJAJ-AUTO"]
    assert e["same_legal_security"] is False
    assert e["primary_download_policy"] == "download_primary_flagged"


def test_bajaj_auto_names_both_successor_entities():
    succ = LINEAGE["BAJAJ-AUTO"]["successor_security"]
    assert "BHIL" in succ  # the renamed erstwhile BAJAJ-AUTO
    assert "BAJAJFSL" in succ  # Bajaj Finserv


# ----------------------------------------------- RHONE-POUL regression test

def test_rhone_poul_is_not_mapped_to_sanofi():
    """Requirement 25: the global Rhone-Poulenc lineage is IRRELEVANT."""
    assert sym("RHONE-POUL") == ""
    entry = LINEAGE["RHONE-POUL"]
    assert entry["event_type"] == "acquired_and_merged"
    assert "SANOFI" not in entry["primary_yahoo_candidate"].upper()
    assert "SANOFI" not in str(entry["successor_security"]).upper()
    assert "SANOFI" not in " ".join(entry["additional_yahoo_candidates"]).upper()


def test_rhone_poul_is_documented_against_nicholas_piramal():
    """The Indian listed company went to Nicholas Piramal India, not Sanofi."""
    e = LINEAGE["RHONE-POUL"]
    text = (e["evidence_notes"] + " " + str(e["successor_security"])).lower()
    assert "piramal" in text
    assert e["event_date"].startswith("2001")


# ------------------------------------------------------------- IDBI & POND'S

def test_idbi_is_statutory_continuation():
    e = LINEAGE["IDBI"]
    assert e["event_type"] == "statutory_continuation"
    assert e["event_type"] != "same_security_rename"
    assert sym("IDBI") == "IDBI.NS"


def test_ponds_date_anomaly_is_documented_not_deleted():
    uni = UNIVERSE
    assert "POND'S" in uni["tickers"]  # NOT deleted
    assert "POND'S" in uni["date_anomaly_labels"]
    rec = uni["user_recollection"]
    assert rec["intended_snapshot_date"] == "2000-01-03"
    assert rec["date_confidence"] == "needs_independent_verification"
    assert rec["date_anomaly_detected"] is True
    # Pond's had already gone in 1998, BEFORE the recalled 2000-01-03 date.
    assert LINEAGE["POND'S"]["event_date"].startswith("1998")
    assert "1998" in uni["user_recollection"]["date_anomaly_detail"]


# ----------------------------------------------------- naming / filesystem

def test_safe_filenames_are_filesystem_safe():
    for label in LABELS:
        name = dl.safe_filename(label)
        assert name and not any(c in name for c in r'\/:*?"<>|')
        assert name == name.strip()


@pytest.mark.parametrize("label,expected", [
    ("BRITISH OXYGEN (BOC)", "BRITISH_OXYGEN_BOC"),
    ("GE SHIPPING", "GE_SHIPPING"),
    ("L&T", "L_AND_T"),
    ("P&G", "P_AND_G"),
    ("POND'S", "PONDS"),
    ("RHONE-POUL", "RHONE-POUL"),
])
def test_expected_safe_filenames(label, expected):
    assert dl.safe_filename(label) == expected


def test_every_label_is_a_valid_excel_sheet_name_used_verbatim():
    """All 50 supplied labels are legal Excel names, so none is rewritten."""
    for label in LABELS:
        assert dl.legacy_sheet_name(label) == label


def test_modern_universe_sheet_naming_unchanged():
    """The modern workbook convention must not be disturbed by the legacy work."""
    assert dl.excel_sheet_name("M&M") == "M-M"


# ------------------------------------------------ downloader generalization

def test_lineage_loader_rejects_stitching_registry(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("stitch_successors_into_canonical: true\nsecurities: []\n")
    with pytest.raises(ValueError, match="stitch"):
        dl.load_lineage(REPO_ROOT, str(bad))


def test_default_universe_is_still_the_modern_one():
    """Backward compatibility: no --universe-config means configs/nifty50.yaml."""
    modern = dl.load_universe(REPO_ROOT, None)
    legacy = dl.load_universe(REPO_ROOT, "configs/nifty50_legacy_user_supplied.yaml")
    assert len(modern) == 50 and len(legacy) == 50
    assert set(modern) != set(legacy)


def test_yahoo_symbol_falls_back_without_a_registry():
    assert dl.yahoo_symbol("RELIANCE", None) == "RELIANCE.NS"
    assert dl.yahoo_symbol("M&M", None) == "M&M.NS"
