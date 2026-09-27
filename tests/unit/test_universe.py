"""Ticker universe policy: no cross-company substitution (item 11)."""

from __future__ import annotations

from pathlib import Path

from agentic_forecaster.data.universe import load_universe_config, resolve_universe

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "nifty50.yaml"


def _universe():
    return load_universe_config(CONFIG)


def test_universe_has_fifty_symbols():
    u = _universe()
    assert len(u.requested) == 50
    assert len(set(u.requested)) == 50


def test_tatacomm_is_not_used_as_a_stand_in_for_a_different_company():
    """TATACOMM is Tata Communications, NOT Tata Motors.

    The reconstruction must never alias TATAMOTORS to TATACOMM.
    """
    u = _universe()
    assert "TATACOMM" not in u.requested, (
        "TATACOMM is a different company and must not appear in the NIFTY-50 universe"
    )
    assert "TATACOMM" not in u.aliases.values()
    if "TATAMOTORS" in u.requested:
        assert u.aliases.get("TATAMOTORS") != "TATACOMM"


def test_m_and_m_alias_is_same_security():
    u = _universe()
    assert u.aliases.get("M&M") == "MM"


def test_aliases_only_reference_genuine_format_variants():
    u = _universe()
    for requested, dataset_symbol in u.aliases.items():
        # A symbol-format alias differs only by punctuation, never by identity.
        assert dataset_symbol in requested.replace("&", "").replace("-", "").upper() or \
            requested.replace("&", "") == dataset_symbol


def test_unavailable_symbols_are_documented():
    u = _universe()
    for symbol in u.known_unavailable:
        assert symbol in u.requested


def test_resolve_marks_missing_as_unavailable_not_substituted():
    u = _universe()
    discovered = {"RELIANCE": object(), "MM": object()}
    resolved = resolve_universe(u, discovered)
    assert resolved.available.get("RELIANCE") == "RELIANCE"
    assert resolved.available.get("M&M") == "MM"
    assert "ZOMATO" in resolved.unavailable
    assert "ZOMATO" not in resolved.available
    for symbol in resolved.unavailable:
        assert symbol not in resolved.available


def test_availability_csv_is_regenerated():
    csv = REPO_ROOT / "results" / "ticker_availability.csv"
    assert csv.is_file()
    header = csv.read_text().splitlines()[0]
    for column in ("requested_ticker", "discovered_symbol", "available",
                   "source_file", "source_start", "source_end",
                   "fold_0_available", "fold_1_available"):
        assert column in header
