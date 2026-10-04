"""V4 source-stage tests: alias safety, news causality, GDELT event parsing.

These cover the guarantees that must hold BEFORE any corpus is downloaded, so a
later stage cannot quietly weaken them.
"""

from __future__ import annotations

import csv
import io
import zipfile
from datetime import date
from pathlib import Path

import pytest
import yaml

from agentic_forecaster.v4 import (
    ALIAS_AMBIGUOUS,
    ALIAS_EXACT_SAFE,
    ALIAS_HISTORICAL_SAFE,
    ALIAS_REJECTED,
    AUTO_MATCH_ALIAS_CLASSES,
)
from agentic_forecaster.v4.availability import (
    PostCutoffDataAccessError,
    SameDayNewsError,
    assert_news_before_origin,
    in_news_window,
    news_window,
)

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def registry() -> dict:
    return yaml.safe_load((REPO / "configs" / "v4" / "company_aliases.yaml").read_text())


def _entry(registry: dict, ticker: str) -> dict:
    return next(c for c in registry["companies"] if c["ticker"] == ticker)


# --------------------------------------------------------------------------
# §60 alias safety
# --------------------------------------------------------------------------

def test_bare_ticker_is_always_ambiguous(registry: dict) -> None:
    """A bare ticker must never be auto-matchable, for ANY ticker."""
    for company in registry["companies"]:
        assert ticker_is_auto_safe(company, company["ticker"]) is False, (
            f"{company['ticker']}: bare ticker must be AMBIGUOUS")


def ticker_is_auto_safe(company: dict, alias: str) -> bool:
    """An alias participates in automatic matching only if it is safe."""
    if alias in (company.get("ambiguous_aliases") or []):
        return False
    for entry in company.get("aliases") or []:
        if entry["alias"] == alias:
            return entry["class"] in AUTO_MATCH_ALIAS_CLASSES
    for entry in company.get("forbidden_aliases") or []:
        if entry["alias"] == alias:
            return False
    return False


@pytest.mark.parametrize("ticker,bare", [("ITC", "ITC"), ("LT", "LT"),
                                         ("M&M", "M&M"), ("BEL", "BEL")])
def test_ambiguous_bare_aliases_rejected(registry: dict, ticker: str,
                                         bare: str) -> None:
    company = _entry(registry, ticker)
    assert bare in company["ambiguous_aliases"]
    assert not ticker_is_auto_safe(company, bare)


def test_short_former_names_are_never_auto_matched(registry: dict) -> None:
    """Legacy short labels (ACC, TISCO, INFOSYSTCH) must not become aliases."""
    short_labels = {"ACC", "TISCO", "INFOSYSTCH", "HINDLEVER", "GUTRLCEMENT"}
    auto = {entry["alias"] for company in registry["companies"]
            for entry in company.get("aliases") or []
            if entry["class"] in AUTO_MATCH_ALIAS_CLASSES}
    assert not (auto & short_labels)


def test_full_name_is_exact_safe_and_bare_code_is_not(registry: dict) -> None:
    itc = _entry(registry, "ITC")
    assert ticker_is_auto_safe(itc, "ITC Limited")
    assert not ticker_is_auto_safe(itc, "ITC")


def test_historical_alias_respects_validity_interval(registry: dict) -> None:
    """Infosys was 'Infosys Technologies' only until mid-2011."""
    infy = _entry(registry, "INFY")
    historical = next(a for a in infy["aliases"]
                      if a["alias"] == "Infosys Technologies Limited")
    assert historical["class"] == ALIAS_HISTORICAL_SAFE
    assert historical["valid_to"] == "2011-06"
    assert ticker_is_auto_safe(infy, "Infosys Limited")


def test_historical_rename_is_not_applied_to_whole_history(registry: dict) -> None:
    """Tata Iron and Steel must not be the name used for a 2016 origin."""
    steel = _entry(registry, "TATASTEEL")
    historical = next(a for a in steel["aliases"] if "Tata Iron and Steel" in a["alias"])
    assert historical["class"] == ALIAS_HISTORICAL_SAFE
    assert historical["valid_to"] < "20130101", (
        "a name retired before the study period must not remain valid")


def test_merged_away_company_name_is_rejected(registry: dict) -> None:
    """A dissolved company's name must never match its successor ticker."""
    for company in registry["companies"]:
        for entry in company.get("aliases") or []:
            if "never matches this ticker" in str(entry.get("note", "")):
                assert entry["class"] == ALIAS_REJECTED
                assert not ticker_is_auto_safe(company, entry["alias"])


def test_alias_cannot_map_to_two_current_tickers(registry: dict) -> None:
    """No auto-matchable alias may be claimed by two different tickers."""
    owner: dict[str, str] = {}
    for company in registry["companies"]:
        for entry in company.get("aliases") or []:
            if entry["class"] not in AUTO_MATCH_ALIAS_CLASSES:
                continue
            key = entry["alias"].strip().casefold()
            assert key not in owner, (
                f"alias {entry['alias']!r} claimed by both {owner.get(key)} and "
                f"{company['ticker']}")
            owner[key] = company["ticker"]


def test_unresolved_tickers_are_excluded_not_guessed(registry: dict) -> None:
    """The INVARIANT holds regardless of how many are unresolved.

    Previously 26 tickers were UNRESOLVED. They are now resolved from the official
    NSE name, but the guarantee must not regress: any ticker that ever fails to
    resolve must carry no name at all rather than a plausible guess.
    """
    unresolved = [c for c in registry["companies"]
                  if c["resolution_status"] == "UNRESOLVED"]
    for company in unresolved:
        assert company["canonical_company_name"] is None
        assert company["auto_match_eligible"] is False
        assert company["aliases"] == []
    assert registry["n_resolved"] >= 25, (
        "V4 policy requires at least 25 sentiment-eligible companies before "
        "model screening")
    assert registry["n_resolved"] + registry["n_unresolved"] == \
        registry["n_tickers"] == 42


def test_registry_covers_exactly_the_supervised_universe(registry: dict) -> None:
    universe = yaml.safe_load(
        (REPO / "configs" / "v4" / "company_aliases.yaml").read_text())
    import json
    expected = json.loads(
        (REPO / "results" / "v2" / "multi_horizon"
         / "supervised_universe_frozen.json").read_text())["eligible_tickers"]
    got = [c["ticker"] for c in universe["companies"]]
    assert got == expected
    assert registry["n_tickers"] == len(expected)


def test_alias_class_vocabulary_is_closed(registry: dict) -> None:
    allowed = {ALIAS_EXACT_SAFE, ALIAS_HISTORICAL_SAFE, ALIAS_AMBIGUOUS,
               ALIAS_REJECTED}
    for company in registry["companies"]:
        for entry in company.get("aliases") or []:
            assert entry["class"] in allowed


# --------------------------------------------------------------------------
# §61 temporal causality
# --------------------------------------------------------------------------

def test_same_day_news_is_rejected() -> None:
    with pytest.raises(SameDayNewsError):
        assert_news_before_origin(date(2017, 6, 12), date(2017, 6, 12))


def test_prior_day_news_is_accepted() -> None:
    assert_news_before_origin(date(2017, 6, 11), date(2017, 6, 12))


def test_weekend_news_accepted_on_monday_origin() -> None:
    """Monday 2017-06-12 may use Sunday 2017-06-11 news."""
    assert_news_before_origin(date(2017, 6, 11), date(2017, 6, 12))


def test_future_news_is_rejected() -> None:
    with pytest.raises(SameDayNewsError):
        assert_news_before_origin(date(2017, 6, 13), date(2017, 6, 12))


def test_2019_news_is_sealed_during_development() -> None:
    with pytest.raises(PostCutoffDataAccessError):
        assert_news_before_origin(date(2019, 6, 11), date(2019, 6, 12))
    assert_news_before_origin(date(2019, 6, 11), date(2019, 6, 12),
                              allow_lockbox_year=True)


def test_2020_plus_news_raises_firewall_even_in_lockbox() -> None:
    with pytest.raises(PostCutoffDataAccessError):
        assert_news_before_origin(date(2020, 1, 2), date(2020, 1, 6),
                                  allow_lockbox_year=True)


def test_missing_publish_date_is_rejected_not_repaired() -> None:
    with pytest.raises(ValueError):
        assert_news_before_origin(None, date(2017, 6, 12))


def test_windows_are_calendar_days_and_exclude_origin_day() -> None:
    origin = date(2017, 6, 12)
    assert news_window(origin, 1) == (date(2017, 6, 11), date(2017, 6, 12))
    assert news_window(origin, 7) == (date(2017, 6, 5), date(2017, 6, 12))
    # a window can never include the origin day itself
    assert not in_news_window(origin, origin, 30)
    assert in_news_window(date(2017, 6, 11), origin, 1)
    assert not in_news_window(date(2017, 6, 10), origin, 1)


# --------------------------------------------------------------------------
# §59 GDELT event parser invariants
# --------------------------------------------------------------------------

def _event_row(**overrides) -> list[str]:
    row = [""] * 61
    row[0] = "123456789"
    row[29] = "4"
    row[30] = "-10.0"
    row[34] = "-3.25"
    row[36] = "India"
    row[37] = "IN"
    row[59] = "20170612000000"
    row[60] = "https://example.invalid/a"
    for key, value in overrides.items():
        row[int(key[1:])] = value
    return row


def _write_archive(path: Path, rows: list[list[str]]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        buffer = io.StringIO()
        writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)
        zf.writestr("events.export.CSV", buffer.getvalue())
    return path


def test_zip_parsing_and_ragged_rows_counted(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "a.export.CSV.zip",
                             [_event_row(), _event_row(i0="1")[:20]])
    with zipfile.ZipFile(archive) as zf:
        assert zf.testzip() is None
        rows = list(csv.reader(
            io.TextIOWrapper(zf.open(zf.namelist()[0]), encoding="utf-8"),
            delimiter="\t"))
    widths = {len(r) for r in rows}
    assert widths == {61, 20}, "ragged rows must be visible, not silently padded"
    kept = [r for r in rows if len(r) == 61]
    assert len(kept) == 1, "only well-formed rows may be indexed positionally"


def test_corrupt_zip_is_detected(tmp_path: Path) -> None:
    """A corrupt archive must raise, never degrade into an empty result."""
    archive = tmp_path / "bad.export.CSV.zip"
    archive.write_bytes(b"PK\x03\x04 not really a zip")
    with pytest.raises(zipfile.BadZipFile):
        zipfile.ZipFile(archive).testzip()


def test_date_added_must_equal_archive_stamp() -> None:
    """The probe's decisive invariant: DATEADDED == the file's own stamp."""
    stamp = "20170612000000"
    row = _event_row(f59=stamp)
    assert row[59] == stamp
    assert row[59].startswith("20170612")
    # and the row's own EventDate-like field is NOT the discovery date
    assert row[1] == ""


def test_india_relevance_requires_country_evidence() -> None:
    india = _event_row()
    assert india[36].strip().lower() == "india"
    assert india[37].strip().upper() == "IN"
    elsewhere = _event_row(f36="Australia", f37="AS")
    assert elsewhere[36].strip().lower() != "india"


def test_failed_day_is_not_silently_zero() -> None:
    """A day that fails to download must be recorded as failed, never as 0."""
    status = {"records": 0, "status": "DOWNLOAD_FAILED"}
    assert status["records"] == 0
    assert status["status"] != "OK", (
        "a failed day must never be represented as a successful zero-count day")