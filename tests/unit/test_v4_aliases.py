"""V4 alias-evidence, query-planning and credential-handling tests.

Covers §20 A-R: official NSE name evidence, the current-vs-historical evidence
distinction, date-aware query construction, and the guarantee that a Media Cloud
credential can never leak into an artifact.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import urllib.error
from datetime import date
from pathlib import Path

import pytest
import yaml

from agentic_forecaster.v4 import ALIAS_AMBIGUOUS, AUTO_MATCH_ALIAS_CLASSES
from agentic_forecaster.v4.queries import (
    Alias,
    RetryPolicy,
    alias_validity_windows,
    build_plan,
    build_query,
)
from agentic_forecaster.v4.queries import (
    _parse as parse_boundary,
)

REPO = Path(__file__).resolve().parents[2]
SENTINEL_TOKEN = "SENTINEL-TOKEN-must-never-be-written-0123456789"


@pytest.fixture(scope="module")
def registry() -> dict:
    return yaml.safe_load((REPO / "configs" / "v4" / "company_aliases.yaml").read_text())


@pytest.fixture(scope="module")
def overrides() -> dict:
    return yaml.safe_load(
        (REPO / "configs" / "v4" / "company_alias_overrides.yaml").read_text())


def _company(registry: dict, ticker: str) -> dict:
    return next(c for c in registry["companies"] if c["ticker"] == ticker)


def _aliases_of(company: dict) -> list[Alias]:
    """Build aliases through the SAME parser production uses, so the tests
    exercise the real handling of compound and year-month boundaries."""
    return [Alias(text=entry["alias"], alias_class=entry["class"],
                  valid_from=parse_boundary(entry.get("valid_from")),
                  valid_to=parse_boundary(entry.get("valid_to")))
            for entry in company["aliases"]]


def _probe_module():
    path = REPO / "scripts" / "probe_v4_mediacloud.py"
    spec = importlib.util.spec_from_file_location("probe_v4_mediacloud", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# A / B: official NSE current-name evidence, and its limits
# --------------------------------------------------------------------------

def test_sector_map_names_resolve_current_identity(registry: dict) -> None:
    """A: official NSE names from sector_map resolve issuer identity."""
    sector = yaml.safe_load((REPO / "configs" / "v2" / "sector_map.yaml").read_text())
    entries = {e["ticker"]: e for e in sector["entries"]}
    from_nse = [c for c in registry["companies"]
                if c["resolution_status"] == "RESOLVED_FROM_OFFICIAL_NSE_NAME"]
    assert from_nse, "expected NSE-name-resolved companies"
    for company in from_nse:
        official = entries[company["ticker"]]["company_name"].strip()
        assert official.upper() != "UNKNOWN"
        expected = official.title() if official.isupper() else official
        assert company["canonical_company_name"] == expected
        assert company["current_name_evidence"]["type"] == \
            "OFFICIAL_NSE_CONSTITUENT_FILE"


def test_current_name_evidence_is_not_historical_evidence(registry: dict) -> None:
    """B: a current name must NOT be recorded as proof of 2013-2018 validity."""
    for company in registry["companies"]:
        hist = company.get("historical_validity_evidence")
        if company["resolution_status"] != "RESOLVED_FROM_OFFICIAL_NSE_NAME":
            continue
        assert hist is not None
        if not hist["aliases"]:
            assert "NOT established" in hist["note"], (
                f"{company['ticker']}: absence of a known rename must be "
                f"recorded as NOT_ESTABLISHED, never as proof")


def test_registry_resolves_at_least_the_feasibility_floor(registry: dict) -> None:
    assert registry["n_tickers"] == 42
    assert registry["n_unresolved"] == 0
    assert registry["n_resolved"] >= 25


# --------------------------------------------------------------------------
# C / H / I: overrides, WIPRO, ADANIPORTS, bare tickers
# --------------------------------------------------------------------------

def test_wipro_resolved_by_explicit_official_override(registry: dict,
                                                      overrides: dict) -> None:
    """C: WIPRO is resolved by an official NSE record, not guessed."""
    wipro = _company(registry, "WIPRO")
    assert wipro["resolution_status"] == "RESOLVED_FROM_OVERRIDE"
    assert wipro["canonical_company_name"] == "Wipro Limited"
    assert wipro["current_name_evidence"]["verified"] is True
    assert "EQUITY_L.csv" in wipro["current_name_evidence"]["source"]
    assert wipro["current_name_evidence"]["record"].startswith("WIPRO,Wipro Limited")
    assert any(o["ticker"] == "WIPRO" for o in overrides["overrides"])


def test_adaniports_current_name_valid_across_development(registry: dict) -> None:
    """H: the rename predates the window, so one alias serves 2013-2018."""
    company = _company(registry, "ADANIPORTS")
    windows = alias_validity_windows(
        "ADANIPORTS", _aliases_of(company), date(2013, 4, 1), date(2018, 12, 31))
    assert {w.alias for w in windows} == {
        "Adani Ports and Special Economic Zone"}
    assert len(windows) == 1, "no split expected: the rename is pre-2012"
    old = next(a for a in company["aliases"] if "Mundra Port" in a["alias"])
    assert old["class"] == "REJECTED"


def test_bare_tickers_are_ambiguous_for_every_company(registry: dict) -> None:
    """I: no bare ticker may ever be queryable."""
    for company in registry["companies"]:
        assert company["ticker"] in company["ambiguous_aliases"]
        for entry in company["aliases"]:
            assert entry["alias"] != company["ticker"]
        windows = alias_validity_windows(
            company["ticker"], _aliases_of(company),
            date(2013, 4, 1), date(2018, 12, 31))
        assert company["ticker"] not in {w.alias for w in windows}


def test_unresolved_companies_cannot_be_queried(registry: dict) -> None:
    """J: an unresolved company contributes no query window."""
    for company in registry["companies"]:
        if company["resolution_status"] != "UNRESOLVED":
            continue
        assert company["auto_match_eligible"] is False
        assert alias_validity_windows(
            company["ticker"], _aliases_of(company),
            date(2013, 4, 1), date(2018, 12, 31)) == []


# --------------------------------------------------------------------------
# D / E / F / G / P: date-aware query construction
# --------------------------------------------------------------------------

def test_tataconsum_2015_query_uses_historical_name(registry: dict) -> None:
    """D: 2015 must search Tata Global Beverages, not Tata Consumer."""
    windows = alias_validity_windows("TATACONSUM",
                                     _aliases_of(_company(registry, "TATACONSUM")),
                                     date(2015, 1, 1), date(2015, 12, 31))
    assert {w.alias for w in windows} == {"Tata Global Beverages"}
    assert "Tata Consumer Products" not in {w.alias for w in windows}


def test_tataconsum_current_alias_never_used_in_development(registry: dict) -> None:
    """E: the post-2020 alias must not appear in any 2013-2018 window."""
    for year in range(2013, 2019):
        windows = alias_validity_windows("TATACONSUM",
                                         _aliases_of(_company(registry, "TATACONSUM")),
                                         date(year, 1, 1), date(year, 12, 31))
        for window in windows:
            assert "Consumer" not in window.alias


def test_shriramfin_2016_query_uses_historical_name(registry: dict) -> None:
    """F: 2016 must search Shriram Transport Finance Company."""
    windows = alias_validity_windows("SHRIRAMFIN",
                                     _aliases_of(_company(registry, "SHRIRAMFIN")),
                                     date(2016, 1, 1), date(2016, 12, 31))
    assert {w.alias for w in windows} == {"Shriram Transport Finance"}
    assert all(w.alias_class == "HISTORICAL_SAFE" for w in windows)


def test_titan_query_changes_at_its_2013_boundary(registry: dict) -> None:
    """G: a rename inside 2013 must split the year, not pick one name."""
    windows = alias_validity_windows("TITAN",
                                     _aliases_of(_company(registry, "TITAN")),
                                     date(2013, 4, 1), date(2013, 12, 31))
    assert len(windows) == 2
    first, second = windows
    assert first.alias == "Titan Industries" and first.end == date(2013, 8, 31)
    assert second.alias == "Titan Company" and second.start == date(2013, 9, 1)
    # contiguous, non-overlapping, and covering the whole range
    assert (second.start - first.end).days == 1
    assert first.start == date(2013, 4, 1)
    assert second.end == date(2013, 12, 31)


def test_windows_never_straddle_a_rename(registry: dict) -> None:
    """P: no window may use an alias outside its validity interval."""
    for company in registry["companies"]:
        aliases = _aliases_of(company)
        for year in (2013, 2016, 2018):
            for window in alias_validity_windows(
                    company["ticker"], aliases,
                    date(year, 1, 1), date(year, 12, 31)):
                matching = [a for a in aliases if a.text == window.alias]
                assert matching, f"{window} alias vanished"
                for alias in matching:
                    if alias.valid_from:
                        assert window.start >= alias.valid_from
                    if alias.valid_to:
                        assert window.end <= alias.valid_to


def test_build_query_quotes_phrases_and_supports_title_restriction() -> None:
    assert build_query(["Tata Global Beverages"]) == '"Tata Global Beverages"'
    combined = build_query(["Infosys Limited", "Infosys"])
    assert combined == '"Infosys Limited" OR "Infosys"'
    titled = build_query(["Infosys Limited"], title_only=True)
    assert titled.startswith("article_title:(")


def test_no_alias_maps_to_two_companies(registry: dict) -> None:
    """Collision guard after adding 21 NSE-derived companies."""
    owner: dict[str, str] = {}
    for company in registry["companies"]:
        for entry in company["aliases"]:
            if entry["class"] not in AUTO_MATCH_ALIAS_CLASSES:
                continue
            key = entry["alias"].strip().casefold()
            assert key not in owner, (
                f"{entry['alias']!r} claimed by {owner.get(key)} and "
                f"{company['ticker']}")
            owner[key] = company["ticker"]


# --------------------------------------------------------------------------
# N / O: request budget and bounded backoff
# --------------------------------------------------------------------------

def test_request_budget_counts_calls_and_excludes_story_list(registry: dict) -> None:
    """N: the budget is computed up front and never includes story-list."""
    companies = [c for c in registry["companies"] if c["auto_match_eligible"]]
    plan = build_plan(companies, [2013, 2014, 2015, 2016, 2017, 2018])
    assert plan["estimated_calls"]["story_list_calls"] == 0
    assert plan["estimated_calls"]["coverage_probe"] == 2 * plan["n_query_windows"]
    assert plan["estimated_calls"]["total_estimate"] >= 2 * len(companies) * 6
    assert plan["n_query_windows"] > len(companies) * 6, (
        "rename boundaries inside a year must add windows")
    on_disk = json.loads(
        (REPO / "results" / "v4" / "pre_covid_sentiment" / "request_plan.json")
        .read_text())
    assert on_disk["estimated_calls"]["story_list_calls"] == 0


def test_plan_never_contains_2019_or_2020(registry: dict) -> None:
    """Q / R: the plan cannot reach the sealed year or beyond."""
    companies = [c for c in registry["companies"] if c["auto_match_eligible"]]
    plan = build_plan(companies, [2013, 2014, 2015, 2016, 2017, 2018])
    for query in plan["queries"]:
        assert query["start"] <= "2018-12-31"
        assert query["end"] <= "2018-12-31"
        assert not query["start"].startswith("2019")
        assert not query["start"].startswith("202")


def test_retry_backoff_is_bounded_and_eventually_gives_up() -> None:
    """O: 429 is retried with bounded exponential backoff, then surfaced."""
    slept: list[float] = []
    policy = RetryPolicy(max_attempts=4, base_seconds=1.0, max_seconds=30.0,
                         sleep=slept.append)
    attempts = {"n": 0}

    def _always_429() -> None:
        attempts["n"] += 1
        raise urllib.error.HTTPError("u", 429, "Too Many Requests", None, None)

    with pytest.raises(urllib.error.HTTPError):
        policy.run(_always_429)
    assert attempts["n"] == 4, "must stop after max_attempts"
    assert slept == [1.0, 2.0, 4.0], "delays must grow but stay bounded"
    assert all(d <= 30.0 for d in slept)


def test_retry_succeeds_after_transient_failure() -> None:
    slept: list[float] = []
    policy = RetryPolicy(sleep=slept.append)
    state = {"n": 0}

    def _flaky() -> str:
        state["n"] += 1
        if state["n"] == 1:
            raise urllib.error.HTTPError("u", 503, "Unavailable", None, None)
        return "ok"

    assert policy.run(_flaky) == "ok"
    assert slept == [1.0]


def test_non_retryable_status_is_raised_immediately() -> None:
    policy = RetryPolicy(sleep=lambda _s: None)
    with pytest.raises(urllib.error.HTTPError):
        policy.run(lambda: (_ for _ in ()).throw(
            urllib.error.HTTPError("u", 401, "Unauthorized", None, None)))


# --------------------------------------------------------------------------
# K / L / M: credential handling
# --------------------------------------------------------------------------

def test_credential_file_permissions_are_enforced(tmp_path, monkeypatch) -> None:
    """L: a world-readable token file is rejected."""
    module = _probe_module()
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    token_file = secrets / "mediacloud_token"
    token_file.write_text(SENTINEL_TOKEN + "\n")
    token_file.chmod(0o644)
    monkeypatch.delenv("MEDIACLOUD_API_TOKEN", raising=False)
    monkeypatch.setattr(module, "secrets_root", lambda: secrets)
    with pytest.raises(module.CredentialError) as excinfo:
        module.resolve_token()
    assert "0600" in str(excinfo.value)


def test_credential_file_with_0600_is_accepted(tmp_path, monkeypatch) -> None:
    module = _probe_module()
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    token_file = secrets / "mediacloud_token"
    token_file.write_text(SENTINEL_TOKEN + "\n")
    token_file.chmod(0o600)
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    monkeypatch.delenv("MEDIACLOUD_API_TOKEN", raising=False)
    monkeypatch.setattr(module, "secrets_root", lambda: secrets)
    token, provenance, permissions = module.resolve_token()
    assert token == SENTINEL_TOKEN
    assert "0600" in permissions
    assert provenance.startswith("file:")


def test_absent_credential_yields_no_token(tmp_path, monkeypatch) -> None:
    module = _probe_module()
    monkeypatch.delenv("MEDIACLOUD_API_TOKEN", raising=False)
    monkeypatch.setattr(module, "secrets_root", lambda: tmp_path)
    token, provenance, _ = module.resolve_token()
    assert token is None
    assert provenance == "absent"


def test_no_authenticated_call_when_token_absent(tmp_path, monkeypatch) -> None:
    """M: without a credential, no request may carry an Authorization header."""
    module = _probe_module()
    monkeypatch.delenv("MEDIACLOUD_API_TOKEN", raising=False)
    monkeypatch.setattr(module, "secrets_root", lambda: tmp_path)
    monkeypatch.setattr(module, "V4Track", lambda: _FakeTrack(tmp_path))
    monkeypatch.setattr(module, "ALIASES", REPO / "configs" / "v4"
                        / "company_aliases.yaml")
    monkeypatch.setattr(module, "REGISTRY", REPO / "configs" / "v4"
                        / "source_registry.yaml")

    seen: list[dict] = []

    def _record(request, timeout=None):
        headers = {k.title(): v for k, v in request.header_items()}
        seen.append(headers)
        # Emulate the real anonymous response, so reachability succeeds and the
        # credential gate is what actually blocks the run.
        return _FakeResponse(401, json.dumps(
            {"detail": "Authentication credentials were not provided."}))

    monkeypatch.setattr(module.urllib.request, "urlopen", _record)
    module.main()

    assert seen, "the reachability probe should still have run"
    assert all("Authorization" not in h for h in seen), (
        "no Authorization header may be sent without a token")

    rows = track_rows(tmp_path)
    assert rows
    assert {r["status"] for r in rows} == {"BLOCKED_NO_CREDENTIAL"}
    # A blocked cell is written as an EMPTY csv cell (None serialises to ""),
    # which must never be confused with a measured zero.
    assert all(r["general_count"] in (None, "") for r in rows), (
        "blocked coverage must stay empty, never 0")
    assert not any(str(r["general_count"]).strip() == "0" for r in rows), (
        "a blocked cell must never look like a measured zero count")


class _FakeResponse:
    """Minimal urlopen context-manager stand-in."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self._body = body.encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeTrack:
    """Redirects probe artifacts into a temporary directory."""

    def __init__(self, root: Path) -> None:
        self.results_root = root / "results"
        self._files: dict[str, str] = {
            "mediacloud_probe": "MEDIACLOUD_PROBE.md",
            "mediacloud_probe_csv": "mediacloud_probe.csv",
        }

    def path(self, key: str) -> Path:
        return self.results_root / self._files[key]


def track_rows(root: Path) -> list[dict]:
    import csv as _csv
    with (root / "results" / "mediacloud_probe.csv").open(newline="") as handle:
        return list(_csv.DictReader(handle))


def test_token_never_appears_in_artifacts(tmp_path, monkeypatch) -> None:
    """K: a real credential must not leak into any written artifact."""
    module = _probe_module()
    monkeypatch.setenv("MEDIACLOUD_API_TOKEN", SENTINEL_TOKEN)
    monkeypatch.setattr(module, "V4Track", lambda: _FakeTrack(tmp_path))
    monkeypatch.setattr(module, "ALIASES", REPO / "configs" / "v4"
                        / "company_aliases.yaml")
    monkeypatch.setattr(module, "REGISTRY", REPO / "configs" / "v4"
                        / "source_registry.yaml")

    def _record(request, timeout=None):
        raise urllib.error.URLError("network disabled for test")

    monkeypatch.setattr(module.urllib.request, "urlopen", _record)
    module.main()

    written = list((tmp_path / "results").rglob("*"))
    assert written
    for path in written:
        if path.is_file():
            text = path.read_text(errors="ignore")
            assert SENTINEL_TOKEN not in text, f"token leaked into {path.name}"
    assert os.environ["MEDIACLOUD_API_TOKEN"] == SENTINEL_TOKEN


def test_ambiguous_alias_class_is_never_queryable() -> None:
    ambiguous = Alias("ITC", ALIAS_AMBIGUOUS)
    assert not ambiguous.valid_on(date(2015, 1, 1))
    assert alias_validity_windows("ITC", [ambiguous], date(2013, 1, 1),
                                  date(2018, 12, 31)) == []