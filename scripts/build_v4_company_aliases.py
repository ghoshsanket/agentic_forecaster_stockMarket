"""Build the V4 company alias registry from dated, layered evidence.

EVIDENCE HIERARCHY (strongest first)
------------------------------------
1. ``configs/legacy_security_lineage.yaml`` -- curated, dated corporate lineage.
2. ``configs/v4/company_alias_overrides.yaml`` -- explicit dated operator/NSE
   evidence, including every documented rename boundary.
3. ``configs/v2/sector_map.yaml`` -- official NSE CURRENT company name.
4. Targeted official NSE/company verification (recorded in the overrides file).
5. Otherwise ``UNRESOLVED``.

THE CRITICAL DISTINCTION
------------------------
An official *current* name is evidence of WHO the issuer is. It is NOT evidence
that the name was valid in 2013-2018. Applying today's name to historical news
would be a silent anachronism, so the two kinds of evidence are recorded in
separate fields (``current_name_evidence`` vs ``historical_validity_evidence``)
and never merged.

Bare tickers are permanently ``AMBIGUOUS``: ``ITC``, ``LT``, ``M&M`` and ``BEL``
are acronyms or common words that match unrelated documents, so they can never
participate in a query automatically.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date, datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agentic_forecaster.v4 import (
    ALIAS_AMBIGUOUS,
    ALIAS_EXACT_SAFE,
    ALIAS_HISTORICAL_SAFE,
    ALIAS_REJECTED,
    AUTO_MATCH_ALIAS_CLASSES,
    V4Track,
    repo_root,
)

LINEAGE = repo_root() / "configs" / "legacy_security_lineage.yaml"
OVERRIDES = repo_root() / "configs" / "v4" / "company_alias_overrides.yaml"
SECTOR_MAP = repo_root() / "configs" / "v2" / "sector_map.yaml"
UNIVERSE = (repo_root() / "results" / "v2" / "multi_horizon"
            / "supervised_universe_frozen.json")
OUT = repo_root() / "configs" / "v4" / "company_aliases.yaml"

#: Lineage event types after which the historical name stopped belonging to this
#: issuer. Those names are REJECTED rather than queried.
MERGED_AWAY = {"merger_into_other_company", "acquired_and_merged"}

#: Legal suffixes stripped to build the SAFE COMMON CORPORATE FORM.
LEGAL_SUFFIXES = ("limited", "ltd.", "ltd", "corporation", "corp.")

AUDIT_COLUMNS = [
    "ticker", "canonical_current_name", "resolution_status",
    "auto_match_eligible", "aliases_used", "validity_intervals",
    "alias_classes", "evidence_source", "evidence_type", "confidence",
    "current_name_evidence", "historical_validity_evidence",
    "matched_organisation_strings", "total_matching_documents",
    "unique_domains", "collision_count", "ambiguous_alias_count", "notes",
]

NOT_MEASURED = "NOT_MEASURED_SOURCE_UNAVAILABLE"


def _json_default(value):
    """YAML parses bare dates into ``date`` objects; render them as ISO text."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"unserialisable {type(value).__name__}")


def _root(symbol: str) -> str:
    return str(symbol).split(".")[0].upper()


def _common_form(name: str) -> str | None:
    """Strip a trailing legal suffix to get a SAFE COMMON CORPORATE FORM."""
    text = name.strip()
    for suffix in LEGAL_SUFFIXES:
        if text.lower().endswith(f" {suffix}"):
            trimmed = text[: len(text) - len(suffix) - 1].strip()
            # A bare acronym would be a bare alias, which is never safe.
            if trimmed and " " in trimmed:
                return trimmed
    return None


def _alias(text: str, alias_class: str, valid_from=None, valid_to=None,
           note: str = "") -> dict:
    return {"alias": text, "class": alias_class, "valid_from": valid_from,
            "valid_to": valid_to, "note": note}


def _from_lineage(ticker: str, sec: dict) -> list[dict]:
    canonical = sec["current_or_final_company_name"]
    historical = sec.get("historical_company_name")
    event_type = str(sec.get("event_type") or "")
    event_date = sec.get("event_date")
    merged_away = event_type in MERGED_AWAY

    aliases = [_alias(canonical, ALIAS_EXACT_SAFE,
                      note="canonical current name from dated lineage")]
    common = _common_form(canonical)
    if common:
        aliases.append(_alias(common, ALIAS_EXACT_SAFE,
                              note="safe common corporate form"))
    if historical and historical.strip().lower() != canonical.strip().lower():
        if merged_away:
            aliases.append(_alias(
                historical, ALIAS_REJECTED, valid_to=event_date,
                note=f"successor lineage: {event_type}; never matches {ticker}"))
        else:
            aliases.append(_alias(
                historical, ALIAS_HISTORICAL_SAFE, valid_to=event_date,
                note=f"historical name of the same security; {event_type} on "
                     f"{event_date}"))
            hist_common = _common_form(historical)
            if hist_common:
                aliases.append(_alias(
                    hist_common, ALIAS_HISTORICAL_SAFE, valid_to=event_date,
                    note="safe common corporate form of the historical name"))
    return aliases


def _from_sector_map(name: str, evidence: dict) -> list[dict]:
    canonical = name.strip()
    if canonical.upper() == "UNKNOWN" or not canonical:
        return []
    # Normalise the NSE file's shouty form to ordinary title case.
    if canonical.isupper():
        canonical = canonical.title()
    aliases = [_alias(canonical, ALIAS_EXACT_SAFE,
                      note="official NSE current company name")]
    common = _common_form(canonical)
    if common:
        aliases.append(_alias(common, ALIAS_EXACT_SAFE,
                              note="safe common corporate form"))
    return aliases


def build() -> dict:
    universe = json.loads(UNIVERSE.read_text())["eligible_tickers"]
    securities = yaml.safe_load(LINEAGE.read_text())["securities"]
    sector = yaml.safe_load(SECTOR_MAP.read_text())
    sector_entries = {e["ticker"]: e for e in sector["entries"]}
    overrides_doc = yaml.safe_load(OVERRIDES.read_text()) or {}
    overrides = {o["ticker"]: o for o in overrides_doc.get("overrides", [])}

    by_root: dict[str, dict] = {}
    for sec in securities:
        for cand in ([sec.get("primary_yahoo_candidate")]
                     + list(sec.get("additional_yahoo_candidates") or [])):
            if cand:
                by_root.setdefault(_root(cand), sec)

    entries: list[dict] = []
    for ticker in universe:
        sec = by_root.get(ticker)
        override = overrides.get(ticker)
        sector_entry = sector_entries.get(ticker)

        if override is not None:
            aliases = [dict(a) for a in override.get("aliases") or []]
            canonical = override.get("canonical_company_name")
            entries.append({
                "ticker": ticker,
                "canonical_company_name": canonical,
                "aliases": aliases,
                "ambiguous_aliases": [ticker],
                "forbidden_aliases": [{"alias": ticker, "reason":
                                       "bare ticker is an acronym or common word"}],
                "valid_from": override.get("valid_from"),
                "valid_to": override.get("valid_to"),
                "resolution_status": "RESOLVED_FROM_OVERRIDE",
                "provenance": (f"configs/v4/company_alias_overrides.yaml; "
                               f"{override.get('evidence_type')}; "
                               f"{override.get('evidence_source')}"),
                "evidence_type": override.get("evidence_type"),
                "evidence_source": override.get("evidence_source"),
                "evidence_date": override.get("evidence_date"),
                "confidence": "high",
                "auto_match_eligible": True,
                "current_name_evidence": {
                    "source": override.get("evidence_source"),
                    "type": override.get("evidence_type"),
                    "date": override.get("evidence_date"),
                    "record": override.get("evidence_record"),
                    "verified": True,
                },
                "historical_validity_evidence": {
                    "aliases": [a["alias"] for a in aliases
                                if a.get("class") == ALIAS_HISTORICAL_SAFE],
                    "independently_fetched": bool(
                        override.get("historical_validity_independently_fetched")),
                    "note": override.get("notes"),
                },
                "legacy_label": (sec or {}).get("legacy_label"),
                "event_type": (sec or {}).get("event_type"),
            })
            continue

        if sec is not None:
            aliases = _from_lineage(ticker, sec)
            entries.append({
                "ticker": ticker,
                "canonical_company_name": sec["current_or_final_company_name"],
                "aliases": aliases,
                "ambiguous_aliases": [ticker],
                "forbidden_aliases": [{"alias": ticker, "reason":
                                       "bare ticker is an acronym or common word"}],
                "valid_from": sec.get("event_date"),
                "valid_to": None,
                "resolution_status": "RESOLVED_FROM_LINEAGE",
                "provenance": (f"configs/legacy_security_lineage.yaml"
                               f"[{sec['legacy_label']}]"),
                "evidence_type": "CURATED_SECURITY_LINEAGE",
                "evidence_source": "configs/legacy_security_lineage.yaml",
                "evidence_date": sec.get("event_date"),
                "confidence": str(sec.get("confidence") or "medium"),
                "auto_match_eligible": True,
                "current_name_evidence": {
                    "source": "configs/legacy_security_lineage.yaml",
                    "type": "CURATED_SECURITY_LINEAGE",
                    "date": sec.get("event_date"),
                    "verified": True,
                },
                "historical_validity_evidence": {
                    "aliases": [a["alias"] for a in aliases
                                if a.get("class") == ALIAS_HISTORICAL_SAFE],
                    "independently_fetched": False,
                    "note": "dated rename recorded in the curated lineage",
                },
                "legacy_label": sec.get("legacy_label"),
                "event_type": sec.get("event_type"),
            })
            continue

        if sector_entry is not None:
            aliases = _from_sector_map(sector_entry["company_name"], sector)
            if aliases:
                entries.append({
                    "ticker": ticker,
                    "canonical_company_name": aliases[0]["alias"],
                    "aliases": aliases,
                    "ambiguous_aliases": [ticker],
                    "forbidden_aliases": [{"alias": ticker, "reason":
                                           "bare ticker is an acronym or common word"}],
                    "valid_from": None,
                    "valid_to": None,
                    "resolution_status": "RESOLVED_FROM_OFFICIAL_NSE_NAME",
                    "provenance": (f"configs/v2/sector_map.yaml"
                                   f"[{sector.get('source')}]"),
                    "evidence_type": "OFFICIAL_NSE_CONSTITUENT_FILE",
                    "evidence_source": sector.get("source"),
                    "evidence_date": None,
                    "confidence": "high",
                    "auto_match_eligible": True,
                    "current_name_evidence": {
                        "source": sector.get("source"),
                        "type": "OFFICIAL_NSE_CONSTITUENT_FILE",
                        "sha256": sector.get("source_sha256"),
                        "verified": True,
                    },
                    "historical_validity_evidence": {
                        "aliases": [],
                        "independently_fetched": False,
                        "note": ("NOT established: an official CURRENT name is "
                                 "not evidence of validity in 2013-2018. No "
                                 "rename is known for this issuer, but absence "
                                 "of a known rename is not proof."),
                    },
                    "legacy_label": None,
                    "event_type": None,
                })
                continue

        entries.append({
            "ticker": ticker,
            "canonical_company_name": None,
            "aliases": [],
            "ambiguous_aliases": [ticker],
            "forbidden_aliases": [],
            "valid_from": None,
            "valid_to": None,
            "resolution_status": "UNRESOLVED",
            "provenance": ("no lineage row, no override and no official NSE name"),
            "evidence_type": None,
            "evidence_source": None,
            "evidence_date": None,
            "confidence": "none",
            "auto_match_eligible": False,
            "current_name_evidence": None,
            "historical_validity_evidence": None,
            "legacy_label": None,
            "event_type": None,
        })

    return {
        "registry_id": "V4_COMPANY_ALIAS_REGISTRY",
        "generated_by": "scripts/build_v4_company_aliases.py",
        "universe_sha256": json.loads(UNIVERSE.read_text())["universe_sha256"],
        "evidence_hierarchy": [
            "configs/legacy_security_lineage.yaml",
            "configs/v4/company_alias_overrides.yaml",
            "configs/v2/sector_map.yaml (official NSE current name)",
            "targeted official NSE/company verification",
            "UNRESOLVED",
        ],
        "n_tickers": len(entries),
        "n_resolved": sum(1 for e in entries if e["auto_match_eligible"]),
        "n_unresolved": sum(1 for e in entries
                            if e["resolution_status"] == "UNRESOLVED"),
        "policy": {
            "auto_match_classes": list(AUTO_MATCH_ALIAS_CLASSES),
            "never_auto_match": [ALIAS_AMBIGUOUS, ALIAS_REJECTED],
            "bare_ticker": "always AMBIGUOUS; requires a validated phrase query",
            "current_name_is_not_historical_evidence": True,
        },
        "companies": entries,
    }


def _interval(entry: dict) -> str:
    if entry.get("valid_from") or entry.get("valid_to"):
        return f"{entry.get('valid_from') or 'earliest'} .. " \
               f"{entry.get('valid_to') or 'present'}"
    return "undated (current)"


def write_audit(registry: dict) -> None:
    """Alias collision audit, with current-name evidence kept separate from
    historical-validity evidence."""
    auto: dict[str, list[str]] = {}
    for company in registry["companies"]:
        for entry in company.get("aliases") or []:
            if entry["class"] in AUTO_MATCH_ALIAS_CLASSES:
                auto.setdefault(entry["alias"].strip().casefold(), []).append(
                    company["ticker"])

    rows = []
    for company in registry["companies"]:
        safe = [a for a in company.get("aliases") or []
                if a["class"] in AUTO_MATCH_ALIAS_CLASSES]
        used = [a["alias"] for a in safe]
        collisions = {a: auto[a.strip().casefold()] for a in used
                      if len(auto.get(a.strip().casefold(), [])) > 1}
        hist = company.get("historical_validity_evidence") or {}
        current_ev = company.get("current_name_evidence") or {}
        rows.append({
            "ticker": company["ticker"],
            "canonical_current_name": (company["canonical_company_name"]
                                       or "UNRESOLVED"),
            "resolution_status": company["resolution_status"],
            "auto_match_eligible": company["auto_match_eligible"],
            "aliases_used": "; ".join(used) or "none",
            "validity_intervals": "; ".join(
                f"{a['alias']} [{_interval(a)}]" for a in safe) or "none",
            "alias_classes": "; ".join(sorted({a["class"] for a in safe})) or "none",
            "evidence_source": company.get("evidence_source") or "",
            "evidence_type": company.get("evidence_type") or "",
            "confidence": company.get("confidence") or "",
            "current_name_evidence": (
                f"{current_ev.get('type')} @ {current_ev.get('source')}"
                if current_ev else "none"),
            "historical_validity_evidence": (
                "; ".join(hist.get("aliases") or []) or
                "NOT_ESTABLISHED (current name is not historical evidence)"),
            "matched_organisation_strings": NOT_MEASURED,
            "total_matching_documents": NOT_MEASURED,
            "unique_domains": NOT_MEASURED,
            "collision_count": len(collisions),
            "ambiguous_alias_count": len(company.get("ambiguous_aliases") or []),
            "notes": (f"collisions={collisions}" if collisions
                      else company["provenance"]),
        })

    track = V4Track()
    track.results_root.mkdir(parents=True, exist_ok=True)
    with track.path("alias_audit_csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    resolved = [r for r in rows if r["auto_match_eligible"]]
    unresolved = [r for r in rows if not r["auto_match_eligible"]]
    by_status: dict[str, list[str]] = {}
    for r in resolved:
        by_status.setdefault(r["resolution_status"], []).append(r["ticker"])
    total_collisions = sum(r["collision_count"] for r in rows)
    with_renames = [r for r in resolved
                    if r["historical_validity_evidence"].startswith(("Infosys",
                                                                     "Tata",
                                                                     "The ",
                                                                     "BOC",
                                                                     "Hindustan",
                                                                     "Imperial",
                                                                     "Gujarat",
                                                                     "Essel",
                                                                     "Madras",
                                                                     "Procter",
                                                                     "Bharti"))]

    lines = [
        "# V4 company alias collision audit",
        "",
        "Generated by `scripts/build_v4_company_aliases.py`.",
        "",
        "## 1. Resolution outcome",
        "",
        f"- supervised tickers: **{len(rows)}**",
        f"- resolved: **{len(resolved)}**",
        f"- unresolved: **{len(unresolved)}**",
        f"- alias collisions among auto-matched aliases: **{total_collisions}**",
        "",
        "| evidence tier | tickers |",
        "|---|---|",
    ]
    for status, tickers in sorted(by_status.items()):
        lines.append(f"| {status} | {len(tickers)} |")
    lines += [
        "",
        "## 2. Two kinds of evidence, kept apart",
        "",
        "**Current-name evidence** answers *who is the issuer today*. It comes",
        "from the official NSE constituent list or the official NSE security",
        "master.",
        "",
        "**Historical-validity evidence** answers *was this name in force during",
        "2013-2018*. It exists only where a dated rename is documented. An",
        "official current name is deliberately NOT counted as historical",
        "evidence, so `NOT_ESTABLISHED` appears wherever no rename is known --",
        "absence of a known rename is not proof of a constant name.",
        "",
        f"- companies with a documented historical alias: **{len(with_renames)}**",
        "",
        "## 3. Bare tickers",
        "",
        "Every bare ticker is recorded AMBIGUOUS and excluded from automatic",
        "queries. `ITC`, `LT`, `M&M` and `BEL` are acronyms or common words.",
        "",
        "## 4. Corpus-dependent columns",
        "",
        "Matched organisation strings, document counts and domain diversity need",
        "an authenticated news corpus. Media Cloud is reachable but no credential",
        f"is present, so these read `{NOT_MEASURED}` -- deliberately not `0`.",
        "",
        "## 5. Per-company table",
        "",
        ("| ticker | canonical current name | aliases | validity | class | "
         "evidence | confidence |"),
        "|---|---|---|---|---|---|---|",
    ]
    for row in resolved:
        lines.append(
            f"| {row['ticker']} | {row['canonical_current_name']} | "
            f"{row['aliases_used'][:110]} | {row['validity_intervals'][:80]} | "
            f"{row['alias_classes']} | {row['evidence_type']} | "
            f"{row['confidence']} |")
    lines += ["", "## 6. Unresolved (excluded, never guessed)", "",
              "| ticker | reason |", "|---|---|"]
    for row in unresolved:
        lines.append(f"| {row['ticker']} | {row['notes']} |")
    lines.append("")
    track.path("alias_audit").write_text("\n".join(lines))
    print(f"wrote {track.path('alias_audit')}")
    print(f"wrote {track.path('alias_audit_csv')}")


def main() -> int:
    registry = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump(registry, sort_keys=False, allow_unicode=True,
                                 width=100))
    print(json.dumps({k: registry[k] for k in
                      ("n_tickers", "n_resolved", "n_unresolved")}, indent=2))
    unresolved = [c["ticker"] for c in registry["companies"]
                  if c["resolution_status"] == "UNRESOLVED"]
    print(f"unresolved tickers: {unresolved or 'none'}")
    statuses: dict[str, int] = {}
    for company in registry["companies"]:
        statuses[company["resolution_status"]] = (
            statuses.get(company["resolution_status"], 0) + 1)
    print(f"resolution tiers: {statuses}")
    print(f"wrote {OUT}")
    write_audit(registry)
    track = V4Track()
    track.results_root.mkdir(parents=True, exist_ok=True)
    (track.results_root / "company_aliases_resolved.json").write_text(
        json.dumps(registry, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())