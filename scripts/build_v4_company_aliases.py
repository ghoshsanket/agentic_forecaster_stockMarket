"""Build the V4 company alias registry from existing lineage evidence.

WHY THIS EXISTS
---------------
News sources must never be queried with a bare ticker.  ``ITC``, ``LT``, ``BEL``
and ``M&M`` are common words or acronyms that match thousands of unrelated
documents, and a company's legal name changed inside the study period.  The
registry therefore records, per ticker:

* the canonical name valid *today*,
* every historical name with an explicit validity interval,
* the bare ticker, permanently classified ``AMBIGUOUS`` so it can never match
  automatically,
* names of companies that were dissolved or merged away, classified
  ``REJECTED`` with the reason.

EVIDENCE HIERARCHY (strongest first)
-------------------------------------
1. ``configs/legacy_security_lineage.yaml`` -- curated, dated, with confidence.
2. An explicit ``configs/v4/company_alias_overrides.yaml`` -- operator evidence.
3. Nothing.  A ticker with no evidence is emitted as ``UNRESOLVED`` and is
   EXCLUDED from automatic matching.  Guessing a legal name from memory is
   exactly the silent-error this registry exists to prevent.
"""

from __future__ import annotations

import csv
import json
import sys
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
UNIVERSE = repo_root() / "results" / "v2" / "multi_horizon" / "supervised_universe_frozen.json"
OUT = repo_root() / "configs" / "v4" / "company_aliases.yaml"

#: Lineage event types after which the historical name stopped being the
#: company's own name.  Used only to decide whether a historical name is safe.
MERGED_AWAY = {"merger_into_other_company", "acquired_and_merged"}


def _root(symbol: str) -> str:
    return str(symbol).split(".")[0].upper()


def build() -> dict:
    universe = json.loads(UNIVERSE.read_text())["eligible_tickers"]
    securities = yaml.safe_load(LINEAGE.read_text())["securities"]
    overrides = {}
    if OVERRIDES.exists():
        overrides = (yaml.safe_load(OVERRIDES.read_text()) or {}).get("overrides", {})

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
        if sec is None and override is None:
            entries.append({
                "ticker": ticker,
                "canonical_company_name": None,
                "aliases": [],
                "ambiguous_aliases": [ticker],
                "forbidden_aliases": [],
                "valid_from": None,
                "valid_to": None,
                "resolution_status": "UNRESOLVED",
                "provenance": "no lineage row matched this yfinance symbol; "
                              "official NSE/company evidence required",
                "confidence": "none",
                "auto_match_eligible": False,
            })
            continue

        canonical = (override or {}).get("canonical_company_name") or sec[
            "current_or_final_company_name"]
        historical = sec.get("historical_company_name")
        event_type = str(sec.get("event_type") or "")
        event_date = sec.get("event_date")
        confidence = str(sec.get("confidence") or "medium")

        aliases: list[dict] = [{
            "alias": canonical,
            "class": ALIAS_EXACT_SAFE,
            "valid_from": None,
            "valid_to": None,
            "note": "canonical current name",
        }]
        # A historical name is only auto-matchable if the SAME security carried
        # it; a company that was merged away never matches this ticker.
        if historical and historical.strip().lower() != canonical.strip().lower():
            merged_away = event_type in MERGED_AWAY
            aliases.append({
                "alias": historical,
                "class": ALIAS_REJECTED if merged_away else ALIAS_HISTORICAL_SAFE,
                "valid_from": None,
                "valid_to": event_date if not merged_away else None,
                "note": (f"historical name; {event_type} on {event_date}"
                         if not merged_away
                         else f"successor lineage: {event_type}; never matches this ticker"),
            })

        forbidden = []
        if merged_away and historical:
            forbidden.append({
                "alias": historical,
                "reason": f"company was merged/acquired away ({event_type}); "
                          f"successor is {canonical}",
            })

        entries.append({
            "ticker": ticker,
            "canonical_company_name": canonical,
            "aliases": aliases,
            "ambiguous_aliases": [ticker],
            "forbidden_aliases": forbidden,
            "valid_from": event_date,
            "valid_to": None,
            "resolution_status": "RESOLVED_FROM_LINEAGE",
            "provenance": (f"configs/legacy_security_lineage.yaml[{sec['legacy_label']}]"
                           + (f"; override {override}" if override else "")),
            "confidence": confidence,
            "auto_match_eligible": True,
            "legacy_label": sec.get("legacy_label"),
            "event_type": event_type,
        })

    return {
        "registry_id": "V4_COMPANY_ALIAS_REGISTRY",
        "generated_by": "scripts/build_v4_company_aliases.py",
        "universe_sha256": json.loads(UNIVERSE.read_text())["universe_sha256"],
        "n_tickers": len(entries),
        "n_resolved": sum(1 for e in entries if e["auto_match_eligible"]),
        "n_unresolved": sum(1 for e in entries
                            if e["resolution_status"] == "UNRESOLVED"),
        "policy": {
            "auto_match_classes": [ALIAS_EXACT_SAFE, ALIAS_HISTORICAL_SAFE],
            "never_auto_match": [ALIAS_AMBIGUOUS, ALIAS_REJECTED],
            "bare_ticker": "always AMBIGUOUS; requires a validated phrase query",
        },
        "companies": entries,
    }


AUDIT_COLUMNS = [
    "ticker", "canonical_name", "resolution_status", "auto_match_eligible",
    "aliases_used", "matched_gdelt_or_mediacloud_organisation_strings",
    "total_matching_documents", "active_date_range", "unique_domains",
    "collision_count", "ambiguous_match_count", "unmatched_years", "notes",
]

NOT_MEASURED = "NOT_MEASURED_SOURCE_UNAVAILABLE"


def write_audit(registry: dict) -> None:
    """Alias collision audit (§11).

    Only the parts that do not depend on a news corpus can be filled here. The
    corpus-dependent columns are emitted as NOT_MEASURED rather than as 0,
    because an unmeasured coverage number must never be read as a measured zero.
    """
    auto: dict[str, list[str]] = {}
    for company in registry["companies"]:
        for entry in company.get("aliases") or []:
            if entry["class"] in AUTO_MATCH_ALIAS_CLASSES:
                auto.setdefault(entry["alias"].strip().casefold(), []).append(
                    company["ticker"])

    rows = []
    for company in registry["companies"]:
        used = [e["alias"] for e in company.get("aliases") or []
                if e["class"] in AUTO_MATCH_ALIAS_CLASSES]
        collisions = {a: auto[a.strip().casefold()] for a in used
                      if len(auto.get(a.strip().casefold(), [])) > 1}
        rows.append({
            "ticker": company["ticker"],
            "canonical_name": company["canonical_company_name"] or "UNRESOLVED",
            "resolution_status": company["resolution_status"],
            "auto_match_eligible": company["auto_match_eligible"],
            "aliases_used": "; ".join(used) or "none",
            "matched_gdelt_or_mediacloud_organisation_strings": NOT_MEASURED,
            "total_matching_documents": NOT_MEASURED,
            "active_date_range": NOT_MEASURED,
            "unique_domains": NOT_MEASURED,
            "collision_count": len(collisions),
            "ambiguous_match_count": len(company.get("ambiguous_aliases") or []),
            "unmatched_years": NOT_MEASURED,
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
    total_collisions = sum(r["collision_count"] for r in rows)
    lines = [
        "# V4 company alias collision audit",
        "",
        "Generated by `scripts/build_v4_company_aliases.py`.",
        "",
        "## 1. Outcome",
        "",
        f"- supervised tickers in the registry: **{len(rows)}**",
        f"- resolved from curated lineage evidence: **{len(resolved)}**",
        f"- unresolved (excluded from automatic matching): **{len(unresolved)}**",
        f"- alias collisions among auto-matched aliases: **{total_collisions}**",
        "",
        "An unresolved ticker is NOT given a guessed name. It is excluded from",
        "automatic matching until official NSE/company evidence is supplied.",
        "",
        "## 2. What could not be measured, and why",
        "",
        "Columns for matched organisation strings, document counts, active date",
        "range, domain diversity and unmatched years require a searchable news",
        "corpus. Media Cloud is reachable but needs a credential this environment",
        "does not have, so those columns are `NOT_MEASURED` -- deliberately not",
        "`0`, because zero would assert a coverage fact that was never tested.",
        "",
        "## 3. Resolved companies",
        "",
        "| ticker | canonical name | aliases used | ambiguous | collisions |",
        "|---|---|---|---|---|",
    ]
    for row in resolved:
        lines.append(f"| {row['ticker']} | {row['canonical_name']} | "
                     f"{row['aliases_used']} | {row['ambiguous_match_count']} | "
                     f"{row['collision_count']} |")
    lines += [
        "",
        "## 4. Unresolved companies (excluded, never guessed)",
        "",
        "| ticker | reason |",
        "|---|---|",
    ]
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
    print(f"wrote {OUT}")
    write_audit(registry)
    (REPO / "results" / "v4" / "pre_covid_sentiment"
     / "company_aliases_resolved.json").write_text(json.dumps(registry, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())