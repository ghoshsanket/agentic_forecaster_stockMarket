#!/usr/bin/env python3
"""PROBE every declared V3 external source and write the SOURCE QUALITY AUDIT.

Nothing is assumed to work.  For each declared source this reports whether the
provider actually returns data, its real first/last dates, row count, missing
fraction, duplicate dates and suspicious jumps -- and then applies the COVERAGE
GATE (mandatory 2014-2018 coverage, missing fraction < 5 %, no unexplained
multi-month gap).

Failed probes are NEVER hidden: every attempt appears in the audit with its reason.

Outputs::

    results/v3/pre_covid_exogenous/SOURCE_AUDIT.md
    results/v3/pre_covid_exogenous/source_audit.json
    configs/v3/source_registry_resolved.json   (probe results merged back)

Usage::

    uv run python scripts/probe_v3_sources.py
    uv run python scripts/probe_v3_sources.py --no-store     # probe only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.utils import atomic_json_dump, ensure_dir, setup_logging
from agentic_forecaster.v3 import V3Track, data_root, raw_root
from agentic_forecaster.v3.download import probe_registry, write_manifest, write_provenance_document
from agentic_forecaster.v3.sources import load_registry

CONFIG = REPO_ROOT / "configs" / "v3" / "source_registry.yaml"


def _table(rows: list[dict], columns: tuple[str, ...]) -> list[str]:
    """Render a markdown table from dict rows."""
    out = ["| " + " | ".join(columns) + " |",
           "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(column, "-")) for column in columns)
                   + " |")
    return out


def render_audit(registry, results: list[dict], *, final_allowed: str) -> str:
    """The SOURCE_QUALITY_REPORT of section 46, for EVERY attempted source."""
    n_accepted = sum(1 for r in results if r["accepted"])
    lines: list[str] = []
    add = lines.append

    add("# V3 EXOGENOUS SOURCE AUDIT")
    add("")
    add("Every external source attempted, whether or not it worked. A rejected "
        "source is excluded, never imputed, proxied or silently replaced.")
    add("")
    add(f"- absolute final allowed date: `{final_allowed}`")
    add("- download window end (exclusive): `2020-01-01`")
    add(f"- declared: **{len(results)}** | accepted: **{n_accepted}** | rejected: "
        f"**{len(results) - n_accepted}**")
    add("")
    lines.extend(_table(results, (
        "source_id", "provider", "identifier", "available", "first_actual_date",
        "last_actual_date", "row_count", "missing_fraction", "duplicate_dates",
        "suspicious_jump_count", "availability_class", "accepted",
        "exclusion_reason")))

    add("")
    add("## Accepted sources and their final lag rule")
    add("")
    accepted_rows = []
    for row in results:
        if not row["accepted"]:
            continue
        spec = registry.get(row["source_id"])
        digest = row["raw_sha256"] or "-"
        accepted_rows.append({"source_id": row["source_id"],
                              "availability_class": row["availability_class"],
                              "final_lag_rule": spec.final_lag_rule,
                              "SHA256": digest[:16] + "..."})
    lines.extend(_table(accepted_rows, ("source_id", "availability_class",
                                        "final_lag_rule", "SHA256")))

    add("")
    add("## Rejected sources and why")
    add("")
    rejected = [r for r in results if not r["accepted"]]
    if rejected:
        lines.extend(_table(rejected, ("source_id", "identifier", "available",
                                       "exclusion_reason")))
    else:
        add("_none_")

    add("")
    add("## Source-quality rules applied")
    add("")
    add("- mandatory valid coverage across every development year 2014-2018")
    add("- missing fraction in the development years below 5 %")
    add("- no unexplained gap longer than 45 calendar days")
    add("- business-day granularity where the session expects it")
    add("- every non-Indian source uses conservative lag1 unless a documented earlier "
        "session close is recorded in its availability class")
    add("- the reconstructed equal-weight stock proxy is NOT used and is NOT called "
        "an index")
    add("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--no-store", action="store_true",
                        help="probe without writing raw snapshots")
    parser.add_argument("--max-missing-fraction", type=float, default=0.05)
    args = parser.parse_args(argv)
    setup_logging()

    registry = load_registry(args.config)
    track = V3Track()
    ensure_dir(track.results_root)

    probes = probe_registry(registry, raw_root=raw_root(), store=not args.no_store,
                            max_missing_fraction=args.max_missing_fraction)
    rows = []
    for probe in probes:
        spec = registry.get(probe.spec.source_id)
        row = probe.to_dict()
        row["accepted"] = bool(spec.accepted)
        row["availability_class"] = spec.availability_policy
        row["final_lag_rule"] = spec.final_lag_rule
        row["exclusion_reason"] = spec.exclusion_reason
        rows.append(row)
        print(f"[probe] {spec.source_id:<20} {spec.provider_symbol_or_identifier:<16} "
              f"{'ACCEPTED' if spec.accepted else 'REJECTED':<9} "
              f"rows={probe.row_count or 0:<6} "
              f"{probe.first_actual_date or '-'} .. {probe.last_actual_date or '-'} "
              f"{'' if spec.accepted else spec.exclusion_reason}", flush=True)

    audit = {
        "track": "V3_EXOGENOUS_PRECOVID",
        "final_allowed_date": "2019-12-31",
        "download_end_exclusive": "2020-01-01",
        "n_declared": len(rows),
        "n_accepted": sum(1 for r in rows if r["accepted"]),
        "n_rejected": sum(1 for r in rows if not r["accepted"]),
        "coverage_gate": {"development_years": [2014, 2018],
                          "max_missing_fraction": args.max_missing_fraction,
                          "max_unexplained_gap_days": 45},
        "sources": rows,
        "raw_root": str(raw_root()),
        "no_failed_probe_hidden": True,
    }
    atomic_json_dump(audit, track.path("source_audit_json"))
    track.path("source_audit").write_text(
        render_audit(registry, rows, final_allowed="2019-12-31"), encoding="utf-8")

    manifest = write_manifest(registry, manifests_root=data_root() / "manifests")
    write_provenance_document(manifests_root=data_root() / "manifests", registry=registry)
    atomic_json_dump(registry.to_dict(), track.path("registry"))

    print(json.dumps({"declared": audit["n_declared"], "accepted": audit["n_accepted"],
                      "rejected": audit["n_rejected"],
                      "manifest_sha256": manifest["manifest_sha256"],
                      "source_audit": str(track.path("source_audit"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())