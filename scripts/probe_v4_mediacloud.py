"""Probe Media Cloud BEFORE building any V4 dataset.

WHAT THIS DOES
--------------
1. Establishes reachability of the real API surface (an anonymous request must
   return a JSON auth error, not an HTML page -- that distinction is what proves
   the endpoint exists).
2. Establishes whether a credential is available.  It is read from
   ``$RESEARCH_ROOT/secrets`` or the environment and is NEVER printed, logged or
   written to any artifact.
3. If a credential exists, measures what §2 asks for, per year and per company:
   result count, earliest publish date, latest publish date, unique domains and
   the number of days with matches.
4. Writes ``mediacloud_probe.csv`` and ``MEDIACLOUD_PROBE.md``.

It downloads no corpus. If the credential is missing, every measurement row is
recorded as ``BLOCKED_NO_CREDENTIAL`` rather than being silently omitted, because
a missing measurement must never be mistaken for zero coverage.
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agentic_forecaster.config import load_config
from agentic_forecaster.v4 import V4Track, secrets_root

REGISTRY = REPO / "configs" / "v4" / "source_registry.yaml"
TIMEOUT = 60
USER_AGENT = "mediacloud-probe/1.0 (V4 source feasibility probe)"

COLUMNS = [
    "source_id", "query_id", "ticker", "year", "query_expression",
    "alias_provenance", "alias_confidence", "result_count",
    "earliest_publish_date", "latest_publish_date", "unique_domains",
    "days_with_matches", "status", "detail",
]


def _resolve_token() -> tuple[str | None, str]:
    """Return ``(token, provenance)`` without ever exposing the value."""
    import os

    env = os.environ.get("MEDIACLOUD_API_TOKEN")
    if env and env.strip():
        return env.strip(), "environment:MEDIA_CLOUD_API_TOKEN"
    candidate = secrets_root() / "mediacloud_token"
    if candidate.exists():
        token = candidate.read_text().strip()
        if token:
            return token, f"file:{candidate}"
    return None, "absent"


def _get(base: str, endpoint: str, params: dict,
         token: str | None) -> tuple[int, dict | str]:
    """One GET. Returns ``(status, parsed_json_or_text)``; never raises."""
    url = base + endpoint
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Token {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body = response.read().decode("utf-8", "replace")
            status = response.status
    except urllib.error.HTTPError as exc:
        status, body = exc.code, exc.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, f"transport-error: {type(exc).__name__}"
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body


def _json_api_alive(base: str) -> tuple[bool, str]:
    """An anonymous call must return a JSON auth error, not the SPA shell."""
    status, payload = _get(base, "search/total-count",
                           {"query": '"Reliance Industries"',
                            "start_date": "20150101", "end_date": "20150131"},
                           token=None)
    if isinstance(payload, dict) and "detail" in payload:
        return True, f"HTTP {status} JSON {json.dumps(payload)[:120]}"
    if isinstance(payload, str) and payload.lstrip().startswith("<"):
        return False, (f"HTTP {status} returned an HTML page, not the JSON API "
                       f"(endpoint surface not served)")
    return False, f"HTTP {status} unexpected payload: {str(payload)[:120]}"


def _probe_year(base: str, token: str, query: str, year: int) -> dict:
    """Count, coverage and domain statistics for one query-year."""
    start, end = date(year, 1, 1), date(year, 12, 31)
    common = {"query": query, "start_date": start.strftime("%Y%m%d"),
              "end_date": end.strftime("%Y%m%d")}

    status, total = _get(base, "search/total-count", common, token)
    if status != 200 or not isinstance(total, dict) or "count" not in total:
        detail = (f"total-count failed HTTP {status}: {str(total)[:120]}")
        return {"result_count": None, "earliest_publish_date": None,
                "latest_publish_date": None, "unique_domains": None,
                "days_with_matches": None, "status": "QUERY_FAILED",
                "detail": detail}

    _, counts = _get(base, "search/count-over-time",
                     dict(common, interval="daily"), token)
    days = None
    if isinstance(counts, dict):
        series = (counts.get("count_over_time") or {}).get("counts") or []
        days = sum(1 for point in series if (point.get("count") or 0) > 0)

    _, sources = _get(base, "search/count-by-source-week", common, token)
    domains = None
    if isinstance(sources, dict):
        buckets = sources.get("counts") or sources.get("by_source") or []
        names = {str(b.get("source_name") or b.get("domain") or "").strip()
                 for b in buckets if isinstance(b, dict)}
        domains = len({n for n in names if n}) or None

    earliest = latest = None
    for sort_order, assign in (("asc", "first"), ("desc", "last")):
        _, listed = _get(base, "search/story-list",
                         dict(common, sort_order=sort_order, page_size=10), token)
        if isinstance(listed, dict):
            stories = listed.get("stories") or []
            dates = [str(s.get("publish_date") or "")[:10] for s in stories
                     if s.get("publish_date")]
            if dates:
                if assign == "first":
                    earliest = min(dates)
                else:
                    latest = max(dates)

    return {"result_count": total.get("count"), "earliest_publish_date": earliest,
            "latest_publish_date": latest, "unique_domains": domains,
            "days_with_matches": days, "status": "OK", "detail": ""}


def main() -> int:
    config = load_config(REGISTRY)
    track = V4Track()
    track.results_root.mkdir(parents=True, exist_ok=True)
    sources = {s["source_id"]: s for s in config["sources"]}
    primary = sources["MEDIA_CLOUD_ONLINE_NEWS_ARCHIVE"]
    base = str(primary["base_url"])
    plan = config["mediacloud_probe"]

    alive, evidence = _json_api_alive(base)
    token, provenance = _resolve_token()
    print(f"Media Cloud API surface alive : {alive}")
    print(f"  evidence                   : {evidence}")
    print(f"credential available          : {bool(token)} ({provenance})")

    rows: list[dict] = []
    auth_detail = ""
    if alive and token:
        status, profile = _get(base, "auth/profile", {}, token)
        auth_detail = f"auth/profile HTTP {status}"
        if status != 200:
            token, provenance = None, "rejected_by_api"
            auth_detail += f" {str(profile)[:120]}"

    queries = [(c["ticker"], c["query_id"] if "query_id" in c else c["ticker"],
                c["query"], c["alias_provenance"], c["alias_confidence"])
               for c in plan["pilot_companies"]]
    queries += [(None, b["query_id"], b["query"], "broad_query", "n/a")
                for b in plan["broad_queries"]]

    for ticker, query_id, query, alias_prov, alias_conf in queries:
        for year in plan["years"]:
            row = {"source_id": primary["source_id"], "query_id": query_id,
                   "ticker": ticker or "", "year": year,
                   "query_expression": query, "alias_provenance": alias_prov,
                   "alias_confidence": alias_conf, "result_count": None,
                   "earliest_publish_date": None, "latest_publish_date": None,
                   "unique_domains": None, "days_with_matches": None,
                   "status": "", "detail": ""}
            if not alive:
                row["status"] = "API_SURFACE_UNREACHABLE"
                row["detail"] = evidence
            elif not token:
                row["status"] = "BLOCKED_NO_CREDENTIAL"
                row["detail"] = (f"no Media Cloud token ({provenance}); coverage "
                                 f"cannot be measured and is NOT zero")
            else:
                row.update(_probe_year(base, token, query, year))
            rows.append(row)

    csv_path = track.path("mediacloud_probe_csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    blocked = sum(1 for r in rows if r["status"] == "BLOCKED_NO_CREDENTIAL")
    measured = [r for r in rows if r["status"] == "OK"]
    years_measured = sorted({r["year"] for r in measured})
    earliest = min((r["earliest_publish_date"] for r in measured
                    if r["earliest_publish_date"]), default=None)

    lines = [
        "# Media Cloud source probe (V4 primary company-news source)",
        "",
        "Generated by `scripts/probe_v4_mediacloud.py`. No corpus was downloaded.",
        "",
        "## 1. Reachability",
        "",
        f"- API base URL: `{base}`",
        f"- JSON API surface confirmed: **{alive}**",
        f"- evidence: `{evidence}`",
        "",
        "A JSON `401` (rather than an HTML page) is what proves the endpoint",
        "surface is real: this host serves an SPA shell for unknown paths, so an",
        "HTML body would have proved the API was not being served at all.",
        "",
        "## 2. Authentication",
        "",
        f"- authentication required: **{'YES' if alive else 'UNKNOWN'}**",
        f"- auth scheme: `{primary['auth_scheme']}`",
        (f"- credential expected at: `{primary['credential_file']}` or "
         f"`{primary['credential_env']}`"),
        f"- credential present: **{'YES' if token else 'NO'}** (source: {provenance})",
        f"- credential validation: {auth_detail or 'not attempted (no credential)'}",
        "",
        "The credential is never printed, logged or written to any artifact.",
        "",
        "## 3. Coverage measurements",
        "",
        f"- planned query-year cells: **{len(rows)}**",
        f"- measured cells: **{len(measured)}**",
        f"- cells blocked for lack of credential: **{blocked}**",
        (f"- earliest searchable historical date observed: "
         f"**{earliest or 'UNKNOWN'}**"),
        f"- years with any measurement: **{years_measured or 'none'}**",
        "",
    ]
    if not measured:
        lines += [
            "No coverage number could be measured. Historical coverage is",
            "**UNKNOWN, not zero** -- and a zero must never be assumed, because",
            "the distinguishing measurement was never made.",
            "",
        ]
    lines += [
        "## 4. Per-year, per-company probe table",
        "",
        ("| year | ticker | query | result_count | earliest | latest | "
         "unique_domains | days_with_matches | status |"),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['year']} | {row['ticker'] or '-'} | "
            f"`{row['query_expression'][:38]}` | "
            f"{row['result_count'] if row['result_count'] is not None else 'n/a'} | "
            f"{row['earliest_publish_date'] or 'n/a'} | "
            f"{row['latest_publish_date'] or 'n/a'} | "
            f"{row['unique_domains'] if row['unique_domains'] is not None else 'n/a'} | "
            f"{row['days_with_matches'] if row['days_with_matches'] is not None else 'n/a'} | "
            f"{row['status']} |")
    lines += [
        "",
        "## 5. Bare tickers never used as queries",
        "",
    ]
    for company in plan["pilot_companies"]:
        forbidden = company.get("bare_ticker_forbidden")
        if forbidden:
            lines.append(f"- `{company['ticker']}`: bare `{forbidden}` is "
                         f"AMBIGUOUS and is not queried.")
    lines += [
        "",
        f"Machine-readable rows: `{csv_path.name}` ({len(rows)} rows).",
        "",
    ]
    md_path = track.path("mediacloud_probe")
    md_path.write_text("\n".join(lines))

    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())