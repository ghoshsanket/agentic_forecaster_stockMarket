"""Probe Media Cloud coverage BEFORE downloading any V4 corpus.

SEQUENCED GATES
---------------
1. Reachability of the real JSON API (anonymous 401, not an HTML shell).
2. Credential presence, file permissions, and validation via ``auth/profile``.
3. Request budget, written to ``request_plan.json`` BEFORE any real query.
4. Collection discovery, so an India/financial-media collection can be frozen
   rather than searching the whole global archive blindly.
5. Date-aware coverage probe over the alias registry.
6. Title-explicit feasibility probe.

DISCIPLINES ENFORCED HERE
-------------------------
* The token is never printed, logged, or written to any artifact. Only its
  provenance and a PASS/FAIL validation result are recorded.
* With no token, NO authenticated call is attempted at all. Every planned
  query-year is recorded as ``BLOCKED_NO_CREDENTIAL``, never as zero coverage.
* Coverage uses only ``total-count`` and ``count-over-time``. ``story-list`` is
  not called during the probe: it is the expensive endpoint and is only
  warranted once feasibility has already passed.
* Queries are alias-validity-aware, so a renamed company is never searched under
  a name it did not yet use.
* 429 and transient 5xx are retried with bounded exponential backoff.
"""

from __future__ import annotations

import csv
import json
import os
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml

from agentic_forecaster.config import load_config
from agentic_forecaster.v4 import V4Track, secrets_root
from agentic_forecaster.v4.queries import RetryPolicy, build_plan

REGISTRY = REPO / "configs" / "v4" / "source_registry.yaml"
ALIASES = REPO / "configs" / "v4" / "company_aliases.yaml"
TIMEOUT = 60
USER_AGENT = "mediacloud-probe/2.0 (V4 source feasibility probe)"
YEARS = [2013, 2014, 2015, 2016, 2017, 2018]

COLUMNS = [
    "ticker", "year", "window_start", "window_end", "alias_used",
    "alias_class", "query_expression", "title_only",
    "general_count", "title_explicit_count", "earliest_publish_date",
    "latest_publish_date", "days_with_matches", "unique_domains",
    "status", "detail",
]


class CredentialError(RuntimeError):
    """Raised when a credential exists but is unusable or unsafe."""


def resolve_token() -> tuple[str | None, str, str]:
    """Return ``(token, provenance, permissions_note)`` without leaking it."""
    env = os.environ.get("MEDIACLOUD_API_TOKEN")
    if env and env.strip():
        return env.strip(), "environment:MEDIACloud_TOKEN", "env (not a file)"

    candidate = secrets_root() / "mediacloud_token"
    if not candidate.exists():
        return None, "absent", "no token file"
    token = candidate.read_text().strip()
    if not token:
        return None, "empty", "token file exists but is empty"

    mode = stat.S_IMODE(candidate.stat().st_mode)
    if mode & 0o077:
        raise CredentialError(
            f"{candidate} is mode {mode:04o}; a credential file must be 0600 "
            f"because any local user could otherwise read it")
    return token, f"file:{candidate}", f"mode {mode:04o}"


def _get(base: str, endpoint: str, params: dict, token: str | None, *,
         retry: RetryPolicy | None = None) -> tuple[int, object]:
    """One GET with bounded backoff. Returns ``(status, json_or_text)``."""
    url = base + endpoint
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Token {token}"

    def _call() -> tuple[int, object]:
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                status = response.status
                body = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            status, body = exc.code, exc.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # A transport failure must be reported, never raised: an unreachable
            # archive is a probe finding, not a crash.
            return 0, f"transport-error: {type(exc).__name__}: {exc}"
        try:
            return status, json.loads(body)
        except json.JSONDecodeError:
            return status, body

    if retry is None:
        return _call()
    try:
        return retry.run(_call)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
            RuntimeError) as exc:
        return 0, f"request-failed-after-retries: {type(exc).__name__}"


def api_surface_alive(base: str) -> tuple[bool, str]:
    """An anonymous call must return a JSON auth error, not the SPA shell."""
    status, payload = _get(base, "search/total-count",
                           {"query": '"Reliance Industries"',
                            "start_date": "20150101", "end_date": "20150131"},
                           token=None)
    if status == 0:
        return False, str(payload)[:160]
    if isinstance(payload, dict) and "detail" in payload:
        return True, f"HTTP {status} JSON {json.dumps(payload)[:110]}"
    if isinstance(payload, str) and payload.lstrip().startswith("<"):
        return False, (f"HTTP {status} returned HTML, not the JSON API")
    return False, f"HTTP {status} unexpected: {str(payload)[:110]}"


def probe_collections(base: str, token: str) -> list[dict]:
    """Discover India / business-news collections to freeze by ID."""
    found: list[dict] = []
    for name in ("india", "indian", "business", "financial", "market"):
        status, payload = _get(base, "sources/collections/",
                               {"name": name, "page_size": 50}, token,
                               retry=RetryPolicy())
        if status == 200 and isinstance(payload, dict):
            for row in payload.get("results") or []:
                found.append({
                    "collection_id": row.get("id") or row.get("collection_id"),
                    "label": row.get("label") or row.get("name"),
                    "source_count": row.get("story_count") or row.get("source_count"),
                    "notes": f"matched on name~{name}",
                })
    seen: set[object] = set()
    unique = []
    for row in found:
        if row["collection_id"] in seen:
            continue
        seen.add(row["collection_id"])
        unique.append(row)
    return unique


def main() -> int:
    config = load_config(REGISTRY)
    track = V4Track()
    track.results_root.mkdir(parents=True, exist_ok=True)
    primary = {s["source_id"]: s for s in config["sources"]}[
        "MEDIA_CLOUD_ONLINE_NEWS_ARCHIVE"]
    base = str(primary["base_url"])

    aliases_doc = yaml.safe_load(ALIASES.read_text())
    companies = [c for c in aliases_doc["companies"] if c["auto_match_eligible"]]

    alive, evidence = api_surface_alive(base)
    print(f"Media Cloud JSON API surface alive : {alive}")
    print(f"  evidence                         : {evidence}")

    token_error = ""
    try:
        token, provenance, permissions = resolve_token()
    except CredentialError as exc:
        token, provenance, permissions = None, "rejected", str(exc)
        token_error = str(exc)
    print(f"credential available               : {bool(token)} ({provenance})")

    profile_status, profile_payload = (0, "not attempted (no credential)")
    if token:
        profile_status, profile_payload = _get(base, "auth/profile", {}, token,
                                               retry=RetryPolicy())
    print(f"auth/profile                       : HTTP {profile_status}")

    plan = build_plan(companies, YEARS, title_only=False)
    title_plan = build_plan(companies, YEARS, title_only=True)
    plan["title_explicit_plan"] = {
        "n_query_windows": title_plan["n_query_windows"],
        "note": title_plan["estimated_calls"],
    }
    plan["collections_probe_calls"] = 5
    plan_path = track.results_root / "request_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    print(f"request plan                       : {plan['estimated_calls']}")
    print(f"wrote {plan_path}")

    collections: list[dict] = []
    if token:
        collections = probe_collections(base, token)
        print(f"collections matched                : {len(collections)}")

    rows: list[dict] = []

    def _blank(ticker: str, window: dict, status: str, detail: str) -> dict:
        return {
            "ticker": ticker, "year": window["year"],
            "window_start": window["start"], "window_end": window["end"],
            "alias_used": window["alias"], "alias_class": window["alias_class"],
            "query_expression": window["query"],
            "title_only": window["title_only"],
            "general_count": None, "title_explicit_count": None,
            "earliest_publish_date": None, "latest_publish_date": None,
            "days_with_matches": None, "unique_domains": None,
            "status": status, "detail": detail,
        }

    for window in plan["queries"]:
        if not alive:
            rows.append(_blank(window["ticker"], window, "API_SURFACE_UNREACHABLE",
                               evidence))
        elif not token:
            rows.append(_blank(window["ticker"], window, "BLOCKED_NO_CREDENTIAL",
                               f"no Media Cloud token ({provenance}); coverage "
                               f"unknown and NOT zero"))
        else:
            params = {"query": window["query"],
                      "start_date": window["start"].replace("-", ""),
                      "end_date": window["end"].replace("-", "")}
            status, total = _get(base, "search/total-count", params, token,
                                 retry=RetryPolicy())
            if status != 200 or not isinstance(total, dict):
                rows.append(_blank(window["ticker"], window, "QUERY_FAILED",
                                   f"total-count HTTP {status}"))
                continue
            _, over_time = _get(base, "search/count-over-time",
                                dict(params, interval="daily"), token,
                                retry=RetryPolicy())
            days = None
            if isinstance(over_time, dict):
                series = (over_time.get("count_over_time") or {}).get("counts") or []
                days = sum(1 for point in series if (point.get("count") or 0) > 0)
            _, title_total = _get(
                base, "search/total-count",
                dict(params, query=window["query"].replace(
                    '"', 'article_title:"', 1)), token, retry=RetryPolicy())
            title_count = (title_total.get("total")
                           if isinstance(title_total, dict) else None)
            row = _blank(window["ticker"], window, "OK", "")
            row.update({
                "general_count": total.get("count"),
                "title_explicit_count": title_count,
                "days_with_matches": days,
            })
            rows.append(row)

    csv_path = track.path("mediacloud_probe_csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    measured = [r for r in rows if r["status"] == "OK"]
    blocked = [r for r in rows if r["status"] == "BLOCKED_NO_CREDENTIAL"]
    by_year: dict[int, dict] = {}
    for row in measured:
        stats = by_year.setdefault(row["year"], {"general": 0, "title": 0,
                                                 "days": [], "tickers": 0})
        stats["tickers"] += 1
        stats["general"] += row["general_count"] or 0
        stats["title"] += row["title_explicit_count"] or 0
        if row["days_with_matches"] is not None:
            stats["days"].append(row["days_with_matches"])

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
        "## 2. Credential",
        "",
        "- authentication required: **YES**",
        f"- credential present: **{'YES' if token else 'NO'}** ({provenance})",
        f"- permissions: {permissions}",
        (f"- `auth/profile` result: **HTTP {profile_status}**"
         + ("" if not token
            else f" ({profile_payload if profile_status != 200 else 'OK'})")),
        "",
        "The token value is never printed, logged or written to any artifact.",
        "",
    ]
    if token_error:
        lines += [f"- credential rejected: `{token_error}`", ""]
    lines += [
        "## 3. Alias registry",
        "",
        f"- supervised tickers: **{aliases_doc['n_tickers']}**",
        f"- resolved (auto-match eligible): **{aliases_doc['n_resolved']}**",
        f"- unresolved: **{aliases_doc['n_unresolved']}**",
        "- hard feasibility floor: **25** resolved",
        "",
        "## 4. Request budget (written before any query)",
        "",
        f"- planned alias-validity query windows: **{plan['n_query_windows']}**",
        f"- estimated calls: **{plan['estimated_calls']['total_estimate']}**",
        f"- `story-list` calls budgeted: **{plan['estimated_calls']['story_list_calls']}**",
        (f"- assumed limit: {plan['assumed_rate_limit_per_minute']}/min "
         f"(~{plan['estimated_minutes_at_assumed_limit']} min)"),
        f"- plan file: `{plan_path.name}`",
        "",
        "## 5. Collections",
        "",
        (f"- matching collections discovered: **{len(collections)}**"
         if collections else
         ("- collection discovery **not performed** (no credential). Collection "
          "IDs must be frozen in `configs/v4/source_registry.yaml` before any "
          "production query; names alone are not reproducible.")),
        "",
    ]
    for row in collections[:20]:
        lines.append(f"- id `{row['collection_id']}` — {row['label']} "
                     f"({row['notes']})")
    lines += [
        "",
        "## 6. Coverage by year",
        "",
        ("| year | tickers probed | general count | title-explicit count | "
         "median days with matches |"),
        "|---|---|---|---|---|",
    ]
    for year in YEARS:
        stats = by_year.get(year)
        if not stats:
            lines.append(f"| {year} | 0 | not measured | not measured | "
                         "not measured |")
            continue
        days = stats["days"]
        median = (sorted(days)[len(days) // 2] if days else "n/a")
        lines.append(f"| {year} | {stats['tickers']} | {stats['general']} | "
                     f"{stats['title']} | {median} |")
    lines += [
        "",
        "## 7. Query windows (alias-validity aware)",
        "",
        f"- rows recorded: **{len(rows)}**",
        f"- measured: **{len(measured)}**",
        f"- blocked for lack of credential: **{len(blocked)}**",
        "",
    ]
    if blocked:
        lines += [
            ("Coverage is **UNKNOWN, not zero**: the distinguishing measurement "
             "was never made, because no credential is present."),
            "",
        ]
    lines += [
        "## 8. Verdict inputs",
        "",
        (f"- sentiment-eligible tickers: "
         f"**{'not computable without coverage' if not measured else len({r['ticker'] for r in measured})}**"),
        (f"- usable development validation years: "
         f"**{'not computable without coverage' if not measured else sorted(by_year)}**"),
        "",
        f"Machine-readable rows: `{csv_path.name}`.",
        "",
    ]
    md_path = track.path("mediacloud_probe")
    md_path.write_text("\n".join(lines))
    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())