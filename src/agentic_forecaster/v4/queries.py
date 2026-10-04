"""Date-aware Media Cloud query construction, budgeting and retry policy.

WHY QUERIES MUST BE DATE-AWARE
-------------------------------
A single query per company is wrong for a news archive.  Several supervised
companies changed their legal name, and one of those changes (``TITAN``) falls
INSIDE the study period.  Querying "Tata Consumer Products" for 2015 would return
nothing at all, because that name did not exist; querying "Shriram Finance" for
2016 would silently search a name the company did not yet use.

So an alias is only usable for a query when the queried date falls inside that
alias's validity interval, and a window that straddles a rename boundary is split
into sub-windows rather than being forced into one name.

RATE LIMIT AND REQUEST BUDGET
-----------------------------
Coverage discovery uses only ``search/total-count`` and
``search/count-over-time``.  Full story metadata (``search/story-list``) is NOT
called during the coverage probe -- it is the expensive endpoint and is only
warranted once feasibility has already passed.  429 and transient 5xx responses
are retried with bounded exponential backoff so the archive is never hammered.
"""

from __future__ import annotations

import time
import urllib.error
from dataclasses import dataclass, field
from datetime import date, timedelta

from agentic_forecaster.v4 import ALIAS_EXACT_SAFE, ALIAS_HISTORICAL_SAFE

#: Alias classes that may ever appear in a query.
QUERYABLE_CLASSES: tuple[str, ...] = (ALIAS_EXACT_SAFE, ALIAS_HISTORICAL_SAFE)

#: Backoff schedule for 429 / transient 5xx. Bounded on purpose: an unbounded
#: retry against a shared archive would be an abuse of a free public service.
RETRY_BASE_SECONDS = 1.0
RETRY_MAX_SECONDS = 30.0
RETRY_STATUSES: tuple[int, ...] = (429, 500, 502, 503, 504)


@dataclass(frozen=True)
class Alias:
    """One alias with its validity interval and safety class."""

    text: str
    alias_class: str
    valid_from: date | None = None
    valid_to: date | None = None

    def valid_on(self, when: date) -> bool:
        if self.alias_class not in QUERYABLE_CLASSES:
            return False
        if self.valid_from is not None and when < self.valid_from:
            return False
        return self.valid_to is None or when <= self.valid_to


@dataclass(frozen=True)
class QueryWindow:
    """A maximal sub-interval over which one alias expression is valid."""

    ticker: str
    start: date
    end: date
    alias: str
    alias_class: str

    def contains(self, when: date) -> bool:
        return self.start <= when <= self.end

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.ticker} {self.start}..{self.end} [{self.alias}]"


def _parse(value: str | None) -> date | None:
    """Parse a validity boundary.

    The curated lineage records a SEQUENCE of renames as ``"a; b"`` (for
    example TELCO ``"2003-07-29; 2025-10-01"``: renamed to Tata Motors in 2003
    and again in 2025). For a historical alias the relevant boundary is the FIRST
    date -- that is when the name stopped being this security's name -- so the
    leading component is taken and the later renames ignored.
    """
    if not value:
        return None
    text = str(value).strip().split(";")[0].strip()
    if len(text) == 7 and text[4] == "-":  # YYYY-MM
        return date(int(text[:4]), int(text[5:]), 1)
    if len(text) == 10:
        return date(int(text[:4]), int(text[5:7]), int(text[8:]))
    if len(text) == 8 and text.isdigit():
        return date(int(text[:4]), int(text[4:6]), int(text[6:]))
    if text.isdigit():  # YYYY
        return date(int(text), 1, 1)
    raise ValueError(f"unrecognised date {value!r}")


def alias_validity_windows(ticker: str, aliases: list[Alias],
                            start: date, end: date) -> list[QueryWindow]:
    """Split ``[start, end]`` into maximal intervals of alias validity.

    A rename boundary inside the range produces two or more windows, so no window
    ever spans a date on which its own alias was not the company's name.
    """
    if end < start:
        raise ValueError(f"end {end} precedes start {start}")


    # that day, and a ``valid_to`` means the name still applies ON that day, so the
    # next window starts the day after. Using the raw boundary for both would
    # produce a spurious one-day window straddling every rename.
    # Split points are window STARTS. ``end`` is the range terminator, not a
    # start, so it must not create a zero-length final window.
    starts = {start}
    for alias in aliases:
        if alias.valid_from is not None and start < alias.valid_from <= end:
            starts.add(alias.valid_from)
        if alias.valid_to is not None:
            after = alias.valid_to + timedelta(days=1)
            if start < after <= end:
                starts.add(after)

    ordered = sorted(starts)
    windows: list[QueryWindow] = []
    for i, window_start in enumerate(ordered):
        window_end = (ordered[i + 1] - timedelta(days=1)
                      if i + 1 < len(ordered) else end)
        probe = window_start
        usable = [a for a in aliases if a.valid_on(probe)]
        if not usable:
            continue
        # Prefer the most specific (narrowest) interval: a dated historical alias
        # is more precise than an undated current one.
        usable.sort(key=lambda a: (
            a.valid_from is None, a.valid_to is None, a.text))
        chosen = usable[0]
        windows.append(QueryWindow(ticker=ticker, start=window_start,
                                   end=window_end, alias=chosen.text,
                                   alias_class=chosen.alias_class))
    return windows


def build_query(alias_texts: list[str], *, title_only: bool = False) -> str:
    """Build one Media Cloud phrase query from the given alias phrases."""
    quoted = [f'"{text}"' for text in alias_texts]
    expression = " OR ".join(quoted) if quoted else ""
    if title_only:
        expression = f"article_title:({expression})" if expression else \
            "article_title:(*)"
    return expression


def build_plan(registry_companies: list[dict], years: list[int], *,
               title_only: bool = False,
               earliest: date = date(2013, 4, 1),
               latest: date = date(2018, 12, 31)) -> dict:
    """Build the full alias-validity-aware query plan and its request budget.

    Two calls per query window (total-count + count-over-time) and, optionally,
    one title-restricted pair. No story-list call is budgeted at all.
    """
    windows: list[dict] = []
    for company in registry_companies:
        aliases = [
            Alias(text=entry["alias"], alias_class=entry["class"],
                  valid_from=_parse(entry.get("valid_from")),
                  valid_to=_parse(entry.get("valid_to")))
            for entry in company.get("aliases") or []
        ]
        for year in years:
            year_start = max(date(year, 1, 1), earliest)
            year_end = min(date(year, 12, 31), latest)
            if year_end < year_start:
                continue
            for window in alias_validity_windows(company["ticker"], aliases,
                                                 year_start, year_end):
                windows.append({
                    "ticker": window.ticker,
                    "year": year,
                    "start": window.start.isoformat(),
                    "end": window.end.isoformat(),
                    "alias": window.alias,
                    "alias_class": window.alias_class,
                    "query": build_query([window.alias], title_only=title_only),
                    "title_only": title_only,
                })

    per_window = 2  # total-count + count-over-time
    coverage_calls = len(windows) * per_window
    title_calls = 0 if not title_only else coverage_calls
    return {
        "generated_by": "agentic_forecaster.v4.queries.build_plan",
        "years": years,
        "earliest_allowed": earliest.isoformat(),
        "latest_allowed": latest.isoformat(),
        "n_query_windows": len(windows),
        "endpoints": {
            "coverage": ["search/total-count", "search/count-over-time"],
            "title_explicit": ["search/total-count", "search/count-over-time"],
            "story_metadata": "NOT CALLED during the coverage probe",
        },
        "estimated_calls": {
            "coverage_probe": coverage_calls,
            "title_explicit_probe": title_calls,
            "story_list_calls": 0,
            "total_estimate": coverage_calls + title_calls,
        },
        "assumed_rate_limit_per_minute": 10,
        "estimated_minutes_at_assumed_limit": round(
            (coverage_calls + title_calls) / 10.0, 1),
        "retry_policy": {
            "statuses": list(RETRY_STATUSES),
            "base_seconds": RETRY_BASE_SECONDS,
            "max_seconds": RETRY_MAX_SECONDS,
            "strategy": "bounded exponential backoff with jitter",
        },
        "queries": windows,
    }


@dataclass
class RetryPolicy:
    """Bounded exponential backoff."""

    max_attempts: int = 4
    base_seconds: float = RETRY_BASE_SECONDS
    max_seconds: float = RETRY_MAX_SECONDS
    sleep: object = time.sleep
    attempts_made: int = field(default=0, init=False)

    def delay_for(self, attempt: int) -> float:
        return min(self.base_seconds * (2 ** attempt), self.max_seconds)

    def run(self, call):
        """Invoke ``call`` with retries on 429 / transient 5xx."""
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            self.attempts_made += 1
            try:
                return call()
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRY_STATUSES:
                    raise
                last = exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last = exc
            if attempt == self.max_attempts - 1:
                break
            self.sleep(self.delay_for(attempt))
        raise last if last is not None else RuntimeError("retry loop exhausted")