"""Point-in-time news/event availability rules for V4.

THE ONE RULE
-------------
The prediction timestamp is AFTER the NSE close on trading day ``t``.  News
carries a publication DATE but no reliable publication TIME, so the rule is
conservative and absolute::

    publish_date < t          allowed
    publish_date == t         FORBIDDEN   (same-day timing is ambiguous)
    publish_date > t          FORBIDDEN   (future news)

``indexed_date`` is provenance only.  It is never treated as a publication time,
and a story whose ``publish_date`` is missing or implausible is rejected rather
than repaired.

Windows are CALENDAR-day windows, because news exists seven days a week.  Each
window is half-open ``[t - window, t)`` and therefore always excludes the origin
day itself.
"""

from __future__ import annotations

from datetime import date, timedelta

FINAL_ALLOWED_DATE = date(2019, 12, 31)
LOCKBOX_YEAR = 2019


class PostCutoffDataAccessError(RuntimeError):
    """Raised when news/event data from 2019 (dev) or 2020+ is touched."""


class SameDayNewsError(RuntimeError):
    """Raised when same-day news would enter an after-close origin."""


def assert_news_before_origin(publish_date: date, origin_date: date, *,
                              final_allowed: date = FINAL_ALLOWED_DATE,
                              allow_lockbox_year: bool = False) -> None:
    """Validate one publication date against one origin date."""
    if publish_date is None:
        raise ValueError("publish_date is required; indexed_date is provenance "
                         "only and must never substitute for it")
    if publish_date >= origin_date:
        raise SameDayNewsError(
            f"news dated {publish_date} may not enter origin {origin_date}: "
            f"the origin is after the NSE close, so only strictly earlier "
            f"publication dates are available")
    if publish_date > final_allowed:
        raise PostCutoffDataAccessError(
            f"news dated {publish_date} is after the absolute final allowed "
            f"date {final_allowed}")
    if not allow_lockbox_year and publish_date.year == LOCKBOX_YEAR:
        raise PostCutoffDataAccessError(
            f"news dated {publish_date} is inside the sealed {LOCKBOX_YEAR} "
            f"lockbox year")


def news_window(origin_date: date, window_days: int) -> tuple[date, date]:
    """Half-open calendar window ``[t - window_days, t)`` for news."""
    if window_days < 1:
        raise ValueError(f"window_days must be >= 1, got {window_days}")
    return origin_date - timedelta(days=window_days), origin_date


def in_news_window(publish_date: date, origin_date: date,
                   window_days: int) -> bool:
    """True when ``publish_date`` lies in the causal window for ``origin_date``.

    The strict ``<`` comparison is applied again here so that a window can never
    widen into the same day, whatever the caller passes in.
    """
    if publish_date >= origin_date:
        return False
    start, _end = news_window(origin_date, window_days)
    return publish_date >= start