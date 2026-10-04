"""V2 firewalls: the paper-test firewall and the V2 2021 lockbox firewall.

Two independent protections, both enforced in code rather than by convention.

1. HARD PAPER-TEST FIREWALL (``V2TestFirewallError``)
   Any V2 target date on or after ``2022-01-01`` is forbidden for the whole of
   V2 development.  It is not "refused politely": it raises immediately and
   aborts the run, so a leak cannot be silently averaged into a result.  The
   2022/2023 paper test years remain untouched in this project.

2. V2 LOCKBOX FIREWALL (``V2LockboxFirewallError``)
   2021 is the V2 architecture lockbox.  By default the development script
   refuses to score any 2021 target date, so the A/B/C/D/E/F comparison cannot
   be tuned against the lockbox year.  Only ``scripts/run_v2_lockbox.py`` with
   an explicit ``V2_LOCKBOX=1`` may score 2021, and that script can never score
   2022 or later.

Both fire on the *target* date, because a target date is what identifies the
supervised outcome.  A sample whose origin is in 2021 but whose target is in
2022 is exactly the leak these firewalls exist to stop.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import pandas as pd

#: First date V2 may never score.  The paper's final test years (2022, 2023)
#: stay unseen for the entire V2 programme.
V2_PAPER_TEST_FIREWALL_START = pd.Timestamp("2022-01-01")

#: First date the V2 development script may not score without the explicit
#: lockbox switch.
V2_LOCKBOX_FIREWALL_START = pd.Timestamp("2021-01-01")

#: The V2 lockbox year, for manifests and reports.
V2_LOCKBOX_YEAR = 2021


class V2FirewallError(RuntimeError):
    """Base class for V2 firewall violations."""


class V2TestFirewallError(V2FirewallError):
    """A V2 run attempted to touch 2022 or later."""

    __test__ = False


class V2LockboxFirewallError(V2FirewallError):
    """A V2 *development* run attempted to score the 2021 lockbox."""

    __test__ = False


def _iter_timestamps(values: Iterable) -> Iterator[pd.Timestamp]:
    for v in values:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        ts = pd.Timestamp(v)
        if pd.isna(ts):
            continue
        yield ts


def max_target_date(values: Iterable) -> str | None:
    """Latest date in ``values`` as an ISO string, or None when empty."""
    stamps = list(_iter_timestamps(values))
    if not stamps:
        return None
    return max(stamps).strftime("%Y-%m-%d")


def assert_no_paper_test_targets(values: Iterable, *, where: str = "v2") -> str | None:
    """Raise on any target date >= 2022-01-01.  Always active, no opt-out.

    Returns the maximum target date seen, for manifests.
    """
    worst = None
    for ts in _iter_timestamps(values):
        worst = ts if worst is None or ts > worst else worst
        if ts >= V2_PAPER_TEST_FIREWALL_START:
            raise V2TestFirewallError(
                f"{where}: refusing target date {ts.date()}. The V2 paper-test "
                f"firewall forbids scoring anything on or after "
                f"{V2_PAPER_TEST_FIREWALL_START.date()}. 2022 and 2023 must remain "
                "untouched for the whole V2 programme."
            )
    return None if worst is None else worst.strftime("%Y-%m-%d")


def lockbox_unlocked(explicit: bool | None = None) -> bool:
    """Resolve whether the 2021 lockbox has been explicitly unlocked.

    Precedence: explicit argument > ``V2_LOCKBOX`` environment variable > False.
    """
    if explicit is not None:
        return bool(explicit)
    return os.environ.get("V2_LOCKBOX", "") == "1"


def assert_no_lockbox_targets(values: Iterable, *, where: str = "v2",
                              unlocked: bool | None = None) -> str | None:
    """Raise on any 2021 target date unless the lockbox is explicitly unlocked.

    The 2022+ firewall above is NOT bypassed by this switch.
    """
    if lockbox_unlocked(unlocked):
        return max_target_date(values)
    worst = None
    for ts in _iter_timestamps(values):
        worst = ts if worst is None or ts > worst else worst
        if ts >= V2_LOCKBOX_FIREWALL_START:
            raise V2LockboxFirewallError(
                f"{where}: refusing target date {ts.date()}. {V2_LOCKBOX_YEAR} is the "
                "V2 architecture LOCKBOX and must not be read while comparing V2-A .. "
                "V2-F. Only scripts/run_v2_lockbox.py with V2_LOCKBOX=1 may score it, "
                "and never for 2022 or later."
            )
    return None if worst is None else worst.strftime("%Y-%m-%d")


def assert_window_safe(window_start, window_end, *, where: str = "v2 window",
                       unlocked: bool | None = None) -> None:
    """Validate a declared split/validation window against both firewalls.

    ``window_end`` is the last date that may be SCORED, so it is compared
    directly against the forbidden boundaries.
    """
    end = pd.Timestamp(window_end)
    assert_no_paper_test_targets([end], where=where)
    assert_no_lockbox_targets([end], where=where, unlocked=unlocked)


@dataclass(frozen=True)
class V2FirewallDescription:
    """Machine-readable firewall description for manifests."""

    paper_test_firewall_start: str = V2_PAPER_TEST_FIREWALL_START.strftime("%Y-%m-%d")
    lockbox_firewall_start: str = V2_LOCKBOX_FIREWALL_START.strftime("%Y-%m-%d")
    lockbox_year: int = V2_LOCKBOX_YEAR
    lockbox_env_var: str = "V2_LOCKBOX"
    protected_years_never_scored: tuple[int, ...] = (2022, 2023)
    note: str = (
        "2021 is the V2 architecture lockbox: readable only by the explicit "
        "lockbox script. 2022 and 2023 are never scored by any V2 script."
    )


def describe_firewall() -> dict:
    return V2FirewallDescription().__dict__ | {"description": V2FirewallDescription.note}