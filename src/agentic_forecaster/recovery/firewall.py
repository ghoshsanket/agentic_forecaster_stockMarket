"""HARD TEST-SET FIREWALL for performance recovery.

The final paper test years are **2022 and 2023**.  They must stay unseen while
model search happens, otherwise the recovered configuration is simply fitted to
the answer and the exercise measures nothing.

This module makes that structural rather than procedural: in ``search_mode``
any attempt to score a date on or after :data:`TEST_FIREWALL_START` raises
:class:`TestSetFirewallError`.  It is enforced in three independent places so a
single oversight cannot leak:

1. :func:`assert_pre_test_dates` - called by every metric computation.
2. :func:`firewall_guard` - a context manager wrapping a whole search stage.
3. :func:`search_mode` - a flag stored in the run context and checked by the
   data layer before labels are produced.

The firewall does NOT prevent verifying that test files and test dates EXIST.
It only prevents reading test LABELS or computing test METRICS.  That
distinction matters: the search needs to know the final data is present without
learning anything from it.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import date, datetime

import numpy as np
import pandas as pd

#: First date that may never be scored during search.  The paper's final test
#: years are 2022 (fold 0) and 2023 (fold 1).
TEST_FIREWALL_START = pd.Timestamp("2022-01-01")

#: Exclusive upper bound of the firewalled window, i.e. end of fold 1's test year.
TEST_FIREWALL_END = pd.Timestamp("2024-01-01")


class TestSetFirewallError(RuntimeError):
    """Raised when search code attempts to score a firewalled date."""

    # Tell pytest this is an exception, not a test class (its name starts with
    # "Test").
    __test__ = False


def _as_timestamp(value) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, (datetime, date)):
        return pd.Timestamp(value)
    return pd.Timestamp(str(value))


def date_in_firewalled_window(value) -> bool:
    """True when ``value`` is on/after the firewall start (or in 2022-2023)."""
    ts = _as_timestamp(value)
    return ts >= TEST_FIREWALL_START


def is_search_mode(flag: bool | None = None) -> bool:
    """Resolve the active search mode.

    Precedence: explicit ``flag`` > ``RECOVERY_SEARCH_MODE`` env var > False.
    """
    if flag is not None:
        return bool(flag)
    return os.environ.get("RECOVERY_SEARCH_MODE", "") == "1"


def assert_pre_test_dates(values: Iterable, *, where: str = "scoring",
                          search: bool | None = None) -> None:
    """Raise if any date in ``values`` falls in the firewalled window.

    This is the primitive every scoring path must call.  It is deliberately
    cheap and message-specific so a violation names the offending date.
    """
    if not is_search_mode(search):
        return
    for v in values:
        if date_in_firewalled_window(v):
            raise TestSetFirewallError(
                f"{where}: refusing to use date {pd.Timestamp(v).date()} - it is on or "
                f"after {TEST_FIREWALL_START.date()}. The 2022/2023 paper test years "
                "must remain unseen during search. Use a pre-2022 search fold "
                "(SEARCH_FOLD_A/B/C) instead."
            )


@contextmanager
def firewall_guard(search: bool | None = None) -> Iterator[None]:
    """Context manager that enforces the firewall for a block of work.

    >>> with firewall_guard(True):
    ...     assert_pre_test_dates(["2021-12-31"])      # fine
    ...     assert_pre_test_dates(["2022-01-03"])      # raises
    TestSetFirewallError: ...
    """
    previous = os.environ.get("RECOVERY_SEARCH_MODE")
    if is_search_mode(search):
        os.environ["RECOVERY_SEARCH_MODE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("RECOVERY_SEARCH_MODE", None)
        else:
            os.environ["RECOVERY_SEARCH_MODE"] = previous


def assert_labels_not_firewalled(labels: np.ndarray | pd.Series, *,
                                dates=None, where: str = "labels",
                                search: bool | None = None) -> None:
    """Guard a label array, using ``dates`` when available.

    Guards on DATES, not on label values: the paper test years are identified by
    their dates, so this is the correct axis to check.
    """
    if not is_search_mode(search):
        return
    if dates is None:
        # Without dates we cannot prove the labels are pre-2022.  Refuse rather
        # than assume: a silent pass would defeat the whole firewall.
        raise TestSetFirewallError(
            f"{where}: refusing to use {len(labels)} labels without dates. "
            "The firewall can only prove labels are pre-2022 if their dates are "
            "supplied."
        )
    assert_pre_test_dates(pd.to_datetime(pd.Series(dates)), where=where, search=search)


def assert_config_windows_safe(cfg: dict, *, where: str = "config",
                               search: bool | None = None) -> None:
    """Reject a config whose test/validation window touches 2022 or later.

    A search config must place its evaluation window strictly before
    :data:`TEST_FIREWALL_START`.  ``test_*`` keys are checked even though search
    folds do not use them, because a copy-pasted paper config would otherwise
    silently evaluate 2022.
    """
    if not is_search_mode(search):
        return
    for key in ("val_start", "val_end", "test_start", "test_end"):
        raw = cfg.get(key)
        if not raw:
            continue
        ts = _as_timestamp(raw)
        if ts >= TEST_FIREWALL_START:
            raise TestSetFirewallError(
                f"{where}: data.{key}={ts.date()} is on or after "
                f"{TEST_FIREWALL_START.date()}. Search folds must evaluate strictly "
                "before 2022."
            )


def describe_firewall() -> dict:
    """Machine-readable description of the firewall, for manifests."""
    return {
        "firewall_start_exclusive_of": str(TEST_FIREWALL_START.date()),
        "firewall_end_exclusive_of": str(TEST_FIREWALL_END.date()),
        "protected_test_years": [2022, 2023],
        "search_folds_allowed": ["SEARCH_FOLD_A (2016-2018/2019)",
                                "SEARCH_FOLD_B (2016-2019/2020)",
                                "SEARCH_FOLD_C (2016-2020/2021)"],
        "note": ("The firewall blocks scoring and label use for 2022/2023. It does "
                 "NOT block verifying that test files and dates exist."),
    }
