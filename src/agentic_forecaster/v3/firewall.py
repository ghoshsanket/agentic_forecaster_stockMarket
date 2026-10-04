"""V3 firewalls: the PRE-COVID date boundary and the sealed 2019 lockbox.

1. ``PostCovidDataAccessError`` (reused from V2)
   ANY V3 feature, exogenous observation or target dated after ``2019-12-31``
   raises, in any role: feature construction, alignment, scaling, PCA, training,
   validation, early stopping, model selection, confidence selection or metrics.

2. ``V3LockboxError`` (new)
   2019 is the single V3 lockbox.  A development run may not read a 2019 target or
   score a 2019 origin.  Only ``scripts/run_v3_precovid_lockbox.py`` with
   ``V3_PRECOVID_LOCKBOX=1`` may, and only once.

3. ``ExogenousCausalityError`` (new)
   Raised when an exogenous feature value cannot be shown to have been available
   at the NSE prediction timestamp of the origin it is attached to.  This is the
   guard that makes "no silent same-day foreign-market data" structural rather
   than aspirational.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import pandas as pd

from agentic_forecaster.v2.firewall import PostCovidDataAccessError

#: Absolute final allowed date for every V3 date.
FINAL_ALLOWED_DATE = pd.Timestamp("2019-12-31")
FIRST_FORBIDDEN_DATE = pd.Timestamp("2020-01-01")

LOCKBOX_ENV = "V3_PRECOVID_LOCKBOX"
LOCKBOX_YEAR = 2019
LOCKBOX_YEAR_START = pd.Timestamp(f"{LOCKBOX_YEAR}-01-01")
LOCKBOX_YEAR_END = pd.Timestamp(f"{LOCKBOX_YEAR}-12-31")

#: The NSE regular session close, in Asia/Kolkata.  The V3 prediction timestamp is
#: AFTER this close on day t.
NSE_TIMEZONE = "Asia/Kolkata"
NSE_CLOSE_LOCAL = "15:30"


class V3LockboxError(RuntimeError):
    """A V3 development run attempted to read the 2019 lockbox."""

    __test__ = False


class ExogenousCausalityError(RuntimeError):
    """An exogenous feature was not available at the NSE prediction timestamp."""

    __test__ = False


def prediction_timestamp(origin_date) -> pd.Timestamp:
    """The V3 prediction timestamp: after the NSE close on ``origin_date``.

    Naive local time, because every V3 date column is a trading DATE rather than a
    provider timestamp.  Comparisons are made on dates, which is conservative: a
    same-date foreign close is rejected unless its class explicitly proves an
    earlier session close.
    """
    return pd.Timestamp(pd.Timestamp(origin_date).date())


def lockbox_unlocked(explicit: bool | None = None) -> bool:
    """Resolve whether the ONE 2019 lockbox run is authorised."""
    if explicit is not None:
        return bool(explicit)
    return os.environ.get(LOCKBOX_ENV, "") == "1"


def assert_no_post_2019(*, feature_dates: Iterable = (), origin_dates: Iterable = (),
                       exogenous_dates: Iterable = (), target_dates: Iterable = (),
                       where: str = "v3 exogenous",
                       final_allowed_date=FINAL_ALLOWED_DATE) -> dict:
    """Reject ANY 2020+ date before it can influence a V3 model.

    ``PostCovidDataAccessError`` is reused deliberately: one boundary, one error,
    one behaviour across every track.
    """
    boundary = pd.Timestamp(final_allowed_date)
    for label, values in (("feature", feature_dates), ("origin", origin_dates),
                          ("exogenous", exogenous_dates), ("target", target_dates)):
        for value in values:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            stamp = pd.Timestamp(value)
            if pd.isna(stamp):
                continue
            if stamp > boundary:
                raise PostCovidDataAccessError(
                    f"{where}: refusing {label} date {stamp.date()}. The V3 "
                    f"exogenous programme is PRE-COVID and ends at {boundary.date()}; "
                    "nothing from 2020 onward may be consumed by feature "
                    "construction, alignment, scaling, PCA, training, validation, "
                    "early stopping, model selection, confidence selection or "
                    "metrics.")
    return {"final_allowed_date": str(boundary.date()), "checked": True}


def assert_no_lockbox_year(values: Iterable, *, where: str = "v3 exogenous",
                           unlocked: bool | None = None) -> str | None:
    """Reject any 2019 target during development."""
    if lockbox_unlocked(unlocked):
        from agentic_forecaster.v2.firewall import max_target_date

        return max_target_date(values)
    worst = None
    for value in values:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            continue
        worst = stamp if worst is None or stamp > worst else worst
        if LOCKBOX_YEAR_START <= stamp <= LOCKBOX_YEAR_END:
            raise V3LockboxError(
                f"{where}: refusing date {stamp.date()}. {LOCKBOX_YEAR} is the V3 "
                f"exogenous LOCKBOX: it is read exactly once by "
                f"scripts/run_v3_precovid_lockbox.py with {LOCKBOX_ENV}=1, after the "
                "exogenous feature family, horizon and model are frozen. No "
                "information family or model may be selected using 2019.")
    return None if worst is None else worst.strftime("%Y-%m-%d")


def assert_availability(available_dates: Iterable, prediction_dates: Iterable, *,
                        where: str = "exogenous alignment") -> dict:
    """Every exogenous observation must PRECEDE its prediction timestamp.

    ``available_dates`` and ``prediction_dates`` are parallel sequences of the
    source observation date and the origin's NSE prediction date.  A violation is
    an error, never a warning.
    """
    checked = 0
    for available, predicted in zip(available_dates, prediction_dates, strict=True):
        if available is None or pd.isna(pd.Timestamp(available)):
            continue
        checked += 1
        if pd.Timestamp(available) > pd.Timestamp(predicted):
            raise ExogenousCausalityError(
                f"{where}: source observation {pd.Timestamp(available).date()} is "
                f"AFTER the NSE prediction timestamp "
                f"{pd.Timestamp(predicted).date()}. That value could not have been "
                "known when the prediction was made. Use a conservative lag, or "
                "reclassify the source only with documented proof that its session "
                "closes before the NSE close.")
    return {"n_checked": checked, "violations": 0}


def describe_firewall() -> dict:
    return {
        "track": TRACK_ID,
        "final_allowed_date": str(FINAL_ALLOWED_DATE.date()),
        "first_forbidden_date": str(FIRST_FORBIDDEN_DATE.date()),
        "post_covid_error": "PostCovidDataAccessError",
        "lockbox_env_var": LOCKBOX_ENV,
        "lockbox_year": LOCKBOX_YEAR,
        "lockbox_error": "V3LockboxError",
        "causality_error": "ExogenousCausalityError",
        "prediction_timestamp": "after the NSE close on day t (Asia/Kolkata 15:30)",
        "note": ("a foreign daily close that happens after the NSE close is NOT "
                 "available at t, even when the provider labels it with the same "
                 "calendar date; such sources default to the previous source trading "
                 "observation"),
    }


from agentic_forecaster.v3 import TRACK_ID