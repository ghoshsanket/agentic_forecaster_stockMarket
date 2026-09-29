"""Shared data validation for OHLCV frames.

Kept in the package (rather than duplicated in each build script) so the
downloader, the artifact builders and the runtime all apply the SAME rule.

OHLC consistency and floating point
-----------------------------------
The identity checks ``High >= max(Open, Close)`` and
``Low <= min(Open, Close)`` are exact mathematical facts, but on
``float64`` data that has been back-adjusted by a corporate-action factor the
values are only equal to within rounding.  A bar where the high *is* the
close can therefore end up with ``High`` a few ULP below ``Close``, which an
exact comparison reports as a violation.

Measured on the legacy adjusted dataset: 563 such exact-inequality breaches,
with a maximum magnitude of 5.7e-14 INR, versus zero in the unadjusted
variant.  These are rounding artefacts, not data errors, so the validator
compares with a tolerance and reports the worst observed breach so material
violations stay visible.

The tolerance is applied to the COMPARISON ONLY.  No price is ever altered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Absolute tolerance in price units.  Comfortably above the 5.7e-14 rounding
# observed, and far below any economically meaningful breach.
DEFAULT_ATOL = 1e-9
# Relative tolerance, for high-priced series where ULP scales with magnitude.
DEFAULT_RTOL = 1e-9

# (rule name, lhs, rhs, relation)
#   "ge": lhs must be >= rhs, so a VIOLATION is lhs < rhs
#   "le": lhs must be <= rhs, so a VIOLATION is lhs > rhs
OHLC_RULES = (
    ("high_below_low", "High", "Low", "ge"),
    ("high_below_open", "High", "Open", "ge"),
    ("high_below_close", "High", "Close", "ge"),
    ("low_above_open", "Low", "Open", "le"),
    ("low_above_close", "Low", "Close", "le"),
)


def ohlc_violations(df: pd.DataFrame, *, atol: float = DEFAULT_ATOL,
                    rtol: float = DEFAULT_RTOL) -> dict:
    """Return per-rule violation counts and the worst breach magnitude.

    For a rule ``(name, lhs, rhs, "ge")`` a violation is ``lhs < rhs``;
    for ``(name, lhs, rhs, "le")`` a violation is ``lhs > rhs``.

    A bar is counted as violating a rule only when it breaches the relation by
    MORE than the tolerance, so rounding noise is ignored while a genuine
    mis-ordered bar is still reported.  ``tolerance_cleared`` records how many
    breaches were discarded as rounding, and ``worst_breach`` keeps the largest
    observed magnitude so a near-miss is still auditable.
    """
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"frame is missing OHLC columns: {sorted(missing)}")

    out: dict = {
        "counts": {},
        "worst_breach": {},
        "tolerance_cleared": {},
        "atol": atol,
        "rtol": rtol,
    }
    valid = df.dropna(subset=list(required))
    for name, lhs, rhs, relation in OHLC_RULES:
        left = valid[lhs].to_numpy(dtype=float)
        right = valid[rhs].to_numpy(dtype=float)
        close_enough = np.isclose(left, right, rtol=rtol, atol=atol)
        breach = (left < right) if relation == "ge" else (left > right)
        hard = breach & ~close_enough
        out["counts"][name] = int(hard.sum())
        out["tolerance_cleared"][name] = int((breach & close_enough).sum())
        out["worst_breach"][name] = (
            float(np.abs(left - right)[breach].max()) if breach.any() else 0.0)
    out["total_material_violations"] = int(sum(out["counts"].values()))
    out["total_tolerance_cleared"] = int(sum(out["tolerance_cleared"].values()))
    return out


def frame_issues(df: pd.DataFrame, *, start: str | None = None,
                 end: str | None = None, atol: float = DEFAULT_ATOL,
                 rtol: float = DEFAULT_RTOL) -> list[str]:
    """Full quality gate for a canonical daily OHLCV frame.

    ``end`` is treated as EXCLUSIVE, matching the yfinance request semantics.
    """
    issues: list[str] = []
    if df.empty:
        return ["empty_frame"]
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"])
        if not dates.is_monotonic_increasing:
            issues.append("date_not_monotonic")
        if dates.duplicated().any():
            issues.append("duplicate_dates")
        if start is not None and dates.min() < pd.Timestamp(start):
            issues.append(f"date_before_start:{dates.min().date()}")
        if end is not None and dates.max() >= pd.Timestamp(end):
            issues.append(f"date_on_or_after_end:{dates.max().date()}")

    value_cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    if df[value_cols].isna().any().any():
        issues.append("nulls")
    if "Volume" in df.columns and (df["Volume"] < 0).any():
        issues.append("negative_volume")

    v = ohlc_violations(df, atol=atol, rtol=rtol)
    for name, count in v["counts"].items():
        if count:
            issues.append(f"{name}:{count}")
    return issues
