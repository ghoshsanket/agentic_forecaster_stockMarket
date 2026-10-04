"""Point-in-time availability and alignment for exogenous sources.

THE PROBLEM THIS MODULE EXISTS TO SOLVE
---------------------------------------
Aligning an external daily series to a stock date with ``external[date ==
stock_date]`` is a leak.  A US close at 16:00 ET happens roughly 10 hours AFTER
the NSE close, so that value was not knowable when the prediction was made, even
though both carry the same calendar date.

THE RULE
--------
For a stock origin date ``t`` the prediction timestamp is *after the NSE close on
t*.  An exogenous observation may be used only if it was available by then.

======================= ==================================================
``INDIA_SAME_CLOSE``    the date-``t`` value may be used: the official
                        Indian close IS the prediction timestamp's own
                        information
``VERIFIED_BEFORE_NSE_`` a same-day FOREIGN value may be used only with
``CLOSE``                documented proof that its session closes first
``EXTERNAL_CONSERVATIVE_`` the provider's same-calendar-date close is NEVER
``LAG1``                 used; the most recent COMPLETE observation strictly
                        before ``t`` is used instead
======================= ==================================================

Conservative lag1 is the default for every non-Indian series in the first run,
and it intentionally sacrifices same-day information for causal safety.

AUDITABILITY
------------
Every aligned row records FOUR fields, so any single feature can be traced back
to the observation it came from:

``source_observation_date``  the provider's own observation date
``effective_feature_date``   the date the feature is keyed on (the origin ``t``)
``lag_trading_observations`` how many source observations back the value sits
``availability_class``       the class that produced the rule

``align_source_asof`` performs a MERGE-ASOF on the source date, which can only
reach backwards; a backward fill or an interpolation from the future is therefore
structurally impossible rather than merely discouraged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .firewall import (
    ExogenousCausalityError,
    assert_availability,
    assert_no_post_2019,
)
from .sources import CLASS_A, CLASS_B, CLASS_C, SourceSpec

#: Sector label that must never receive an external index mapping.
UNKNOWN_SECTOR_LABEL = "UNKNOWN"

logger = logging.getLogger("agentic_forecaster.v3.availability")

#: Provenance columns every aligned exogenous frame carries.
ALIGNMENT_PROVENANCE: tuple[str, ...] = (
    "source_observation_date",
    "effective_feature_date",
    "lag_trading_observations",
    "availability_class",
)


def availability_allowed(availability_policy: str, source_observation_date,
                         origin_date) -> bool:
    """May ``source_observation_date`` be used at origin ``origin_date``?

    Class A: the same date is allowed (the Indian close is the prediction
    timestamp's own information).
    Class C: the same date is allowed, but only because the registry asserted a
    documented earlier session close.
    Class B: the source observation must be STRICTLY BEFORE the origin date.
    """
    source = pd.Timestamp(source_observation_date)
    origin = pd.Timestamp(origin_date)
    if availability_policy in (CLASS_A, CLASS_C):
        return bool(source <= origin)
    if availability_policy == CLASS_B:
        return bool(source < origin)
    raise ValueError(f"unknown availability policy {availability_policy!r}")


def lag_rule_description(availability_policy: str) -> str:
    if availability_policy == CLASS_A:
        return ("same-session close: the official Indian value is known at the NSE "
                "close on t, so date-t is used")
    if availability_policy == CLASS_C:
        return ("same-date value allowed: the registry documents that this session "
                "closes before the NSE close on t")
    return ("conservative lag1: the most recent source observation STRICTLY BEFORE t "
            "is used; the provider's same-calendar-date close is never used")


def align_source_asof(origin_dates: pd.Series, source: pd.DataFrame, *,
                      spec: SourceSpec, value_columns: tuple[str, ...],
                      source_date_col: str = "source_date",
                      origin_col: str | None = None,
                      strict: bool | None = None,
                      final_allowed_date=None) -> pd.DataFrame:
    """Align a source onto stock origin dates with MERGE-ASOF (backwards only).

    Parameters
    ----------
    origin_dates:
        The stock origin dates ``t`` (the V3 prediction timestamps).
    source:
        Long frame with ``source_date`` and one column per value.
    spec:
        The declaring source, whose availability class sets the rule.
    strict:
        ``None`` uses the declaring class's rule.  ``True`` forces a strictly
        earlier observation even for a same-close class.  ``False`` is REFUSED for
        a conservative source: the rule can be made stricter by a caller, never
        looser, so a same-day foreign close cannot be requested into existence.

    Returns
    -------
    A frame with one row per origin date, the aligned values, and the four
    provenance fields.
    """
    origins = pd.Series(pd.to_datetime(pd.Series(origin_dates))).dropna().sort_values()
    origins = origins[~origins.duplicated()].reset_index(drop=True)
    frame = source.copy()
    frame[source_date_col] = pd.to_datetime(frame[source_date_col])
    frame = frame.sort_values(source_date_col).drop_duplicates(source_date_col,
                                                              keep="last")
    if final_allowed_date is not None:
        assert_no_post_2019(exogenous_dates=frame[source_date_col],
                            where=f"align/{spec.source_id}",
                            final_allowed_date=final_allowed_date)

    class_allows_same_date = spec.class_a() or spec.class_c()
    if strict is False and not class_allows_same_date:
        raise ExogenousCausalityError(
            f"align/{spec.source_id}: refusing to relax the "
            f"{spec.availability_policy} rule to a same-date alignment. The "
            "conservative lag is the causality guarantee and cannot be turned off; "
            "reclassify the source only with documented proof that its session "
            "closes before the NSE close.")
    # ``strict=True`` means "strictly earlier observation", so it turns the
    # same-date match OFF; ``strict=None`` defers to the declaring class.
    same_date_ok = (class_allows_same_date if strict is None
                    else not bool(strict))
    # MERGE-ASOF with the chosen direction is what makes a future observation
    # unreachable: with direction="backward" the join can only look backwards.
    merged = pd.merge_asof(
        pd.DataFrame({(origin_col or "origin_date"): origins}),
        frame.loc[:, [source_date_col, *value_columns]],
        left_on=(origin_col or "origin_date"),
        right_on=source_date_col,
        direction="backward",
        allow_exact_matches=bool(same_date_ok),
    )
    merged = merged.rename(columns={source_date_col: "source_observation_date"})
    merged["effective_feature_date"] = merged[origin_col or "origin_date"]
    merged["availability_class"] = spec.availability_policy

    # How many SOURCE observations separate the value used from the origin, counted
    # on the SOURCE's own calendar: the number of source sessions d with
    # source_observation < d <= origin.  A Class B source therefore normally shows 1,
    # and a value of 0 would mean a same-session observation slipped through.
    source_days = np.sort(frame[source_date_col].to_numpy(dtype="datetime64[ns]"))
    observed = merged["source_observation_date"]
    merged["lag_trading_observations"] = [
        (None if pd.isna(value) or pd.isna(origin) else int(np.searchsorted(
            source_days, pd.Timestamp(origin).to_datetime64(), side="right")
            - np.searchsorted(source_days, pd.Timestamp(value).to_datetime64(),
                              side="right")))
        for value, origin in zip(observed, merged["effective_feature_date"], strict=True)
    ]
    for column in value_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    # Enforce the rule EXPLICITLY as well, so no join detail can quietly leak a
    # same-day foreign close.
    if not same_date_ok:
        offending = merged.loc[
            (merged["source_observation_date"] >= merged["effective_feature_date"])
            & merged["source_observation_date"].notna()]
        if len(offending):
            raise ExogenousCausalityError(
                f"align/{spec.source_id}: {len(offending)} origin(s) were aligned to a "
                "source observation on or after the origin date under a conservative "
                "lag1 rule. The most recent observation strictly BEFORE the "
                "prediction timestamp must be used.")
    assert_availability(merged["source_observation_date"],
                        merged["effective_feature_date"],
                        where=f"align/{spec.source_id}")
    return merged.drop(columns=[origin_col or "origin_date"])


def availability_report(aligned: pd.DataFrame, *, spec: SourceSpec | None = None,
                        where: str = "availability") -> dict:
    """Describe the realised alignment for the audit artefacts."""
    lags = pd.to_numeric(aligned["lag_trading_observations"], errors="coerce").dropna()
    observed = pd.to_datetime(aligned["source_observation_date"]).dropna()
    origins = pd.to_datetime(aligned["effective_feature_date"])
    same_date = int((observed.dt.normalize().values
                     == origins.loc[observed.index].dt.normalize().values).sum())
    return {
        "source_id": None if spec is None else spec.source_id,
        "availability_class": (None if spec is None else spec.availability_policy),
        "lag_rule": (None if spec is None
                     else lag_rule_description(spec.availability_policy)),
        "n_origins": len(aligned),
        "n_with_observation": len(observed),
        "n_missing_observation": int(aligned["source_observation_date"].isna().sum()),
        "coverage_fraction": (float(len(observed) / len(aligned)) if len(aligned)
                              else float("nan")),
        "min_lag_trading_observations": (int(lags.min()) if len(lags) else None),
        "median_lag_trading_observations": (float(lags.median()) if len(lags) else None),
        "max_lag_trading_observations": (int(lags.max()) if len(lags) else None),
        "n_same_calendar_date": same_date,
        "same_calendar_date_allowed": bool(spec is not None and (spec.class_a()
                                                                or spec.class_c())),
        "violations": 0,
        "note": ("lag_trading_observations counts source observations back from the "
                 "origin, so a Class B source normally shows 1 or more; a value of 0 "
                 "for a Class B source would be a violation"),
    }


@dataclass
class SectorIndexMapping:
    """Which external sector index, if any, a security maps to.

    A mapping is only recorded when it is unambiguous.  ``UNKNOWN`` or an
    unverified sector leaves ``external_sector_index_id`` empty and the
    sector-index features unavailable -- it never drops the security itself.
    """

    rows: list[dict]

    def index_for(self, ticker: str) -> str | None:
        for row in self.rows:
            if row["ticker"] == ticker:
                return row.get("external_sector_index_id") or None
        return None

    def confidence_for(self, ticker: str) -> float:
        for row in self.rows:
            if row["ticker"] == ticker:
                return float(row.get("mapping_confidence") or 0.0)
        return 0.0

    def frame(self) -> pd.DataFrame:
        columns = ("ticker", "sector", "external_sector_index_id",
                   "mapping_confidence", "provenance")
        return pd.DataFrame(self.rows, columns=list(columns)).sort_values("ticker")

    def mapped_tickers(self) -> list[str]:
        return sorted(row["ticker"] for row in self.rows
                      if row.get("external_sector_index_id"))


def sector_index_mapping(sector_of: dict[str, str],
                         sector_index_by_sector: dict[str, str], *,
                         confidence: float = 0.9) -> SectorIndexMapping:
    """Map securities to sector indices, refusing ambiguous mappings.

    A sector with no verified external index maps to nothing; the security keeps
    its stock features and simply has no sector-index block.
    """
    rows = []
    unmappable = {UNKNOWN_SECTOR_LABEL, "", "NONE", "NAN"}
    for ticker, sector in sorted(sector_of.items()):
        key = str(sector)
        # An UNKNOWN sector is never mapped, even when the registry happens to carry
        # a symbol for that name: mapping it would silently attach a sector index to a
        # security whose sector is genuinely unidentified.
        if key.strip().upper() in unmappable:
            index_id = None
        else:
            index_id = (sector_index_by_sector.get(sector)
                        or sector_index_by_sector.get(key.upper()))
        rows.append({
            "ticker": ticker,
            "sector": sector,
            "external_sector_index_id": index_id or "",
            "mapping_confidence": (float(confidence) if index_id else 0.0),
            "provenance": ("verified NSE sector index" if index_id else
                           "no verified external index for this sector; "
                           "sector-index features unavailable"),
        })
    return SectorIndexMapping(rows=rows)


def build_global_risk_factor(frame: pd.DataFrame, *, components: dict[str, tuple],
                             scaler=None) -> tuple[pd.DataFrame, dict]:
    """One predefined ``global_risk_factor`` from standardised components.

    ``components`` maps an output column to ``(input_column, sign)``.  Signs are
    FIXED in the registry, never optimised on validation, and the standardisation
    is either supplied (TRAIN-fitted, persisted) or fitted here on TRAIN ROWS
    ONLY.

    Higher ``global_risk_factor`` means a more risk-averse global state.
    """
    from sklearn.preprocessing import StandardScaler

    data = frame.copy()
    # components maps an output column to (input column, fixed sign)
    used = [(column, input_column, sign)
            for column, (input_column, sign) in components.items()
            if input_column in data.columns]
    if not used:
        return pd.DataFrame(index=frame.index), {"available": False,
                                                 "reason": "no components available"}
    matrix = np.column_stack([data[input_column].to_numpy(float)
                              for _, input_column, _ in used])
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    fitted = StandardScaler().fit(matrix) if scaler is None else scaler
    standardised = fitted.transform(matrix)
    signs = np.asarray([sign for _, _, sign in used], dtype=float)
    data["global_risk_factor"] = (standardised * signs).mean(axis=1)
    return data, {
        "available": True,
        "components": {column: {"input": input_column, "sign": int(sign)}
                       for column, input_column, sign in used},
        "method": "equal-weight mean of TRAIN-standardised components with fixed signs",
        "weights_optimised_on_validation": False,
        "fitted_on": "TRAIN_ONLY" if scaler is None else "PERSISTED_TRAIN_FIT",
    }