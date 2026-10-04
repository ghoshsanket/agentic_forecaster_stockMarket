"""The V3 external source registry: declaration, validation and resolution.

Every external series is DECLARED before it is used, with an explicit availability
class.  Nothing is hard-coded into the feature code, and no symbol is assumed to
work: a symbol that fails its probe is recorded as ``available: false`` with a
rejection reason and is EXCLUDED, never imputed.

The registry fields are exactly those required by the specification:

``source_id``, ``description``, ``provider``, ``provider_symbol_or_identifier``,
``asset_class``, ``market_timezone``, ``expected_session``,
``availability_policy``, ``frequency``, ``requested_start``, ``requested_end``,
``first_actual_date``, ``last_actual_date``, ``missing_fraction``,
``source_url_or_provenance``, ``raw_sha256``, ``enabled``, ``exclusion_reason``.

The first four fields after the identifier are the ones that determine causality,
so they are validated most strictly:

* ``availability_policy`` is the AVAILABILITY CLASS (see :mod:`.availability`).
* ``market_timezone`` and ``expected_session`` document WHEN the session closes,
  which is what a Class C promotion would have to prove.

AVAILABILITY CLASSES
--------------------
``INDIA_SAME_CLOSE``
    The official value is known at or by the NSE close on date ``t``, so the
    date-``t`` value may be used for an after-close prediction made at ``t``.
    Examples: NSE indices, India VIX.
``EXTERNAL_CONSERVATIVE_LAG1``
    The default for everything non-Indian.  The provider's same-calendar-date daily
    close is NEVER used; the most recent COMPLETE observation strictly before the
    NSE prediction timestamp is used instead.
``VERIFIED_BEFORE_NSE_CLOSE``
    A same-day foreign series may enter this class only when its session is
    DOCUMENTED to close before the NSE prediction timestamp.  Nikkei and Hang Seng
    are NOT placed here automatically.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .firewall import FINAL_ALLOWED_DATE

logger = logging.getLogger("agentic_forecaster.v3.sources")

#: The three availability classes.
CLASS_A = "INDIA_SAME_CLOSE"
CLASS_B = "EXTERNAL_CONSERVATIVE_LAG1"
CLASS_C = "VERIFIED_BEFORE_NSE_CLOSE"
AVAILABILITY_CLASSES: tuple[str, ...] = (CLASS_A, CLASS_B, CLASS_C)

#: Asset classes.
ASSET_INDEX = "EQUITY_INDEX"
ASSET_VOLATILITY = "VOLATILITY_INDEX"
ASSET_FX = "FX"
ASSET_COMMODITY = "COMMODITY"
ASSET_RATES = "RATES"
ASSET_FLOW = "FLOWS"

#: Information families a source can belong to.
FAMILY_E1 = "E1_INDIA_MARKET"
FAMILY_E2 = "E2_GLOBAL_RISK"
FAMILY_E3 = "E3_FX_COMMODITY_RATES"
FAMILY_E1_E2_E3 = "E1_E2_E3"
FAMILY_FII_DII = "FII_DII_FLOWS"
FAMILIES: tuple[str, ...] = (FAMILY_E1, FAMILY_E2, FAMILY_E3, FAMILY_E1_E2_E3,
                             FAMILY_FII_DII)

#: Every registry field, in the declared order.
REGISTRY_FIELDS: tuple[str, ...] = (
    "source_id",
    "description",
    "provider",
    "provider_symbol_or_identifier",
    "asset_class",
    "market_timezone",
    "expected_session",
    "availability_policy",
    "frequency",
    "requested_start",
    "requested_end",
    "first_actual_date",
    "last_actual_date",
    "missing_fraction",
    "source_url_or_provenance",
    "raw_sha256",
    "enabled",
    "exclusion_reason",
)

#: Extra bookkeeping fields the probe adds on top of the required schema.
PROBE_FIELDS: tuple[str, ...] = (
    "family", "available", "row_count", "duplicate_dates", "suspicious_jump_count",
    "final_lag_rule", "accepted", "probe_note",
)


@dataclass
class SourceSpec:
    """One declared external series."""

    source_id: str
    description: str
    provider: str
    provider_symbol_or_identifier: str
    asset_class: str
    market_timezone: str
    expected_session: str
    availability_policy: str
    frequency: str
    requested_start: str
    requested_end: str = FINAL_ALLOWED_DATE
    family: str = FAMILY_E1_E2_E3
    enabled: bool = True
    source_url_or_provenance: str = ""
    # filled by the probe
    first_actual_date: str | None = None
    last_actual_date: str | None = None
    missing_fraction: float | None = None
    raw_sha256: str | None = None
    exclusion_reason: str = ""
    available: bool | None = None
    row_count: int | None = None
    duplicate_dates: int | None = None
    suspicious_jump_count: int | None = None
    final_lag_rule: str = ""
    accepted: bool = False
    probe_note: str = ""

    def __post_init__(self) -> None:
        if self.availability_policy not in AVAILABILITY_CLASSES:
            raise ValueError(
                f"{self.source_id}: unknown availability_policy "
                f"{self.availability_policy!r}; expected one of {AVAILABILITY_CLASSES}")
        if self.family not in FAMILIES:
            raise ValueError(
                f"{self.source_id}: unknown family {self.family!r}; expected one of "
                f"{FAMILIES}")
        if pd.Timestamp(self.requested_end) > pd.Timestamp(FINAL_ALLOWED_DATE):
            raise ValueError(
                f"{self.source_id}: requested_end {self.requested_end} is past the "
                f"PRE-COVID boundary {FINAL_ALLOWED_DATE}. The V3 snapshot must stop "
                "at 2019-12-31.")

    def to_dict(self) -> dict:
        payload = asdict(self)
        return {key: payload[key] for key in REGISTRY_FIELDS if key in payload} | {
            key: payload[key] for key in PROBE_FIELDS if key in payload}

    def registry_row(self) -> dict:
        """The row exactly as written to the registry CSV, required fields only."""
        payload = self.to_dict()
        return {key: payload.get(key) for key in REGISTRY_FIELDS}

    def class_a(self) -> bool:
        return self.availability_policy == CLASS_A

    def class_c(self) -> bool:
        return self.availability_policy == CLASS_C

    def conservative(self) -> bool:
        return self.availability_policy == CLASS_B


@dataclass
class SourceRegistry:
    """Every declared source, keyed by ``source_id``."""

    sources: list[SourceSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        ids = [spec.source_id for spec in self.sources]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate source_id(s) in the registry: {sorted(duplicates)}")

    def __iter__(self):
        return iter(self.sources)

    def __len__(self) -> int:
        return len(self.sources)

    def get(self, source_id: str) -> SourceSpec:
        for spec in self.sources:
            if spec.source_id == source_id:
                return spec
        raise KeyError(f"no source {source_id!r} in the registry")

    def by_family(self, family: str) -> list[SourceSpec]:
        return [spec for spec in self.sources if spec.family == family]

    def accepted(self) -> list[SourceSpec]:
        """Sources that passed the coverage gate, in declaration order."""
        return [spec for spec in self.sources if spec.accepted]

    def enabled_and_accepted(self) -> list[SourceSpec]:
        return [spec for spec in self.accepted() if spec.enabled]

    def rejected(self) -> list[SourceSpec]:
        return [spec for spec in self.sources if not spec.accepted]

    def ids(self) -> list[str]:
        return [spec.source_id for spec in self.sources]

    def family_manifest(self, families: tuple[str, ...]) -> list[SourceSpec]:
        return [spec for spec in self.enabled_and_accepted() if spec.family in families]

    def to_dict(self) -> dict:
        return {"sources": [spec.to_dict() for spec in self.sources],
                "n_sources": len(self.sources),
                "n_accepted": len(self.accepted()),
                "availability_classes": list(AVAILABILITY_CLASSES)}


def registry_from_rows(rows: list[dict]) -> SourceRegistry:
    """Build a registry from plain dicts, ignoring unknown keys."""
    specs = []
    for row in rows:
        payload = {key: row.get(key) for key in REGISTRY_FIELDS + PROBE_FIELDS
                   if key in row}
        payload = {k: v for k, v in payload.items() if v is not None or k in
                   ("first_actual_date", "last_actual_date", "raw_sha256",
                    "exclusion_reason", "final_lag_rule", "probe_note")}
        if "enabled" in payload:
            payload["enabled"] = str(payload["enabled"]).strip().lower() in (
                "true", "1", "yes")
        specs.append(SourceSpec(**payload))
    return SourceRegistry(sources=specs)


def load_effective_registry(resolved_path, declared_path, audit_path=None
                            ) -> SourceRegistry:
    """Load the registry to BUILD WITH.

    Preference order:

    1. the resolved registry written by the probe (carries acceptance + hashes);
    2. the declared registry with the probe's verdicts merged back from the audit,
       so a store build never crashes merely because the resolved file is absent;
    3. the declared registry alone, in which case nothing is accepted yet and the
       caller must probe first.
    """
    import json

    resolved_path = Path(resolved_path)
    if resolved_path.is_file():
        return load_registry(resolved_path)

    registry = load_registry(declared_path)
    audit_path = Path(audit_path) if audit_path else None
    if audit_path is None or not audit_path.is_file():
        return registry
    audit = json.loads(audit_path.read_text())
    verdicts = {row["source_id"]: row for row in audit.get("sources", [])}
    for spec in registry:
        verdict = verdicts.get(spec.source_id)
        if verdict is None:
            continue
        spec.available = verdict.get("available")
        spec.row_count = verdict.get("row_count")
        spec.first_actual_date = verdict.get("first_actual_date")
        spec.last_actual_date = verdict.get("last_actual_date")
        spec.missing_fraction = verdict.get("missing_fraction")
        spec.duplicate_dates = verdict.get("duplicate_dates")
        spec.suspicious_jump_count = verdict.get("suspicious_jump_count")
        spec.raw_sha256 = verdict.get("raw_sha256")
        spec.accepted = bool(verdict.get("accepted"))
        spec.exclusion_reason = verdict.get("exclusion_reason") or ""
        spec.final_lag_rule = _lag_rule(spec.availability_policy)
    return registry


def _lag_rule(policy: str) -> str:
    if policy == CLASS_A:
        return "same-session close (INDIA_SAME_CLOSE)"
    if policy == CLASS_C:
        return "same-date allowed (VERIFIED_BEFORE_NSE_CLOSE)"
    return ("conservative lag1: most recent source observation strictly before the "
            "NSE prediction timestamp")


def load_registry(path) -> SourceRegistry:
    """Read ``source_registry.yaml``."""
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    rows = payload["sources"] if isinstance(payload, dict) else payload
    return registry_from_rows(list(rows))


def write_registry_csv(registry: SourceRegistry, path) -> pd.DataFrame:
    """Write the resolved registry (probe results included) to CSV."""
    frame = pd.DataFrame([spec.to_dict() for spec in registry])
    ordered = [key for key in REGISTRY_FIELDS + PROBE_FIELDS if key in frame.columns]
    frame = frame.loc[:, ordered]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def coverage_gate(spec: SourceSpec, dates: pd.Series, *, development_years=(2014, 2018),
                  max_missing_fraction: float = 0.05,
                  min_observations_per_year: int = 200) -> dict:
    """The source COVERAGE GATE.

    Required for the 2014-2018 development programme:

    * a plausible number of observations in every development year;
    * each year's density within ``max_missing_fraction`` of the source's OWN best
      development year;
    * no unexplained multi-month gap.

    The expected count is deliberately SELF-REFERENTIAL rather than a generic
    business-day count.  An Indian index trades roughly 246 sessions a year and a US
    index roughly 252, so comparing either against 261 weekdays would report a
    perfectly complete series as 6 % missing and reject it.  What actually matters
    for feature availability is whether the source covers its own calendar
    consistently across the development years, plus a floor that stops a sparse
    series from passing by being consistently sparse.

    A source that fails is EXCLUDED, never heavily imputed.
    """
    stamps = pd.to_datetime(pd.Series(dates)).dropna().sort_values().drop_duplicates()
    if stamps.empty:
        return {"passed": False, "reason": "no observations at all"}
    first, last = stamps.min(), stamps.max()
    problems: list[str] = []

    if last < pd.Timestamp(f"{development_years[1]}-12-31"):
        problems.append(f"coverage ends {last.date()}, before the development window")
    if first > pd.Timestamp("2008-01-01"):
        logger.info("%s: coverage starts %s (2008-01-01 preferred, not mandatory)",
                    spec.source_id, first.date())

    counts = {year: int((stamps.dt.year == year).sum())
              for year in range(development_years[0], development_years[1] + 1)}
    best = max(counts.values()) if counts else 0
    per_year: dict[str, float] = {}
    for year, count in counts.items():
        if count < min_observations_per_year:
            problems.append(f"{year} has only {count} observations "
                            f"(floor {min_observations_per_year})")
            per_year[str(year)] = 1.0
            continue
        fraction = 1.0 - (count / best) if best else 1.0
        per_year[str(year)] = float(fraction)
        if fraction > max_missing_fraction:
            problems.append(f"{year} density {count} is {fraction:.3f} below this "
                            f"source's best development year ({best})")

    gaps = stamps.diff().dt.days.dropna()
    long_gap = int((gaps > 45).sum()) if len(gaps) else 0
    if long_gap:
        problems.append(f"{long_gap} unexplained gap(s) longer than 45 calendar days")

    return {
        "passed": not problems,
        "reason": "; ".join(problems),
        "first_actual_date": str(first.date()),
        "last_actual_date": str(last.date()),
        "n_observations": len(stamps),
        "observations_per_development_year": {str(k): v for k, v in counts.items()},
        "best_development_year_count": int(best),
        "missing_fraction_by_year": per_year,
        "max_missing_fraction_in_development": (max(per_year.values()) if per_year
                                                else None),
        "long_gap_count": long_gap,
        "rule": (f"every development year {development_years[0]}-{development_years[1]} "
                 f"needs >= {min_observations_per_year} observations and within "
                 f"{max_missing_fraction} of this source's best year; no gap > 45 days"),
    }


def duplicate_date_count(dates: pd.Series) -> int:
    stamps = pd.to_datetime(pd.Series(dates))
    return int(len(stamps) - stamps.nunique())


def suspicious_jump_count(values: pd.Series, *, threshold: float = 0.35) -> int:
    """Count one-day moves beyond ``threshold`` in absolute log terms.

    A crude integrity probe: a daily index/FX/commodity move beyond 35 % usually
    means a bad print or a currency-unit change, not a real session.
    """
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(series) < 2:
        return 0
    returns = np.log(series).diff().abs()
    return int((returns > threshold).sum())
