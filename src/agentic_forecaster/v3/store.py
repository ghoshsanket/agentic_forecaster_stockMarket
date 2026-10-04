"""The V3 exogenous store: causally aligned, physically PRE-COVID.

WHAT IS BUILT
--------------
One parquet table of exogenous features keyed by the STOCK TRADING DATE ``t``, plus
the provenance of every value in it::

    exogenous_features.parquet
        [date, *<source_id>_<feature>, stock_minus_<index>_return_*,
         <ticker>_minus_sector_return_*, sector_index_source_id,
         source_observation_<source_id>, lag_<source_id>]
    sector_mapping.parquet
        [ticker, sector, external_sector_index_id, mapping_confidence, provenance]
    metadata.json

THE POINT IS THE ALIGNMENT, NOT THE TABLE
------------------------------------------
For every stock date ``t`` and every accepted source the store records:

``source_observation_<id>``  the provider's own observation date actually used
``lag_<id>``                 how many source observations back that value sits

A Class B (conservative) source therefore shows ``lag >= 1``: the provider's
same-calendar-date close is never used, because a US close happens hours after the
NSE close.  A Class A Indian source may show ``lag = 0``.

PHYSICAL CAP
------------
The store is capped at ``2019-12-31``.  A provider snapshot that ignored the
exclusive download end is truncated before it is ever written, so no post-2019
exogenous bar can be loaded, let alone consumed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_forecaster.utils import atomic_json_dump, ensure_dir

from . import CONTROL_FAMILY, TRACK_ID, V3Track
from .availability import (
    align_source_asof,
    availability_report,
    build_global_risk_factor,
    sector_index_mapping,
)
from .features import (
    ExogenousFeatureSpec,
    assert_causal,
    build_source_features,
    drop_degenerate_columns,
    templates_for,
)
from .firewall import FINAL_ALLOWED_DATE, assert_no_post_2019
from .sources import (
    CLASS_B,
    FAMILY_E1,
    FAMILY_E2,
    FAMILY_E3,
    SourceRegistry,
    SourceSpec,
)

logger = logging.getLogger("agentic_forecaster.v3.store")

STORE_FILES: tuple[str, ...] = ("exogenous_features.parquet", "sector_mapping.parquet")

#: Which accepted families feed which feature family.
FAMILY_MEMBERSHIP: dict[str, tuple[str, ...]] = {
    "X1_INDIA_MARKET": (FAMILY_E1,),
    "X2_GLOBAL_RISK": (FAMILY_E2,),
    "X3_MACRO_COMMODITY": (FAMILY_E3,),
    "X4_ALL_EXOGENOUS": (FAMILY_E1, FAMILY_E2, FAMILY_E3),
}

#: The ``global_risk_factor`` components with FIXED signs (higher = more risk-off).
GLOBAL_RISK_COMPONENTS: dict[str, tuple[str, int]] = {
    "global_risk_spx_move": ("SP500_return_1", -1),
    "global_risk_nasdaq_move": ("NASDAQ_COMPOSITE_return_1", -1),
    "global_risk_vix_change": ("US_VIX_change_1", +1),
    "global_risk_usdinr_move": ("USDINR_return_1", -1),
    "global_risk_crude_move": ("BRENT_CRUDE_return_1", -1),
}

#: The index whose level defines the India-market context features.
INDIA_PRIMARY_INDEX = "NIFTY50"
INDIA_SECONDARY_INDEX = "NIFTY_BANK"


def store_root(root: Path | None = None) -> Path:
    return (Path(root) if root is not None else V3Track().store)


def store_fingerprints(root: Path | None = None) -> dict:
    base = store_root(root)
    metadata_path = base / "metadata.json"
    if not metadata_path.is_file():
        return {"root": str(base), "built": False}
    metadata = json.loads(metadata_path.read_text())
    return {"root": str(base), "built": True,
            "store_sha256": store_sha256(base),
            "config_sha256": metadata.get("config_sha256"),
            "source_manifest_sha256": metadata.get("source_manifest_sha256"),
            "last_date": metadata.get("last_date"),
            "n_sources": metadata.get("n_accepted_sources")}


def store_sha256(root: Path | None = None) -> str:
    base = store_root(root)
    digest = hashlib.sha256()
    for name in STORE_FILES:
        path = base / name
        digest.update(name.encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass
class ExogenousStore:
    """The aligned exogenous feature table plus its provenance."""

    root: Path
    features: pd.DataFrame
    sector_mapping: pd.DataFrame
    metadata: dict

    @property
    def feature_columns(self) -> tuple[str, ...]:
        reserved = {"date", "ticker"} | {
            c for c in self.features.columns if c.startswith(("source_observation_",
                                                               "lag_"))}
        return tuple(c for c in self.features.columns
                     if c not in reserved and c != "sector_index_source_id")

    @property
    def schema_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(sorted(self.feature_columns), sort_keys=False).encode()).hexdigest()

    def summary(self) -> dict:
        return {
            "root": str(self.root),
            "store_sha256": store_sha256(self.root),
            "n_rows": len(self.features),
            "n_exogenous_features": len(self.feature_columns),
            "first_date": str(self.features["date"].min().date()),
            "last_date": str(self.features["date"].max().date()),
            "n_accepted_sources": self.metadata.get("n_accepted_sources"),
            "accepted_sources": self.metadata.get("accepted_sources"),
            "rejected_sources": self.metadata.get("rejected_sources"),
            "feature_schema_sha256": self.schema_sha256,
        }


def _accepted(registry: SourceRegistry) -> list[SourceSpec]:
    return [spec for spec in registry.enabled_and_accepted()]


def build_exogenous_store(registry: SourceRegistry, *, stock_dates: pd.Series,
                          sector_of: dict[str, str], sector_index_symbols: dict[str, str],
                          raw_loader=None, root: Path | None = None,
                          final_allowed_date: str = FINAL_ALLOWED_DATE,
                          global_risk_scaler=None,
                          source_manifest_sha256: str | None = None) -> ExogenousStore:
    """Build the aligned exogenous store for every stock trading date.

    ``stock_dates`` are the STOCK origin dates -- the V3 prediction timestamps.
    Each accepted source is aligned onto them with merge-asof under its own
    availability class, so the alignment rule is per-source and auditable.
    """
    out_root = store_root(root)
    ensure_dir(out_root)
    raw_loader = raw_loader or _default_raw_loader
    origins = pd.Series(pd.to_datetime(pd.Series(stock_dates))).dropna().sort_values()
    origins = origins[~origins.duplicated()].reset_index(drop=True)
    origins = origins.loc[origins <= pd.Timestamp(final_allowed_date)].reset_index(drop=True)
    assert_no_post_2019(origin_dates=origins, where="exogenous store build",
                        final_allowed_date=final_allowed_date)

    frames: list[pd.DataFrame] = []
    specs: list[ExogenousFeatureSpec] = []
    alignment: dict[str, dict] = {}
    per_source_columns: dict[str, tuple[str, ...]] = {}

    for spec in _accepted(registry):
        source = raw_loader(spec)
        if source is None or source.empty:
            logger.warning("accepted source %s has no snapshot; excluded from the store",
                           spec.source_id)
            continue
        built = build_source_features(source, spec, final_allowed_date=final_allowed_date)
        assert_causal(built, source_observation_col="source_observation_date",
                      feature_date_col="source_date", where=f"features/{spec.source_id}")

        templates = templates_for(spec)
        available = [f"{spec.source_id}_{name}" for name in templates
                     if f"{spec.source_id}_{name}" in built.columns]
        if not available:
            continue
        aligned = align_source_asof(
            origins, built.loc[:, ["source_date", *available]],
            spec=spec, value_columns=tuple(available),
            final_allowed_date=final_allowed_date)
        # the report is taken BEFORE the provenance columns are renamed to their
        # per-source names, so it sees the generic availability contract
        alignment[spec.source_id] = availability_report(aligned, spec=spec)
        aligned = aligned.rename(columns={
            "source_observation_date": f"source_observation_{spec.source_id}",
            "lag_trading_observations": f"lag_{spec.source_id}"})
        frames.append(aligned.drop(columns=["effective_feature_date"]))
        specs.append(ExogenousFeatureSpec(source_id=spec.source_id,
                                          asset_class=spec.asset_class,
                                          availability_policy=spec.availability_policy,
                                          templates=tuple(
                                              c.split(f"{spec.source_id}_", 1)[1]
                                              for c in available)))
        per_source_columns[spec.source_id] = tuple(available)

    if not frames:
        raise ValueError("no accepted exogenous source produced any feature")

    features = pd.concat(frames, axis=1)
    features = (pd.DataFrame({"date": origins.to_numpy()})
                .merge(features, left_index=True, right_index=True, how="left"))
    features = features.sort_values("date").reset_index(drop=True)

    # --- the composite global-risk factor (TRAIN-fitted standardisation only) ----
    components = {name: spec_pair for name, spec_pair in GLOBAL_RISK_COMPONENTS.items()
                  if spec_pair[0] in features.columns}
    with_factor, factor_state = build_global_risk_factor(
        features, components=components, scaler=global_risk_scaler)
    if factor_state.get("available"):
        features["global_risk_factor"] = with_factor["global_risk_factor"]
    else:
        factor_state = {"available": False,
                        "reason": "insufficient accepted global components"}

    # --- sector mapping ------------------------------------------------------
    # Only ACCEPTED external indices may be mapped, so a sector whose index failed
    # its probe maps to nothing rather than to a source that has no data.  The
    # stock-relative and sector-relative features are per-TICKER, so they are built
    # in v3.dataset where the stock feature frame lives, not here.
    # The registry declares sector -> PROVIDER SYMBOL (for example "^CNXIT"), while
    # the store keys everything by source_id, so the symbols are translated first.
    # A symbol whose source was rejected by the probe maps to nothing at all: an
    # unavailable index must never appear as a mapped feature.
    accepted_by_symbol = {spec.provider_symbol_or_identifier: spec.source_id
                          for spec in _accepted(registry)}
    mapping = sector_index_mapping(
        sector_of,
        {sector: accepted_by_symbol[symbol]
         for sector, symbol in sector_index_symbols.items()
         if symbol in accepted_by_symbol},
    )

    # Degenerate FEATURES are dropped; the provenance columns are KEPT, because
    # they are the evidence the AVAILABILITY_AUDIT re-checks after loading.  A store
    # that could not prove where its values came from would be unauditable.
    provenance_columns = [c for c in features.columns
                          if c.startswith(("source_observation_", "lag_"))]
    numeric_columns = [c for c in features.columns
                       if c not in provenance_columns and c != "date"
                       and pd.api.types.is_numeric_dtype(features[c])]
    cleaned, dropped_columns = drop_degenerate_columns(
        features.loc[:, ["date", *numeric_columns]], where="v3 exogenous store")
    features = cleaned.merge(features.loc[:, ["date", *provenance_columns]],
                             on="date", how="left", validate="one_to_one")

    features.to_parquet(out_root / "exogenous_features.parquet", index=False)
    mapping.frame().to_parquet(out_root / "sector_mapping.parquet", index=False)

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "track": TRACK_ID,
        "final_allowed_date": final_allowed_date,
        "physical_cap": str(final_allowed_date),
        "n_rows": len(features),
        "first_date": str(features["date"].min().date()),
        "last_date": str(features["date"].max().date()),
        "accepted_sources": [spec.source_id for spec in _accepted(registry)],
        "n_accepted_sources": len(_accepted(registry)),
        "rejected_sources": {spec.source_id: spec.exclusion_reason
                             for spec in registry.rejected()},
        "feature_schemas": [spec.to_dict() for spec in specs],
        "dropped_degenerate_columns": dropped_columns,
        "alignment": alignment,
        "global_risk_factor": factor_state,
        "sector_mapping": {
            "n_mapped": len(mapping.mapped_tickers()),
            "n_unmapped": len(sector_of) - len(mapping.mapped_tickers()),
            "rule": ("a security without a verified external sector index keeps its "
                     "stock features and simply has no sector-index block; it is "
                     "never dropped from the study"),
        },
        "source_manifest_sha256": source_manifest_sha256,
        "config_sha256": hashlib.sha256(
            json.dumps({"families": FAMILY_MEMBERSHIP,
                        "global_risk_components": GLOBAL_RISK_COMPONENTS,
                        "india_primary": INDIA_PRIMARY_INDEX,
                        "india_secondary": INDIA_SECONDARY_INDEX},
                       sort_keys=True, default=str).encode()).hexdigest(),
        "prediction_timestamp": "after the NSE close on the stock trading date",
    }
    metadata["store_sha256"] = store_sha256(out_root)
    atomic_json_dump(metadata, out_root / "metadata.json")
    logger.info("V3 exogenous store written to %s (%d rows, %d features)", out_root,
                len(features), len(metadata["feature_schemas"]) * 12)
    return load_exogenous_store(out_root)


def _default_raw_loader(spec: SourceSpec) -> pd.DataFrame | None:
    from .download import load_accepted_source

    try:
        return load_accepted_source(spec)
    except FileNotFoundError:
        return None


def load_exogenous_store(root: Path | None = None, *,
                         final_allowed_date: str = FINAL_ALLOWED_DATE
                         ) -> ExogenousStore:
    """Load the store and re-prove the PRE-COVID boundary from its own rows."""
    base = store_root(root)
    metadata_path = base / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"No V3 exogenous store at {base}. Run "
            "scripts/build_v3_exogenous_store.py first.")
    metadata = json.loads(metadata_path.read_text())
    features = pd.read_parquet(base / "exogenous_features.parquet")
    features["date"] = pd.to_datetime(features["date"])
    mapping = pd.read_parquet(base / "sector_mapping.parquet")
    assert_no_post_2019(feature_dates=features["date"], where="v3 exogenous store load",
                        final_allowed_date=final_allowed_date)
    for column in features.columns:
        if column.startswith("source_observation_"):
            assert_no_post_2019(
                exogenous_dates=pd.to_datetime(features[column].dropna()),
                where=f"v3 exogenous store load/{column}",
                final_allowed_date=final_allowed_date)
    return ExogenousStore(root=base, features=features, sector_mapping=mapping,
                          metadata=metadata)


def sources_for_family(registry: SourceRegistry, family: str) -> list[SourceSpec]:
    """Accepted sources that belong to a V3 feature family."""
    members = FAMILY_MEMBERSHIP[family]
    return [spec for spec in _accepted(registry) if spec.family in members]


def family_feature_columns(store: ExogenousStore, family: str, *,
                           registry: SourceRegistry | None = None) -> tuple[str, ...]:
    """The exogenous columns a feature family contributes, in a stable order."""
    if family == CONTROL_FAMILY:
        return ()
    if registry is None:
        return tuple(sorted(store.feature_columns))
    # Prefix matching, not a split on "_": a source_id that itself contains an
    # underscore would otherwise contribute nothing at all, silently.
    prefixes = tuple(f"{spec.source_id}_" for spec in sources_for_family(registry, family))
    return tuple(column for column in store.feature_columns
                 if column.startswith(prefixes))


def availability_audit(store: ExogenousStore, *, sample_size: int = 100,
                       seed: int = 42) -> dict:
    """Sample stock origin dates and PROVE availability for every feature.

    This is the AVAILABILITY_AUDIT of section 47: for each sampled origin date and
    each accepted source, the source observation used must not be after the NSE
    prediction timestamp, and the lag distribution is reported so a Class B source
    can be seen to carry the conservative lag it claims.
    """
    features = store.features
    rng = np.random.default_rng(seed)
    dates = features["date"].to_numpy()
    size = min(sample_size, len(dates))
    sampled = np.sort(rng.choice(dates, size=size, replace=False))
    block = features.loc[features["date"].isin(sampled)]

    per_source: dict[str, dict] = {}
    violations: list[dict] = []
    for column in features.columns:
        if not column.startswith("source_observation_"):
            continue
        source_id = column[len("source_observation_"):]
        lag_column = f"lag_{source_id}"
        observed = pd.to_datetime(block[column])
        available = observed.notna()
        if int(available.sum()) == 0:
            per_source[source_id] = {"n_checked": 0,
                                     "reason": "no observation aligned to the sample"}
            continue
        # a source observation must never be after the origin date it is keyed on
        bad = block.loc[available & (observed > pd.to_datetime(block["date"]))]
        lags = pd.to_numeric(block.loc[available, lag_column], errors="coerce").dropna()
        policy = (store.metadata.get("alignment", {}).get(source_id, {})
                  .get("availability_class"))
        entry = {
            "availability_class": policy,
            "n_checked": int(available.sum()),
            "min_source_observation_date": str(observed.min().date()),
            "max_source_observation_date": str(observed.max().date()),
            "min_lag_trading_observations": int(lags.min()) if len(lags) else None,
            "max_lag_trading_observations": int(lags.max()) if len(lags) else None,
            "n_same_date": int((observed.loc[available]
                                == pd.to_datetime(block.loc[available, "date"])).sum()),
            "violations": len(bad),
        }
        if policy == CLASS_B and entry["n_same_date"]:
            entry["violations"] += int(entry["n_same_date"])
        per_source[source_id] = entry
        if entry["violations"]:
            violations.append({"source_id": source_id, **entry})

    return {
        "track": TRACK_ID,
        "sample_size": int(size),
        "seed": int(seed),
        "sampling": "deterministic uniform draw of stock origin dates",
        "prediction_timestamp": "after the NSE close on the stock trading date",
        "per_source": per_source,
        "n_violations": len(violations),
        "violations": violations,
        "passed": not violations,
        "rule": ("every exogenous value used at origin t must come from a source "
                 "observation no later than t, and a conservative-lag1 source must "
                 "never use a same-date observation"),
    }


def render_availability_audit(audit: dict) -> str:
    """``AVAILABILITY_AUDIT.md`` for a human reader."""
    lines: list[str] = []
    add = lines.append
    add("# V3 EXOGENOUS AVAILABILITY AUDIT")
    add("")
    add(f"- sampled stock origin dates: **{audit['sample_size']}** "
        f"(deterministic, seed {audit['seed']})")
    add(f"- prediction timestamp: {audit['prediction_timestamp']}")
    add(f"- causality violations: **{audit['n_violations']}**")
    add(f"- verdict: **{'PASS' if audit['passed'] else 'FAIL -- STOP'}**")
    add("")
    add("For each accepted source: the earliest and latest source observation used, "
        "the realised lag in source trading observations, and how many sampled "
        "origins used a same-date observation. A conservative-lag1 source must show "
        "zero same-date uses.")
    add("")
    add("| source | class | checked | min source obs | max source obs | min lag | "
        "max lag | same-date uses | violations |")
    add("|---|---|---|---|---|---|---|---|---|")
    for source_id, entry in sorted(audit["per_source"].items()):
        add(f"| `{source_id}` | {entry.get('availability_class') or '-'} | "
            f"{entry.get('n_checked', 0)} | "
            f"{entry.get('min_source_observation_date') or '-'} | "
            f"{entry.get('max_source_observation_date') or '-'} | "
            f"{entry.get('min_lag_trading_observations') if entry.get('min_lag_trading_observations') is not None else '-'} | "
            f"{entry.get('max_lag_trading_observations') if entry.get('max_lag_trading_observations') is not None else '-'} | "
            f"{entry.get('n_same_date', '-')} | {entry.get('violations', 0)} |")
    add("")
    add(f"Rule enforced: {audit['rule']}")
    add("")
    return "\n".join(lines) + "\n"