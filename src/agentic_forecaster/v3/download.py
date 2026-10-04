"""Probing and downloading external sources, with immutable raw snapshots.

Nothing is assumed to work.  :func:`probe_source` asks the provider whether a
symbol exists and returns what it actually found -- first date, last date, row
count, duplicate dates, suspicious jumps, and the SHA-256 of the exact bytes that
were stored.  A symbol that fails is recorded as unavailable and EXCLUDED; it is
never imputed, proxied or silently replaced.

RAW STORAGE
-----------
Snapshots live under ``$AGENTIC_DATA_ROOT/exogenous/pre_covid/raw/<family>/`` and
are NEVER modified after download.  Re-running with an existing snapshot reuses
it, so a rebuild cannot quietly change the inputs behind an existing result.

PROVENANCE
----------
Every download records the provider, the symbol, the yfinance version, the
download parameters and the file hash.  The download window is capped at
``2020-01-01`` EXCLUSIVE so no post-2019 bar is even requested where the provider
honours it; any row that still arrives is dropped before it can be stored.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .firewall import FINAL_ALLOWED_DATE, assert_no_post_2019
from .sources import (
    SourceRegistry,
    SourceSpec,
    coverage_gate,
    duplicate_date_count,
    suspicious_jump_count,
)

logger = logging.getLogger("agentic_forecaster.v3.download")

#: Columns a provider snapshot is normalised to.
RAW_COLUMNS: tuple[str, ...] = ("source_date", "open", "high", "low", "close", "volume")

#: Provider default download end, EXCLUSIVE: no post-2019 bar is requested.
DOWNLOAD_END_EXCLUSIVE = "2020-01-01"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_path(spec: SourceSpec, root: Path) -> Path:
    """Immutable snapshot path for one source."""
    return Path(root) / spec.family / f"{spec.source_id}.parquet"


def normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise a provider frame to the RAW_COLUMNS schema, dates ascending."""
    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=list(RAW_COLUMNS))
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        # yfinance returns (field, ticker) columns for a single-symbol request; the
        # TICKER level carries no information here and dropping it is what keeps the
        # OHLC values instead of turning every column name into a tuple
        out.columns = [str(level[0]) for level in out.columns]
    out.columns = [str(c).strip().lower() for c in out.columns]
    if "date" in out.columns:
        out = out.rename(columns={"date": "source_date"})
    if "source_date" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={out.index.name or "index": "source_date"})
    if "source_date" not in out.columns:
        raise ValueError(f"provider frame has no date column: {list(out.columns)}")
    stamps = pd.to_datetime(out["source_date"], errors="coerce")
    if isinstance(stamps.dtype, pd.DatetimeTZDtype):
        # a provider timezone label is dropped, NOT converted: the V3 boundary is a
        # trading DATE, and shifting a timestamp across zones could move a bar onto
        # the wrong day
        stamps = stamps.dt.tz_localize(None)
    out["source_date"] = stamps
    for column in RAW_COLUMNS[1:]:
        if column not in out.columns:
            out[column] = pd.NA
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.loc[:, list(RAW_COLUMNS)].sort_values("source_date").reset_index(drop=True)
    return out


@dataclass
class ProbeResult:
    """What a provider actually returned for one declared symbol."""

    spec: SourceSpec
    available: bool
    row_count: int
    first_actual_date: str | None
    last_actual_date: str | None
    missing_fraction: float | None
    duplicate_dates: int
    suspicious_jump_count: int
    raw_sha256: str | None
    raw_path: str | None
    note: str
    provenance: dict

    def to_dict(self) -> dict:
        return {
            "source_id": self.spec.source_id,
            "provider": self.spec.provider,
            "identifier": self.spec.provider_symbol_or_identifier,
            "family": self.spec.family,
            "available": self.available,
            "row_count": self.row_count,
            "first_actual_date": self.first_actual_date,
            "last_actual_date": self.last_actual_date,
            "missing_fraction": self.missing_fraction,
            "duplicate_dates": self.duplicate_dates,
            "suspicious_jump_count": self.suspicious_jump_count,
            "raw_sha256": self.raw_sha256,
            "raw_path": self.raw_path,
            "note": self.note,
            "provenance": self.provenance,
        }


def _provider_version(provider: str) -> str | None:
    try:
        if provider.lower() in ("yfinance", "yahoo", "yahoo finance"):
            import yfinance

            return str(yfinance.__version__)
    except Exception:  # pragma: no cover - defensive
        return None
    return None


def fetch_provider_frame(spec: SourceSpec) -> tuple[pd.DataFrame | None, dict]:
    """Fetch one symbol from its provider.  Returns ``(frame, provenance)``."""
    provenance: dict = {
        "provider": spec.provider,
        "identifier": spec.provider_symbol_or_identifier,
        "requested_start": spec.requested_start,
        "requested_end_exclusive": DOWNLOAD_END_EXCLUSIVE,
        "provider_version": _provider_version(spec.provider),
        "download_parameters": {"auto_adjust": False, "progress": False,
                                "threads": False, "actions": False,
                                "multi_level_index": False},
    }
    if spec.provider.lower() not in ("yfinance", "yahoo", "yahoo finance"):
        provenance["note"] = (f"provider {spec.provider!r} has no retrieval adapter in "
                              "this programme; the source is probed as unavailable")
        return None, provenance
    try:
        import yfinance as yf

        frame = yf.download(spec.provider_symbol_or_identifier,
                            start=spec.requested_start, end=DOWNLOAD_END_EXCLUSIVE,
                            progress=False, auto_adjust=False, threads=False,
                            actions=False, multi_level_index=False)
    except Exception as error:  # pragma: no cover - network dependent
        provenance["error"] = f"{type(error).__name__}: {error}"
        return None, provenance
    if frame is None or len(frame) == 0:
        provenance["error"] = "provider returned no rows"
        return None, provenance
    provenance["note"] = "downloaded"
    return frame, provenance


def probe_source(spec: SourceSpec, *, raw_root: Path, store: bool = True,
                 max_missing_fraction: float = 0.05) -> ProbeResult:
    """Probe one declared source and, when it passes, store an immutable snapshot."""
    target = raw_path(spec, raw_root)
    if target.is_file():
        frame = pd.read_parquet(target)
        provenance = {"note": "reused existing immutable snapshot",
                      "raw_path": str(target)}
        digest = sha256_file(target)
    else:
        raw, provenance = fetch_provider_frame(spec)
        if raw is None:
            return ProbeResult(spec=spec, available=False, row_count=0,
                               first_actual_date=None, last_actual_date=None,
                               missing_fraction=None, duplicate_dates=0,
                               suspicious_jump_count=0, raw_sha256=None,
                               raw_path=None,
                               note=provenance.get("error", "unavailable"),
                               provenance=provenance)
        frame = normalise_frame(raw)
        # A provider that ignores the exclusive end must still never STORE a
        # post-2019 bar.
        assert_no_post_2019(exogenous_dates=frame["source_date"],
                            where=f"download/{spec.source_id}",
                            final_allowed_date=FINAL_ALLOWED_DATE)
        frame = frame.loc[frame["source_date"] <= pd.Timestamp(FINAL_ALLOWED_DATE)]
        if not store:
            return ProbeResult(spec=spec, available=True, row_count=len(frame),
                               first_actual_date=str(frame["source_date"].min().date()),
                               last_actual_date=str(frame["source_date"].max().date()),
                               missing_fraction=None, duplicate_dates=0,
                               suspicious_jump_count=0, raw_sha256=None, raw_path=None,
                               note=provenance.get("note", ""), provenance=provenance)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
        digest = sha256_file(target)
        provenance["raw_path"] = str(target)
        provenance["rows_written"] = len(frame)

    dates = frame["source_date"]
    close = frame["close"]
    gate = coverage_gate(spec, dates, max_missing_fraction=max_missing_fraction)
    duplicates = duplicate_date_count(dates)
    jumps = suspicious_jump_count(close)
    available = bool(len(frame)) and close.notna().sum() > 0
    note = gate["reason"] if not gate["passed"] else provenance.get("note", "ok")
    if available and duplicates:
        note = f"{note}; {duplicates} duplicate date(s) collapsed"
    if available and jumps:
        note = f"{note}; {jumps} suspicious one-day jump(s) -- inspect before use"
    return ProbeResult(
        spec=spec, available=available and gate["passed"], row_count=len(frame),
        first_actual_date=gate.get("first_actual_date"),
        last_actual_date=gate.get("last_actual_date"),
        missing_fraction=gate.get("max_missing_fraction_in_development"),
        duplicate_dates=duplicates, suspicious_jump_count=jumps,
        raw_sha256=digest, raw_path=str(target) if target.is_file() else None,
        note=note.strip("; "), provenance=provenance | {"coverage_gate": gate})


def probe_registry(registry: SourceRegistry, *, raw_root: Path, store: bool = True,
                   max_missing_fraction: float = 0.05) -> list[ProbeResult]:
    """Probe every declared source, in declaration order.

    A rejected source is reported, never hidden and never silently replaced.
    """
    results: list[ProbeResult] = []
    for spec in registry:
        logger.info("probing %s (%s %s)", spec.source_id, spec.provider,
                    spec.provider_symbol_or_identifier)
        try:
            result = probe_source(spec, raw_root=raw_root, store=store,
                                  max_missing_fraction=max_missing_fraction)
        except Exception as error:  # pragma: no cover - network dependent
            result = ProbeResult(spec=spec, available=False, row_count=0,
                                 first_actual_date=None, last_actual_date=None,
                                 missing_fraction=None, duplicate_dates=0,
                                 suspicious_jump_count=0, raw_sha256=None,
                                 raw_path=None,
                                 note=f"probe raised {type(error).__name__}: {error}",
                                 provenance={})
        spec.available = result.available
        spec.row_count = result.row_count
        spec.first_actual_date = result.first_actual_date
        spec.last_actual_date = result.last_actual_date
        spec.missing_fraction = result.missing_fraction
        spec.duplicate_dates = result.duplicate_dates
        spec.suspicious_jump_count = result.suspicious_jump_count
        spec.raw_sha256 = result.raw_sha256
        spec.accepted = bool(result.available)
        spec.exclusion_reason = "" if spec.accepted else result.note
        spec.final_lag_rule = _lag_rule_text(spec)
        spec.probe_note = result.note
        results.append(result)
    return results


def download_accepted_sources(registry: SourceRegistry, *, raw_root: Path,
                              store: bool = True,
                              max_missing_fraction: float = 0.05
                              ) -> list[ProbeResult]:
    """PROBE every source without storing, then DOWNLOAD only the accepted ones.

    Keeping the two passes separate matters: a snapshot is only written for a series
    that passed the coverage gate, so a rejected symbol never leaves a raw file
    behind that a later step might pick up.
    """
    verdicts = probe_registry(registry, raw_root=raw_root, store=False,
                              max_missing_fraction=max_missing_fraction)
    accepted_ids = {result.spec.source_id for result in verdicts if result.available}
    logger.info("probe accepted %d of %d declared sources", len(accepted_ids),
                len(verdicts))

    results: list[ProbeResult] = []
    for spec in registry:
        if spec.source_id not in accepted_ids:
            verdict = next(r for r in verdicts if r.spec.source_id == spec.source_id)
            results.append(verdict)
            continue
        result = probe_source(spec, raw_root=raw_root, store=store,
                              max_missing_fraction=max_missing_fraction)
        spec.available = result.available
        spec.row_count = result.row_count
        spec.first_actual_date = result.first_actual_date
        spec.last_actual_date = result.last_actual_date
        spec.missing_fraction = result.missing_fraction
        spec.duplicate_dates = result.duplicate_dates
        spec.suspicious_jump_count = result.suspicious_jump_count
        spec.raw_sha256 = result.raw_sha256
        spec.accepted = bool(result.available)
        spec.exclusion_reason = "" if spec.accepted else result.note
        spec.final_lag_rule = _lag_rule_text(spec)
        spec.probe_note = result.note
        results.append(result)
    return results


def _lag_rule_text(spec: SourceSpec) -> str:
    if spec.class_a():
        return "same-session close (INDIA_SAME_CLOSE)"
    if spec.class_c():
        return "same-date allowed (VERIFIED_BEFORE_NSE_CLOSE)"
    return ("conservative lag1: most recent source observation strictly before the "
            "NSE prediction timestamp")


def load_accepted_source(spec: SourceSpec) -> pd.DataFrame:
    """Read one accepted snapshot from its immutable raw path."""
    from . import raw_root as _raw_root

    candidates = sorted(_raw_root(spec.family).glob(f"{spec.source_id}.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"no raw snapshot for accepted source {spec.source_id}; run "
            "scripts/download_v3_exogenous.py first")
    return normalise_frame(pd.read_parquet(candidates[0]))


def write_manifest(registry: SourceRegistry, *, manifests_root: Path,
                   provenance_rows: list[dict] | None = None) -> dict:
    """Write ``source_manifest.csv`` / ``.json`` with a SHA-256 per accepted source."""

    from agentic_forecaster.utils import atomic_json_dump

    root = Path(manifests_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in registry.sources:
        rows.append({
            "source_id": spec.source_id,
            "family": spec.family,
            "provider": spec.provider,
            "identifier": spec.provider_symbol_or_identifier,
            "asset_class": spec.asset_class,
            "market_timezone": spec.market_timezone,
            "expected_session": spec.expected_session,
            "availability_policy": spec.availability_policy,
            "final_lag_rule": spec.final_lag_rule,
            "frequency": spec.frequency,
            "first_actual_date": spec.first_actual_date,
            "last_actual_date": spec.last_actual_date,
            "row_count": spec.row_count,
            "missing_fraction": spec.missing_fraction,
            "duplicate_dates": spec.duplicate_dates,
            "suspicious_jump_count": spec.suspicious_jump_count,
            "raw_sha256": spec.raw_sha256,
            "accepted": spec.accepted,
            "exclusion_reason": spec.exclusion_reason,
            "source_url_or_provenance": spec.source_url_or_provenance,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "source_manifest.csv", index=False)
    manifest = {
        "track": "V3_EXOGENOUS_PRECOVID",
        "final_allowed_date": FINAL_ALLOWED_DATE,
        "download_end_exclusive": DOWNLOAD_END_EXCLUSIVE,
        "n_declared": len(registry),
        "n_accepted": int(sum(1 for spec in registry if spec.accepted)),
        "sources": rows,
        "manifest_sha256": hashlib.sha256(
            frame.to_csv(index=False).encode()).hexdigest(),
        "note": ("raw snapshots are immutable; a rebuild reuses them rather than "
                 "re-downloading, so an existing result can never be re-based"),
    }
    atomic_json_dump(manifest, root / "source_manifest.json")
    if provenance_rows:
        atomic_json_dump({"downloads": provenance_rows}, root / "download_provenance.json")
    return manifest


def write_provenance_document(*, manifests_root: Path, registry: SourceRegistry) -> Path:
    """Write ``SOURCE_PROVENANCE.md`` next to the manifests."""
    path = Path(manifests_root) / "SOURCE_PROVENANCE.md"
    lines = [
        "# V3 EXOGENOUS SOURCE PROVENANCE",
        "",
        ("Every external series used by the PRE-COVID exogenous programme, with the "
         "provider, the exact identifier, the download parameters and the SHA-256 "
         "of the stored bytes. Raw snapshots are never modified after download."),
        "",
        f"- download window end (exclusive): `{DOWNLOAD_END_EXCLUSIVE}`",
        f"- absolute final allowed date: `{FINAL_ALLOWED_DATE}`",
        "",
        ("| source_id | provider | identifier | class | lag rule | first | last | "
         "rows | SHA256 | accepted |"),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for spec in registry.sources:
        lines.append(
            f"| `{spec.source_id}` | {spec.provider} | "
            f"`{spec.provider_symbol_or_identifier}` | {spec.availability_policy} | "
            f"{spec.final_lag_rule} | {spec.first_actual_date or '-'} | "
            f"{spec.last_actual_date or '-'} | {spec.row_count if spec.row_count is not None else '-'} | "
            f"`{(spec.raw_sha256 or '-')[:16]}...` | "
            f"{'yes' if spec.accepted else 'NO'} |")
    rejected = registry.rejected()
    if rejected:
        lines += ["", "## Rejected sources", "",
                  "| source_id | identifier | reason |", "|---|---|---|"]
        for spec in rejected:
            lines.append(f"| `{spec.source_id}` | "
                         f"`{spec.provider_symbol_or_identifier}` | "
                         f"{spec.exclusion_reason or 'unavailable'} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path