"""V2 sector / industry identity: provenance, mapping and vocabulary.

V2 is a SHARED multi-stock model, so it needs a sector identity per security.
That identity comes from the OFFICIAL NSE constituent file used during the
universe recovery (``ind_nifty50list.csv``), which carries an ``Industry``
column.  This is STATIC reference metadata, not a market time series: nothing
here downloads prices, and no additional time-series data source is used for
V2.0.

Provenance rules implemented here
--------------------------------
* A previously downloaded official copy is reused when one exists.
* Otherwise the single official CSV is fetched once and stored, with its
  SHA-256 and retrieval timestamp, under
  ``$AGENTIC_PROCESSED_DATA_ROOT/v2/metadata/``.
* A ticker that cannot be mapped is assigned ``UNKNOWN`` and REPORTED.  It is
  never guessed from a sector index, a price series or a name heuristic.

``broad_sector`` is a deterministic, documented collapse of the NSE ``Industry``
labels into a smaller set of macro groups so that leave-one-out sector
statistics have enough peers to be defined.  The collapse is declared in
:data:`BROAD_SECTOR_BY_INDUSTRY` and is applied verbatim; an industry that is
absent from that table becomes ``UNKNOWN`` and is reported.
"""

from __future__ import annotations

import csv
import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from agentic_forecaster.utils import atomic_json_dump

logger = logging.getLogger("agentic_forecaster.v2.sectors")

#: The official NSE index-constituent file used during universe recovery.
NSE_CONSTITUENT_URL = (
    "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
)

#: Filename of the provenance copy stored under the V2 metadata directory.
NSE_CONSTITUENT_FILENAME = "ind_nifty50list.csv"

#: Columns required from the official file.  The Industry column is the reason
#: this file is fetched at all.
NSE_REQUIRED_COLUMNS = ("Symbol", "Industry")

#: Assigned when a ticker has no official industry record.
UNKNOWN_LABEL = "UNKNOWN"

#: Explicit, deterministic collapse of NSE industry labels into macro sectors.
#: Applied verbatim; anything missing collapses to UNKNOWN.
BROAD_SECTOR_BY_INDUSTRY: dict[str, str] = {
    "Financial Services": "FINANCIALS",
    "Information Technology": "IT_AND_TELECOM",
    "Telecommunication": "IT_AND_TELECOM",
    "Healthcare": "HEALTHCARE",
    "Automobile and Auto Components": "INDUSTRIALS",
    "Capital Goods": "INDUSTRIALS",
    "Construction Materials": "INDUSTRIALS",
    "Metals & Mining": "ENERGY_AND_MATERIALS",
    "Power": "ENERGY_AND_MATERIALS",
    "Oil Gas & Consumable Fuels": "ENERGY_AND_MATERIALS",
    "Fast Moving Consumer Goods": "CONSUMER",
    "Consumer Durables": "CONSUMER",
    "Consumer Services": "CONSUMER",
    "Services": "SERVICES_AND_CONSTRUCTION",
    "Construction": "SERVICES_AND_CONSTRUCTION",
}

#: Ordered output columns of the sector map.
SECTOR_MAP_COLUMNS = (
    "ticker",
    "company_name",
    "industry",
    "broad_sector",
    "source",
    "source_sha256",
    "retrieved_at",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class SectorMap:
    """Static sector identity for every V2 security."""

    frame: pd.DataFrame
    source: str
    source_sha256: str
    retrieved_at: str
    unmapped_tickers: list[str] = field(default_factory=list)
    unmapped_industries: list[str] = field(default_factory=list)
    extra_symbols: list[str] = field(default_factory=list)

    @property
    def tickers(self) -> list[str]:
        return list(self.frame["ticker"])

    def sector_for(self, ticker: str) -> str:
        row = self.frame.loc[self.frame["ticker"] == ticker]
        if row.empty:
            return UNKNOWN_LABEL
        return str(row.iloc[0]["broad_sector"])

    def industry_for(self, ticker: str) -> str:
        row = self.frame.loc[self.frame["ticker"] == ticker]
        if row.empty:
            return UNKNOWN_LABEL
        return str(row.iloc[0]["industry"])

    def sector_series(self) -> dict[str, str]:
        return dict(zip(self.frame["ticker"], self.frame["broad_sector"], strict=True))

    def coverage(self) -> dict:
        n = len(self.frame)
        mapped = n - len(self.unmapped_tickers)
        return {
            "n_tickers": n,
            "n_mapped": mapped,
            "n_unmapped": len(self.unmapped_tickers),
            "unmapped_tickers": list(self.unmapped_tickers),
            "unmapped_industries": list(self.unmapped_industries),
            "coverage_fraction": (mapped / n) if n else 0.0,
            "extra_symbols_in_source": list(self.extra_symbols),
            "source": self.source,
            "source_sha256": self.source_sha256,
            "retrieved_at": self.retrieved_at,
        }

    def sha256(self) -> str:
        """Content hash of the map itself (order-stable)."""
        payload = self.frame.loc[:, list(SECTOR_MAP_COLUMNS)].to_csv(index=False)
        return sha256_text(payload)

    def to_csv_bytes(self) -> bytes:
        return self.frame.loc[:, list(SECTOR_MAP_COLUMNS)].to_csv(index=False).encode()


def load_official_nse_csv(metadata_dir: str | Path, *, allow_download: bool = True,
                          timeout: float = 30.0) -> tuple[Path, str]:
    """Return ``(path, sha256)`` for the official NSE constituent CSV.

    A previously downloaded copy is reused.  Otherwise the single official file
    is fetched (when ``allow_download``) and stored with its provenance so the
    V2 sector map is reproducible without network access later.
    """
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    target = metadata_dir / NSE_CONSTITUENT_FILENAME
    if target.is_file() and target.stat().st_size > 0:
        return target, sha256_file(target)
    if not allow_download:
        raise FileNotFoundError(
            f"Official NSE constituent CSV not found at {target} and downloading is "
            "disabled. Provide a previously downloaded official copy there."
        )
    logger.info("Fetching official NSE constituent CSV: %s", NSE_CONSTITUENT_URL)
    import urllib.request

    request = urllib.request.Request(
        NSE_CONSTITUENT_URL,
        headers={"User-Agent": "agentic-forecaster-v2/1.0 (research)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    target.write_bytes(payload)
    return target, sha256_text(payload.decode("utf-8", errors="replace"))


def build_sector_map(tickers: Iterable[str], csv_path: str | Path, *,
                     source_sha256: str | None = None,
                     retrieved_at: str | None = None,
                     source_url: str = NSE_CONSTITUENT_URL) -> SectorMap:
    """Map every requested ticker to its OFFICIAL industry and broad sector.

    Unmapped tickers are recorded as ``UNKNOWN`` and returned in
    ``unmapped_tickers`` so the caller can report them.  Nothing is guessed.
    """
    csv_path = Path(csv_path)
    raw = pd.read_csv(csv_path)
    missing = [c for c in NSE_REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"{csv_path}: official constituent file is missing column(s) {missing}. "
            f"Expected {list(NSE_REQUIRED_COLUMNS)}."
        )
    raw["Symbol"] = raw["Symbol"].astype(str).str.strip().str.upper()
    raw["Industry"] = raw["Industry"].astype(str).str.strip()
    company_col = "Company Name" if "Company Name" in raw.columns else None

    digest = source_sha256 or sha256_file(csv_path)
    # The retrieval stamp is the moment the official copy was STORED locally (the
    # file's mtime), not the moment this process happened to run.  Using "now"
    # would change the sector-map hash on every rebuild and invalidate the V2
    # feature-store cache for no reason.
    stamp = retrieved_at or datetime.fromtimestamp(
        csv_path.stat().st_mtime, tz=UTC).isoformat()

    rows: list[dict] = []
    unmapped: list[str] = []
    unmapped_industries: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        ticker = str(ticker).upper()
        seen.add(ticker)
        match = raw.loc[raw["Symbol"] == ticker]
        if match.empty:
            rows.append({
                "ticker": ticker,
                "company_name": UNKNOWN_LABEL,
                "industry": UNKNOWN_LABEL,
                "broad_sector": UNKNOWN_LABEL,
                "source": f"{source_url} (symbol absent from file)",
                "source_sha256": digest,
                "retrieved_at": stamp,
            })
            unmapped.append(ticker)
            continue
        industry = str(match.iloc[0]["Industry"]).strip() or UNKNOWN_LABEL
        broad = BROAD_SECTOR_BY_INDUSTRY.get(industry, UNKNOWN_LABEL)
        if broad == UNKNOWN_LABEL and industry != UNKNOWN_LABEL:
            unmapped_industries.append(industry)
        company = (str(match.iloc[0][company_col]).strip()
                   if company_col else UNKNOWN_LABEL)
        rows.append({
            "ticker": ticker,
            "company_name": company or UNKNOWN_LABEL,
            "industry": industry,
            "broad_sector": broad,
            "source": source_url,
            "source_sha256": digest,
            "retrieved_at": stamp,
        })

    extra = sorted(set(raw["Symbol"]) - seen)
    frame = pd.DataFrame(rows, columns=list(SECTOR_MAP_COLUMNS))
    sector_map = SectorMap(
        frame=frame,
        source=source_url,
        source_sha256=digest,
        retrieved_at=stamp,
        unmapped_tickers=sorted(unmapped),
        unmapped_industries=sorted(set(unmapped_industries)),
        extra_symbols=extra,
    )
    if unmapped:
        logger.warning(
            "%d/%d tickers have no official industry record and are marked UNKNOWN: %s",
            len(unmapped), len(frame), ", ".join(unmapped),
        )
    return sector_map


def write_sector_map_csv(sector_map: SectorMap, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(sector_map.to_csv_bytes())
    return path


def write_sector_map_yaml(sector_map: SectorMap, path: str | Path, *,
                          universe_id: str, csv_path: str | Path) -> Path:
    """Write the human-readable sector-map declaration consumed by the runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sector_map_id": f"V2_SECTOR_MAP_{sector_map.source_sha256[:12]}",
        "universe_id": universe_id,
        "status": "OFFICIAL_NSE_INDUSTRY_FIELD",
        "source": sector_map.source,
        "source_sha256": sector_map.source_sha256,
        "retrieved_at": sector_map.retrieved_at,
        "provenance_csv": str(csv_path),
        "unknown_label": UNKNOWN_LABEL,
        "n_tickers": len(sector_map.frame),
        "n_unmapped": len(sector_map.unmapped_tickers),
        "unmapped_tickers": sector_map.unmapped_tickers,
        "extra_symbols_in_source": sector_map.extra_symbols,
        "broad_sector_definition": (
            "Deterministic collapse of the official NSE Industry labels listed in "
            "v2/sectors.py:BROAD_SECTOR_BY_INDUSTRY. Not a vendor classification and "
            "not inferred from any market data."
        ),
        "survivorship_note": (
            "The universe is a fixed 2025-11-04 candidate snapshot, so sector counts "
            "are a TODAY's composition applied backwards. Sector membership is static "
            "reference metadata, which is acceptable for a fixed-universe "
            "reconstruction, but the resulting sector aggregates inherit the same "
            "survivorship bias as the universe itself."
        ),
        "entries": [
            {
                "ticker": r["ticker"],
                "company_name": r["company_name"],
                "industry": r["industry"],
                "broad_sector": r["broad_sector"],
            }
            for r in sector_map.frame.to_dict("records")
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, width=100), encoding="utf-8")
    return path


def load_sector_map(path: str | Path) -> SectorMap:
    """Read a sector-map CSV written by :func:`write_sector_map_csv`."""
    # dtype=str keeps the file byte-faithful: a SHA-256 of zeros would otherwise
    # come back as an integer and change the map's content hash.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in SECTOR_MAP_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: sector map is missing column(s) {missing}")
    frame = frame.loc[:, list(SECTOR_MAP_COLUMNS)]
    unmapped = frame.loc[frame["industry"] == UNKNOWN_LABEL, "ticker"].tolist()
    unmapped_industries = sorted({
        str(i) for i in frame.loc[frame["broad_sector"] == UNKNOWN_LABEL, "industry"]
        if str(i) != UNKNOWN_LABEL
    })
    return SectorMap(
        frame=frame,
        source=str(frame["source"].iloc[0]),
        source_sha256=str(frame["source_sha256"].iloc[0]),
        retrieved_at=str(frame["retrieved_at"].iloc[0]),
        unmapped_tickers=[str(t) for t in unmapped],
        unmapped_industries=unmapped_industries,
    )


def build_vocabulary(values: Iterable[str], *, unknown_label: str = UNKNOWN_LABEL
                     ) -> list[str]:
    """Vocabulary whose index 0 is ALWAYS the UNKNOWN slot.

    The remaining entries are sorted, so the ordering is deterministic, and
    index 0 is reserved so an unseen sector (or a missing mapping) has a valid
    embedding index instead of failing at inference time.
    """
    unique = {str(v) for v in values}
    unique.discard(unknown_label)
    return [unknown_label, *sorted(unique)]


def vocabulary_index(vocab: list[str], value: str, *, unknown_label: str = UNKNOWN_LABEL
                    ) -> int:
    """Index of ``value`` in ``vocab``; the UNKNOWN slot when absent."""
    if value in vocab:
        return vocab.index(value)
    if unknown_label in vocab:
        return vocab.index(unknown_label)
    return 0


def read_universe_tickers(config_path: str | Path) -> tuple[list[str], str]:
    """Return ``(tickers, universe_id)`` from a universe YAML declaration."""
    with Path(config_path).open() as fh:
        raw = yaml.safe_load(fh) or {}
    tickers = [str(t).upper() for t in raw.get("tickers", [])]
    if not tickers:
        raise ValueError(f"{config_path}: no tickers declared")
    universe_id = str(raw.get("universe_id", "UNKNOWN_UNIVERSE"))
    return tickers, universe_id


def read_sheet_name_map(dataset_root: str | Path) -> dict[str, str]:
    """``{ticker -> parquet stem}`` from the downloader's authoritative map.

    The paper-snapshot dataset writes ``M&M`` as ``M_AND_M.parquet``; the
    downloader's ``metadata/sheet_name_map.csv`` is the authoritative record of
    that mapping, so it is preferred over re-implementing the sanitisation rule.
    """
    path = Path(dataset_root) / "metadata" / "sheet_name_map.csv"
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = (row.get("legacy_label") or "").strip().upper()
            stem = (row.get("canonical_filename_stem") or "").strip()
            parquet = (row.get("parquet_filename") or "").strip()
            if label and parquet:
                out[label] = Path(parquet).stem or stem
    return out


def write_sector_map_metadata(sector_map: SectorMap, path: str | Path, *,
                              extra: dict | None = None) -> Path:
    payload = {"coverage": sector_map.coverage(), "sector_map_sha256": sector_map.sha256()}
    payload.update(extra or {})
    atomic_json_dump(payload, path)
    return Path(path)