"""The V2 feature store: one reproducible, cached context store.

Layout (under ``$AGENTIC_PROCESSED_DATA_ROOT/v2/context_store/``)::

    stock_features.parquet
        [ticker, date, *STOCK_FEATURE_NAMES]
    market_context.parquet
        [date, *MARKET_FEATURES]                       date-level market proxy
    sector_context.parquet
        [ticker, date, industry, broad_sector, *SECTOR_FEATURES]
    cross_sectional_features.parquet
        [ticker, date, *MARKET_LOO_FEATURES, *RELATIVE_FEATURES,
         *RANK_FEATURES, *REGIME_FEATURES]            leave-one-out + ranks + regime
    target_frame.parquet
        [ticker, origin_date, target_date, realized_vol_20, raw_next_return,
         y_direction, y_return, y_rank]
    metadata.json

Cache invalidation depends on FINGERPRINTS, not on wall-clock time:

* ``source_manifest_sha256`` -- the dataset's own file manifest;
* ``feature_code_sha256``    -- hash of the V2 feature/context source files;
* ``config_sha256``          -- hash of the resolved build configuration;
* ``sector_map_sha256``      -- hash of the sector map used.

If any fingerprint changes, :func:`build_store` rebuilds; otherwise the cached
store is reused and verified.

HARD DATE CAP
-------------
The store is built with ``max_date`` (default ``2021-12-31``) and no row beyond
it is written.  That makes "no 2022/2023 target label exists in V2" a
structural property rather than a promise, which is exactly what the paper-test
firewall is supposed to guarantee.  Source data is only ever READ.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from agentic_forecaster.utils import atomic_json_dump, ensure_dir

from . import context as ctx
from . import features as feat
from .firewall import V2_PAPER_TEST_FIREWALL_START, assert_no_paper_test_targets
from .sectors import SectorMap

logger = logging.getLogger("agentic_forecaster.v2.store")

STORE_DIRNAME = "context_store"

STORE_FILES: tuple[str, ...] = (
    "stock_features.parquet",
    "market_context.parquet",
    "sector_context.parquet",
    "cross_sectional_features.parquet",
    "target_frame.parquet",
)

#: Fixed, documented clipping bound for the volatility-normalised return target.
RETURN_TARGET_CLIP = 5.0

#: Source files whose content defines the feature build; hashed for invalidation.
FEATURE_SOURCE_FILES: tuple[str, ...] = (
    "src/agentic_forecaster/v2/features.py",
    "src/agentic_forecaster/v2/context.py",
    "src/agentic_forecaster/v2/store.py",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def feature_code_sha256() -> str:
    """Hash of the V2 feature-building source files."""
    h = hashlib.sha256()
    root = repo_root()
    for rel in FEATURE_SOURCE_FILES:
        path = root / rel
        h.update(rel.encode())
        if path.is_file():
            h.update(path.read_bytes())
    return h.hexdigest()


def _hash_config(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _store_fingerprint_payload(*, dataset_root: Path, variant: str, universe_id: str,
                               tickers: list[str], sector_map: SectorMap, max_date: str,
                               manifest_hash: str, config_extra: dict | None) -> dict:
    """Every input the store content depends on.  Cache validity == this hash."""
    return {
        "dataset_root": str(dataset_root),
        "variant": variant,
        "universe_id": universe_id,
        "tickers": list(tickers),
        "max_date": str(max_date),
        "return_target_clip": RETURN_TARGET_CLIP,
        "stock_features": list(feat.STOCK_FEATURE_NAMES),
        "context_features": list(ctx.CONTEXT_FEATURES),
        "rank_features": list(ctx.RANK_FEATURES),
        "regime_features": list(ctx.REGIME_FEATURES),
        "sector_map_sha256": sector_map.sha256(),
        "feature_code_sha256": feature_code_sha256(),
        "source_manifest_sha256": manifest_hash,
        "extra": config_extra or {},
    }


def v2_processed_root() -> Path:
    """``$AGENTIC_PROCESSED_DATA_ROOT/v2``."""
    from agentic_forecaster.config import get_env_roots
    return Path(get_env_roots()["AGENTIC_PROCESSED_DATA_ROOT"]) / "v2"


def v2_metadata_dir() -> Path:
    return ensure_dir(v2_processed_root() / "metadata")


def store_root(root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else v2_processed_root()
    return base / STORE_DIRNAME


def _source_manifest_hash(dataset_root: Path) -> tuple[str, dict]:
    """Hash the dataset's own manifest when present, else hash the bar files."""
    manifest = dataset_root / "manifests" / "manifest_sha256.csv"
    if manifest.is_file():
        payload = manifest.read_bytes()
        return hashlib.sha256(payload).hexdigest(), {
            "manifest_path": str(manifest),
            "manifest_kind": "dataset_manifest_sha256.csv",
            "n_manifest_entries": int(payload.count(b"\n") - 1),
        }
    h = hashlib.sha256()
    n = 0
    for parquet in sorted((dataset_root / "adjusted" / "parquet").glob("*.parquet")):
        h.update(parquet.name.encode())
        h.update(parquet.read_bytes())
        n += 1
    return h.hexdigest(), {
        "manifest_path": None,
        "manifest_kind": "recomputed_from_adjusted_parquet",
        "n_manifest_entries": n,
    }


@dataclass
class FeatureStore:
    """In-memory handle over the cached parquet store plus its metadata."""

    root: Path
    stock: pd.DataFrame
    market: pd.DataFrame
    sector: pd.DataFrame
    cross_sectional: pd.DataFrame
    targets: pd.DataFrame
    metadata: dict = field(default_factory=dict)

    @property
    def store_sha256(self) -> str:
        """Content hash of the store: SHA-256 over every parquet file."""
        h = hashlib.sha256()
        for name in STORE_FILES:
            path = self.root / name
            h.update(name.encode())
            h.update(path.read_bytes())
        return h.hexdigest()

    @property
    def universe_id(self) -> str:
        return str(self.metadata.get("universe_id", ""))

    @property
    def data_variant(self) -> str:
        return str(self.metadata.get("data_variant", ""))

    @property
    def sector_map_sha256(self) -> str:
        return str(self.metadata.get("sector_map_sha256", ""))

    @property
    def max_target_date(self) -> str | None:
        return self.metadata.get("max_target_date")

    def summary(self) -> dict:
        return {
            "root": str(self.root),
            "store_sha256": self.store_sha256,
            "universe_id": self.universe_id,
            "data_variant": self.data_variant,
            "n_rows_stock": len(self.stock),
            "n_rows_market": len(self.market),
            "n_rows_sector": len(self.sector),
            "n_rows_cross_sectional": len(self.cross_sectional),
            "n_rows_targets": len(self.targets),
            "n_tickers": int(self.stock["ticker"].nunique()),
            "first_date": str(pd.Timestamp(self.stock["date"].min()).date()),
            "last_date": str(pd.Timestamp(self.stock["date"].max()).date()),
            "first_target_date": str(pd.Timestamp(self.targets["target_date"].min()).date()),
            "last_target_date": str(pd.Timestamp(self.targets["target_date"].max()).date()),
            "max_target_date": self.max_target_date,
            "sector_map_sha256": self.sector_map_sha256,
        }


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def _read_bars(dataset_root: Path, variant: str, parquet_dir: Path,
               sheet_map: dict[str, str], ticker: str) -> pd.DataFrame | None:
    stem = sheet_map.get(ticker, ticker)
    path = parquet_dir / f"{stem}.parquet"
    if not path.is_file():
        logger.warning("no %s parquet for %s at %s", variant, ticker, path)
        return None
    return pd.read_parquet(path)


def build_stock_feature_frame(dataset_root: Path, variant: str, tickers: list[str],
                              sheet_map: dict[str, str], *, max_date: str
                              ) -> tuple[pd.DataFrame, dict]:
    """Concatenate per-security causal stock features over the whole universe."""
    parquet_dir = Path(dataset_root) / variant / "parquet"
    frames: list[pd.DataFrame] = []
    quality: dict[str, dict] = {}
    for ticker in tickers:
        bars = _read_bars(dataset_root, variant, parquet_dir, sheet_map, ticker)
        if bars is None:
            continue
        if variant == "unadjusted":
            # The unadjusted variant is kept available as a sensitivity check.
            # OHLC are strictly positive for this universe, but the guard makes
            # the requirement explicit rather than incidental.
            for column in ("open", "high", "low"):
                bars = bars.loc[pd.to_numeric(bars[column], errors="coerce") > 0]
        built = feat.build_stock_features(bars)
        built = built.loc[built["date"] <= pd.Timestamp(max_date)]
        if built.empty:
            continue
        built.insert(0, "ticker", ticker)
        quality[ticker] = feat.feature_quality(built)
        frames.append(built)
    if not frames:
        raise ValueError(
            f"no security produced features under {dataset_root}/{variant}/parquet"
        )
    stock = pd.concat(frames, ignore_index=True)
    stock = stock.sort_values(["ticker", "date"]).reset_index(drop=True)
    return stock, quality


def build_target_frame(dataset_root: Path, variant: str, tickers: list[str],
                       sheet_map: dict[str, str], stock: pd.DataFrame, *,
                       max_date: str) -> pd.DataFrame:
    """Build the three supervision targets for every stock and origin date.

    TARGET 1 ``y_direction = int(close[t+1] > close[t])``
    TARGET 2 ``y_return = clip(log(close[t+1]/close[t]) / realized_vol_20[t], +/-5)``
    TARGET 3 ``y_rank`` = percentile rank of the next-day return among the stocks
            available on the TARGET date.  It is a LABEL built from ``t+1``
            information and is never an input.

    ``target_date`` is the security's next ACTUAL trading date, so a suspension
    or a weekend never fabricates a target one calendar day later.
    """
    parquet_dir = Path(dataset_root) / variant / "parquet"
    closes: dict[str, pd.Series] = {}
    for ticker in sorted(stock["ticker"].unique()):
        bars = _read_bars(dataset_root, variant, parquet_dir, sheet_map, ticker)
        if bars is None:
            continue
        normalised = feat.normalise_ohlcv_frame(bars)
        closes[ticker] = normalised.set_index("date")["close"]

    close_frame = pd.DataFrame(closes).sort_index()
    close_frame.index.name = "date"

    stock_idx = stock.set_index(["ticker", "date"])
    vol = stock_idx["realized_vol_20"]

    long_rows: list[pd.DataFrame] = []
    for ticker, series in close_frame.items():
        next_close = series.shift(-1)
        next_date = pd.Series(series.index, index=series.index).shift(-1)
        frame = pd.DataFrame({
            "ticker": ticker,
            "origin_date": series.index,
            "close_t": series.to_numpy(),
            "close_t_plus_1": next_close.to_numpy(),
            "target_date": next_date.to_numpy(),
        })
        # ``close_t`` must be present too: the wide close frame is the union of
        # every security's calendar, so a date on which THIS security had no bar
        # (or a non-positive bar, which the adjusted series carries for a few
        # pre-IPO-era ADANIENT rows) arrives as NaN and would otherwise survive.
        frame = frame.dropna(subset=["close_t", "close_t_plus_1", "target_date"])
        frame = frame.loc[frame["target_date"] <= pd.Timestamp(max_date)]
        if frame.empty:
            continue
        key = pd.MultiIndex.from_arrays([frame["ticker"], frame["origin_date"]])
        frame["realized_vol_20"] = vol.reindex(key).to_numpy()
        frame["raw_next_return"] = np.log(
            frame["close_t_plus_1"].to_numpy() / frame["close_t"].to_numpy())
        long_rows.append(frame)
    targets = pd.concat(long_rows, ignore_index=True)
    targets["y_direction"] = (targets["close_t_plus_1"] > targets["close_t"]).astype(int)

    scale = targets["realized_vol_20"].clip(lower=feat.VOL_FLOOR)
    targets["y_return"] = (targets["raw_next_return"] / scale).clip(
        -RETURN_TARGET_CLIP, RETURN_TARGET_CLIP)
    targets.loc[targets["realized_vol_20"].isna(), "y_return"] = np.nan

    # Cross-sectional rank of the next-day return on the TARGET date.  Uses
    # t+1 information, which is legitimate for a LABEL.
    wide = targets.pivot(index="target_date", columns="ticker", values="raw_next_return")
    rank_long = (wide.rank(axis=1, pct=True, method="average")
                     .stack(future_stack=True).rename("y_rank")
                     .reset_index()[["ticker", "target_date", "y_rank"]])
    targets = targets.merge(rank_long, on=["ticker", "target_date"], how="left",
                            validate="one_to_one")

    keep = ["ticker", "origin_date", "target_date", "realized_vol_20",
            "raw_next_return", "y_direction", "y_return", "y_rank"]
    targets = targets.loc[:, keep].sort_values(["ticker", "origin_date"]).reset_index(drop=True)

    assert_no_paper_test_targets(targets["target_date"], where="v2 store target frame")
    return targets


def build_store(dataset_root: str | Path, *, variant: str, universe_id: str,
                tickers: list[str], sector_map: SectorMap, max_date: str = "2021-12-31",
                root: str | Path | None = None, force: bool = False,
                config_extra: dict | None = None) -> FeatureStore:
    """Build (or reuse) the V2 context store."""
    dataset_root = Path(dataset_root)
    out_root = store_root(root)
    ensure_dir(out_root)

    from .sectors import read_sheet_name_map

    sheet_map = read_sheet_name_map(dataset_root)
    manifest_hash, manifest_info = _source_manifest_hash(dataset_root)
    config_payload = _store_fingerprint_payload(
        dataset_root=dataset_root, variant=variant, universe_id=universe_id,
        tickers=tickers, sector_map=sector_map, max_date=max_date,
        manifest_hash=manifest_hash, config_extra=config_extra,
    )
    config_hash = _hash_config(config_payload)

    metadata_path = out_root / "metadata.json"
    if metadata_path.is_file() and not force:
        cached = json.loads(metadata_path.read_text())
        if cached.get("config_sha256") == config_hash and all(
            (out_root / name).is_file() for name in STORE_FILES
        ):
            logger.info("V2 feature store cache hit: %s", out_root)
            return load_store(root)

    if pd.Timestamp(max_date) >= V2_PAPER_TEST_FIREWALL_START:
        raise ValueError(
            f"max_date={max_date} reaches the paper-test firewall "
            f"({V2_PAPER_TEST_FIREWALL_START.date()}). The V2 store is capped at "
            "2021-12-31 so that no 2022/2023 label can exist."
        )

    logger.info("Building V2 feature store from %s (%s)", dataset_root, variant)
    stock, quality = build_stock_feature_frame(
        dataset_root, variant, tickers, sheet_map, max_date=max_date)

    sector_of = sector_map.sector_series()
    context_frame, market_frame, ctx_diagnostics = ctx.assemble_context_features(
        stock, sector_of)

    sector_frame = context_frame.loc[:, ["ticker", "date", *ctx.SECTOR_FEATURES]].merge(
        sector_map.frame.loc[:, ["ticker", "industry", "broad_sector"]],
        on="ticker", how="left", validate="many_to_one")

    # The leave-one-out market aggregates, the relative-to-market/sector features
    # and the percentile ranks are all CROSS-SECTIONAL quantities, so they are
    # persisted together in cross_sectional_features.parquet.
    cross_columns = tuple(dict.fromkeys(
        (*ctx.MARKET_LOO_FEATURES, *ctx.RELATIVE_FEATURES,
         *ctx.RANK_FEATURES, *ctx.REGIME_FEATURES)))
    cross_frame = context_frame.loc[:, ["ticker", "date", *cross_columns]].reset_index(
        drop=True)

    targets = build_target_frame(dataset_root, variant, tickers, sheet_map, stock,
                                 max_date=max_date)

    stock.to_parquet(out_root / "stock_features.parquet", index=False)
    market_frame.to_parquet(out_root / "market_context.parquet", index=False)
    sector_frame.to_parquet(out_root / "sector_context.parquet", index=False)
    cross_frame.to_parquet(out_root / "cross_sectional_features.parquet", index=False)
    targets.to_parquet(out_root / "target_frame.parquet", index=False)

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_dataset_root": str(dataset_root),
        "universe_id": universe_id,
        "data_variant": variant,
        "adjusted_or_unadjusted": variant,
        "source_manifest_sha256": manifest_hash,
        "source_manifest_info": manifest_info,
        "feature_code_sha256": feature_code_sha256(),
        "config_sha256": config_hash,
        "sector_map_sha256": sector_map.sha256(),
        "sector_map_coverage": sector_map.coverage(),
        "max_date": str(max_date),
        "max_target_date": str(pd.Timestamp(targets["target_date"].max()).date())
        if len(targets) else None,
        "n_tickers": int(stock["ticker"].nunique()),
        "date_range": {
            "first_input_date": str(pd.Timestamp(stock["date"].min()).date()),
            "last_input_date": str(pd.Timestamp(stock["date"].max()).date()),
            "first_target_date": str(pd.Timestamp(targets["target_date"].min()).date()),
            "last_target_date": str(pd.Timestamp(targets["target_date"].max()).date()),
        },
        "feature_schema": {
            "stock_features": list(feat.STOCK_FEATURE_NAMES),
            "context_features": list(ctx.CONTEXT_FEATURES),
            "rank_features": list(ctx.RANK_FEATURES),
            "percentile_features": list(ctx.PERCENTILE_FEATURES),
            "regime_features": list(ctx.REGIME_FEATURES),
            "target_columns": ["y_direction", "y_return", "y_rank"],
            "return_target_clip": RETURN_TARGET_CLIP,
        },
        "rows": {
            "stock_features": len(stock),
            "market_context": len(market_frame),
            "sector_context": len(sector_frame),
            "cross_sectional_features": len(cross_frame),
            "target_frame": len(targets),
        },
        "context_diagnostics": ctx_diagnostics,
        "feature_quality": quality,
        "firewall": {
            "paper_test_firewall_start": str(V2_PAPER_TEST_FIREWALL_START.date()),
            "store_capped_at": str(max_date),
            "no_2022_or_2023_label_in_store": True,
        },
    }
    atomic_json_dump(metadata, metadata_path)
    logger.info("V2 feature store written to %s", out_root)
    return load_store(root)


def load_store(root: str | Path | None = None) -> FeatureStore:
    """Load a previously built store (no rebuild, no network, no writes)."""
    base = store_root(root)
    metadata_path = base / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"No V2 feature store at {base}. Run scripts/build_v2_context.py first."
        )
    metadata = json.loads(metadata_path.read_text())
    store = FeatureStore(
        root=base,
        stock=pd.read_parquet(base / "stock_features.parquet"),
        market=pd.read_parquet(base / "market_context.parquet"),
        sector=pd.read_parquet(base / "sector_context.parquet"),
        cross_sectional=pd.read_parquet(base / "cross_sectional_features.parquet"),
        targets=pd.read_parquet(base / "target_frame.parquet"),
        metadata=metadata,
    )
    assert_no_paper_test_targets(store.targets["target_date"], where="v2 store load")
    return store


def store_fingerprints(root: str | Path | None = None) -> dict:
    """Fingerprint dict used for cache validation and for the ledger."""
    base = store_root(root)
    metadata_path = base / "metadata.json"
    if not metadata_path.is_file():
        return {"root": str(base), "built": False}
    metadata = json.loads(metadata_path.read_text())
    store = FeatureStore(
        root=base,
        stock=pd.DataFrame(),
        market=pd.DataFrame(),
        sector=pd.DataFrame(),
        cross_sectional=pd.DataFrame(),
        targets=pd.DataFrame(),
        metadata=metadata,
    )
    return {
        "root": str(base),
        "built": True,
        "config_sha256": metadata.get("config_sha256"),
        "source_manifest_sha256": metadata.get("source_manifest_sha256"),
        "feature_code_sha256": metadata.get("feature_code_sha256"),
        "sector_map_sha256": metadata.get("sector_map_sha256"),
        "store_sha256": store.store_sha256,
        "max_target_date": metadata.get("max_target_date"),
    }


def store_is_current(root: str | Path | None = None, *,
                     dataset_root: str | Path, variant: str, universe_id: str,
                     tickers: list[str], sector_map: SectorMap,
                     max_date: str = "2021-12-31") -> bool:
    """True when the cached store matches every current fingerprint."""
    base = store_root(root)
    metadata_path = base / "metadata.json"
    if not metadata_path.is_file() or not all(
        (base / name).is_file() for name in STORE_FILES
    ):
        return False
    manifest_hash, _ = _source_manifest_hash(Path(dataset_root))
    payload = _store_fingerprint_payload(
        dataset_root=Path(dataset_root), variant=variant, universe_id=universe_id,
        tickers=tickers, sector_map=sector_map, max_date=max_date,
        manifest_hash=manifest_hash, config_extra=None,
    )
    return json.loads(metadata_path.read_text()).get("config_sha256") == _hash_config(payload)