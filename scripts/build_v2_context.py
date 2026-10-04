#!/usr/bin/env python3
"""Build the V2 sector map and the cached V2 context store.

Two steps, both idempotent and both fingerprinted:

1. SECTOR MAP.  Reuse a previously downloaded official NSE constituent CSV when
   one exists; otherwise fetch ONLY that official file (the one used during
   universe recovery) and store it, with its SHA-256 and retrieval timestamp,
   under ``$AGENTIC_PROCESSED_DATA_ROOT/v2/metadata/``.  A ticker with no
   official industry record is written as ``UNKNOWN`` and REPORTED -- never
   guessed.  Static sector metadata is reference data, not a market time series,
   so no additional time-series download happens here.

2. CONTEXT STORE.  Build (or reuse) the causal feature store: stock features,
   market context, sector context, cross-sectional features, targets and
   metadata.  The store is HARD-CAPPED at 2021-12-31, so no 2022/2023 label can
   exist anywhere in V2.  Source data is only ever read.

Usage::

    uv run python scripts/build_v2_context.py --dataset-root <path>
    uv run python scripts/build_v2_context.py --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.config import get_env_roots
from agentic_forecaster.utils import setup_logging
from agentic_forecaster.v2.sectors import (
    build_sector_map,
    load_official_nse_csv,
    load_sector_map,
    read_universe_tickers,
    write_sector_map_csv,
    write_sector_map_metadata,
    write_sector_map_yaml,
)
from agentic_forecaster.v2.store import (
    build_store,
    store_fingerprints,
    store_is_current,
    store_root,
    v2_metadata_dir,
    v2_processed_root,
)

logger = logging.getLogger("build_v2_context")

DEFAULT_UNIVERSE = REPO_ROOT / "configs" / "nifty50_paper_snapshot_2025_11_04.yaml"
DEFAULT_SECTOR_YAML = REPO_ROOT / "configs" / "v2" / "sector_map.yaml"


def _default_dataset_root() -> Path:
    return Path(get_env_roots()["AGENTIC_DATA_ROOT"]) / "yfinance_paper_snapshot_2025_11_04"


def build_sector_mapping(dataset_root: Path, universe_config: Path, *,
                         sector_yaml: Path, allow_download: bool) -> dict:
    """Create the V2 sector map and write both persisted copies."""
    tickers, universe_id = read_universe_tickers(universe_config)
    metadata_dir = v2_metadata_dir()
    csv_path, digest = load_official_nse_csv(metadata_dir, allow_download=allow_download)
    sector_map = build_sector_map(tickers, csv_path, source_sha256=digest)
    csv_out = metadata_dir / "sector_map.csv"
    write_sector_map_csv(sector_map, csv_out)
    write_sector_map_yaml(sector_map, sector_yaml, universe_id=universe_id,
                          csv_path=csv_out)
    write_sector_map_metadata(sector_map, metadata_dir / "sector_map_metadata.json",
                               extra={"universe_config": str(universe_config),
                                      "official_csv": str(csv_path),
                                      "universe_id": universe_id})
    coverage = sector_map.coverage()
    logger.info("sector map: %d/%d mapped, %d UNKNOWN", coverage["n_mapped"],
                coverage["n_tickers"], coverage["n_unmapped"])
    if coverage["unmapped_tickers"]:
        logger.warning("UNMAPPED tickers (assigned UNKNOWN, not guessed): %s",
                       ", ".join(coverage["unmapped_tickers"]))
    return coverage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None,
                        help="paper-snapshot dataset root (default: "
                             "$AGENTIC_DATA_ROOT/yfinance_paper_snapshot_2025_11_04)")
    parser.add_argument("--universe-config", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--variant", default="adjusted",
                        choices=["adjusted", "unadjusted"])
    parser.add_argument("--max-date", default="2021-12-31")
    parser.add_argument("--store-root", type=Path, default=None,
                        help="processed BRANCH that holds context_store/ "
                             "(default: $AGENTIC_PROCESSED_DATA_ROOT/v2). Use "
                             "$AGENTIC_PROCESSED_DATA_ROOT/v2/pre_covid for the "
                             "PRE-COVID track so the physical store is separate.")
    parser.add_argument("--experiment-regime", default="ORDINARY_V2",
                        help="recorded in metadata.json (e.g. PRE_COVID)")
    parser.add_argument("--final-allowed-date", default=None,
                        help="regime boundary; the build refuses to write anything "
                             "after it (PRE-COVID uses 2019-12-31)")
    parser.add_argument("--sector-yaml", type=Path, default=DEFAULT_SECTOR_YAML)
    parser.add_argument("--no-download", action="store_true",
                        help="refuse to fetch the official NSE file; a previously "
                             "downloaded copy must already exist")
    parser.add_argument("--force", action="store_true", help="rebuild the store")
    parser.add_argument("--verify", action="store_true",
                        help="verify an existing store's fingerprints and exit")
    args = parser.parse_args(argv)

    setup_logging()
    dataset_root = args.dataset_root or _default_dataset_root()
    if not dataset_root.is_dir():
        logger.error("dataset root not found: %s", dataset_root)
        return 2
    if not (dataset_root / args.variant / "parquet").is_dir():
        logger.error("no %s parquet directory under %s", args.variant, dataset_root)
        return 2

    processed_root = Path(args.store_root) if args.store_root else v2_processed_root()
    logger.info("processed branch   : %s", processed_root)
    logger.info("V2 store root     : %s", store_root(processed_root))
    logger.info("source dataset    : %s (%s)", dataset_root, args.variant)

    coverage = build_sector_mapping(
        dataset_root, args.universe_config, sector_yaml=args.sector_yaml,
        allow_download=not args.no_download)

    tickers, universe_id = read_universe_tickers(args.universe_config)
    sector_map = load_sector_map(v2_metadata_dir() / "sector_map.csv")

    if args.verify:
        current = store_is_current(dataset_root=dataset_root, variant=args.variant,
                                   universe_id=universe_id, tickers=tickers,
                                   sector_map=sector_map, max_date=args.max_date,
                                   root=processed_root,
                                   experiment_regime=args.experiment_regime,
                                   final_allowed_date=args.final_allowed_date)
        fingerprints = store_fingerprints(processed_root)
        logger.info("fingerprints: %s", json.dumps(fingerprints, indent=2))
        if not current:
            logger.error("store is stale or missing: rebuild without --verify")
            return 1
        logger.info("store is current for these fingerprints")
        return 0

    store = build_store(dataset_root, variant=args.variant, universe_id=universe_id,
                        tickers=tickers, sector_map=sector_map, max_date=args.max_date,
                        root=processed_root, force=args.force,
                        experiment_regime=args.experiment_regime,
                        final_allowed_date=args.final_allowed_date)
    summary = store.summary() | {"sector_map_coverage": coverage,
                                 "experiment_regime": args.experiment_regime,
                                 "final_allowed_date": args.final_allowed_date}
    print(json.dumps(summary, indent=2))
    logger.info("sector coverage: %d/%d mapped (%s)", coverage["n_mapped"],
                coverage["n_tickers"], coverage["source_sha256"][:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())