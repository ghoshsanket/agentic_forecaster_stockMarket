#!/usr/bin/env python3
"""Download the ACCEPTED external source snapshots into Research-local raw storage.

Two passes, in this order:

1. PROBE every declared source WITHOUT storing, so the coverage gate decides what is
   actually worth keeping;
2. DOWNLOAD only the accepted candidates, writing immutable snapshots plus the
   manifest with a SHA-256 per file and a ``SOURCE_PROVENANCE.md``.

A raw file is never modified after download: a re-run reuses the existing snapshot
rather than re-fetching it, so an already-reported result can never be silently
re-based on new data.  Use ``scripts/probe_v3_sources.py`` to force a re-probe.

Usage::

    uv run python scripts/download_v3_exogenous.py
    uv run python scripts/download_v3_exogenous.py --refresh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.utils import atomic_json_dump, ensure_dir, setup_logging
from agentic_forecaster.v3 import V3Track, data_root, raw_root
from agentic_forecaster.v3.download import (
    download_accepted_sources,
    write_manifest,
    write_provenance_document,
)
from agentic_forecaster.v3.sources import load_registry

CONFIG = REPO_ROOT / "configs" / "v3" / "source_registry.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--refresh", action="store_true",
                        help="delete existing snapshots and re-download every source")
    parser.add_argument("--max-missing-fraction", type=float, default=0.05)
    args = parser.parse_args(argv)
    setup_logging()

    registry = load_registry(args.config)
    track = V3Track()
    ensure_dir(track.results_root)
    if args.refresh and raw_root().exists():
        for path in sorted(raw_root().rglob("*.parquet")):
            path.unlink()
        print(f"[refresh] removed existing snapshots under {raw_root()}")

    results = download_accepted_sources(registry, raw_root=raw_root(),
                                        max_missing_fraction=args.max_missing_fraction)
    for result in results:
        spec = registry.get(result.spec.source_id)
        print(f"[download] {spec.source_id:<20} "
              f"{spec.provider_symbol_or_identifier:<16} "
              f"{'ACCEPTED' if spec.accepted else 'REJECTED':<9} "
              f"rows={result.row_count or 0:<6} "
              f"sha256={(result.raw_sha256 or '-')[:16]}", flush=True)

    manifest = write_manifest(registry, manifests_root=data_root() / "manifests")
    write_provenance_document(manifests_root=data_root() / "manifests", registry=registry)
    atomic_json_dump(registry.to_dict(), track.path("registry"))

    print(json.dumps({"raw_root": str(raw_root()),
                      "n_accepted": manifest["n_accepted"],
                      "n_declared": manifest["n_declared"],
                      "manifest_sha256": manifest["manifest_sha256"],
                      "provenance_document": str(data_root() / "manifests"
                                                  / "SOURCE_PROVENANCE.md")},
                     indent=2))
    if manifest["n_accepted"] == 0:
        print("STOP: no external source passed the coverage gate, so there is nothing "
              "to test. The V3 programme must not continue.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())