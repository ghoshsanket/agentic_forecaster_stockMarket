#!/usr/bin/env python3
"""Build the causally aligned V3 exogenous store, then run the AVAILABILITY AUDIT.

Two steps, in this order, and the second one can stop the programme:

1. BUILD the store.  Every ACCEPTED source is aligned onto the STOCK trading dates
   with merge-asof under its OWN availability class, so an Indian same-close source
   may use date ``t`` while a foreign source is forced to the previous observation.
   Each aligned value keeps its provenance: the source observation date used, and
   the lag in source trading observations.

2. AUDIT availability.  At least 100 deterministically sampled stock origin dates are
   checked, and every exogenous feature must come from a source observation no later
   than the NSE prediction timestamp.  Any violation is a STOP, not a warning.

Outputs::

    $AGENTIC_PROCESSED_DATA_ROOT/v3/pre_covid_exogenous/exogenous_store/
    results/v3/pre_covid_exogenous/AVAILABILITY_AUDIT.md
    results/v3/pre_covid_exogenous/availability_audit.json

Usage::

    uv run python scripts/build_v3_exogenous_store.py
    uv run python scripts/build_v3_exogenous_store.py --audit-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.utils import atomic_json_dump, ensure_dir, setup_logging
from agentic_forecaster.v2.horizons import load_track_config
from agentic_forecaster.v2.sectors import load_sector_map
from agentic_forecaster.v2.store import load_store
from agentic_forecaster.v3 import V3Track
from agentic_forecaster.v3.sources import load_effective_registry
from agentic_forecaster.v3.store import (
    availability_audit,
    build_exogenous_store,
    load_exogenous_store,
    render_availability_audit,
    store_fingerprints,
)

CONFIG = REPO_ROOT / "configs" / "v3" / "precovid_exogenous_base.yaml"


def sector_index_symbols(registry_path: Path) -> dict[str, str]:
    """The declared sector -> external index symbol map, verbatim."""
    import yaml

    payload = yaml.safe_load(Path(registry_path).read_text(encoding="utf-8"))
    return dict(payload.get("sector_index_map") or {})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args(argv)
    setup_logging()

    payload = load_track_config(args.config)
    data = payload["data"]
    final_allowed = str(payload["final_allowed_date"])
    track = V3Track()
    ensure_dir(track.results_root)

    if not args.audit_only:
        stock_store = load_store(Path(data["source_store_root"]),
                                 final_allowed_date=final_allowed)
        sector_map = load_sector_map(Path(data["sector_map_csv"]))
        registry = load_effective_registry(
            Path(data["resolved_registry"]), Path(data["declared_registry"]),
            track.path("source_audit_json"))
        manifest_path = Path(data["source_manifest"])
        manifest_sha = (json.loads(manifest_path.read_text())["manifest_sha256"]
                        if manifest_path.is_file() else None)
        symbols = sector_index_symbols(
            REPO_ROOT / "configs" / "v3" / "source_registry.yaml")

        store = build_exogenous_store(
            registry,
            stock_dates=stock_store.stock["date"],
            sector_of=sector_map.sector_series(),
            sector_index_symbols=symbols,
            root=Path(data["exogenous_store"]),
            final_allowed_date=final_allowed,
            source_manifest_sha256=manifest_sha)
        print(json.dumps(store.summary(), indent=2))
    else:
        store = load_exogenous_store(Path(data["exogenous_store"]),
                                     final_allowed_date=final_allowed)

    audit = availability_audit(store, sample_size=args.sample_size,
                              seed=int(payload["evaluation"]["bootstrap"]["seed"]))
    atomic_json_dump(audit, track.path("availability_audit_json"))
    track.path("availability_audit").write_text(render_availability_audit(audit),
                                                encoding="utf-8")
    print(json.dumps({"availability_audit_passed": audit["passed"],
                      "violations": audit["n_violations"],
                      "sampled_dates": audit["sample_size"],
                      "sources_checked": len(audit["per_source"]),
                      "store": store_fingerprints(Path(data["exogenous_store"]))},
                     indent=2))
    if not audit["passed"]:
        print("STOP: an exogenous value was available later than the prediction "
              "timestamp. The V3 programme must not continue.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())