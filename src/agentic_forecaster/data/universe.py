"""Ticker-universe resolution against the downloaded dataset.

Applies the reconstruction policy documented in ``configs/nifty50.yaml``:

* ``tickers``            — requested NIFTY-50 symbols
* ``aliases``            — genuine symbol-format spellings of the SAME security
* ``known_unavailable``  — symbols absent from the dataset (never substituted)

A symbol is never mapped to a different company.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger("agentic_forecaster.data.universe")


@dataclass
class Universe:
    requested: list[str] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)
    known_unavailable: list[str] = field(default_factory=list)
    available: dict[str, str] = field(default_factory=dict)   # requested -> dataset symbol
    unavailable: dict[str, str] = field(default_factory=dict)  # requested -> reason

    @property
    def n_available(self) -> int:
        return len(self.available)

    @property
    def n_requested(self) -> int:
        return len(self.requested)


def load_universe_config(path: str | Path) -> Universe:
    """Load the universe declaration from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return Universe(
        requested=[str(t).upper() for t in raw.get("tickers", [])],
        aliases={str(k).upper(): str(v).upper() for k, v in (raw.get("aliases") or {}).items()},
        known_unavailable=[str(t).upper() for t in raw.get("known_unavailable", [])],
    )


def resolve_universe(
    universe: Universe,
    discovered: dict[str, object],
) -> Universe:
    """Resolve requested symbols against ``discovered`` (symbol -> path)."""
    resolved = Universe(
        requested=list(universe.requested),
        aliases=dict(universe.aliases),
        known_unavailable=list(universe.known_unavailable),
    )
    for symbol in universe.requested:
        if symbol in discovered:
            resolved.available[symbol] = symbol
        elif symbol in universe.aliases and universe.aliases[symbol] in discovered:
            resolved.available[symbol] = universe.aliases[symbol]
        elif symbol in universe.known_unavailable:
            resolved.unavailable[symbol] = (
                "not present in the downloaded dataset (listed in "
                "configs/nifty50.yaml known_unavailable)"
            )
        else:
            resolved.unavailable[symbol] = "symbol not found in downloaded dataset"
    logger.info(
        "Universe resolved: %d/%d available, %d unavailable",
        resolved.n_available, resolved.n_requested, len(resolved.unavailable),
    )
    return resolved
