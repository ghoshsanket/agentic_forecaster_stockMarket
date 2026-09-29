"""Search and paper walk-forward window definitions.

Two distinct families of folds exist, and they must never be mixed:

**Search folds (A/B/C)** - pre-2022 pseudo walk-forward windows used ONLY for
recovering lost implementation choices.  Every fold is strictly before 2022, so
the paper test years are never touched.

**Paper folds (fold_0/fold_1)** - the publication's exact two folds.  These
evaluate 2022 and 2023 and are reachable only through the final, explicitly
frozen run guarded by ``FINAL_TEST=1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from agentic_forecaster.recovery.firewall import (
    TEST_FIREWALL_START,
    assert_config_windows_safe,
)


@dataclass(frozen=True)
class SearchFold:
    """A pre-2022 pseudo walk-forward window used for model search."""

    name: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    description: str = ""

    def as_data_cfg(self) -> dict[str, str]:
        return {
            "train_start": self.train_start,
            "train_end": self.train_end,
            "val_start": self.val_start,
            "val_end": self.val_end,
        }

    def to_dict(self) -> dict[str, str]:
        return {
            "search_fold": self.name,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "val_start": self.val_start,
            "val_end": self.val_end,
            "description": self.description,
        }

    def assert_safe(self, search: bool | None = True) -> None:
        assert_config_windows_safe(self.as_data_cfg(), where=self.name, search=search)


#: Pre-2022 search windows.  All strictly before TEST_FIREWALL_START.
SEARCH_FOLDS: dict[str, SearchFold] = {
    "SEARCH_FOLD_A": SearchFold(
        name="SEARCH_FOLD_A",
        train_start="2016-01-01", train_end="2018-12-31",
        val_start="2019-01-01", val_end="2019-12-31",
        description="Shortest train window; probes whether the pipeline can learn at all.",
    ),
    "SEARCH_FOLD_B": SearchFold(
        name="SEARCH_FOLD_B",
        train_start="2016-01-01", train_end="2019-12-31",
        val_start="2020-01-01", val_end="2020-12-31",
        description="Intermediate train window; includes the COVID-shock regime.",
    ),
    "SEARCH_FOLD_C": SearchFold(
        name="SEARCH_FOLD_C",
        train_start="2016-01-01", train_end="2020-12-31",
        val_start="2021-01-01", val_end="2021-12-31",
        description="Closest pre-test analogue of paper fold 0's train/val split.",
    ),
}

#: The publication's exact two folds.  These evaluate the firewalled years.
PAPER_FOLDS: tuple[dict[str, str], ...] = (
    {
        "fold": "fold_0",
        "train_start": "2016-01-01", "train_end": "2020-12-31",
        "val_start": "2021-01-01", "val_end": "2021-12-31",
        "test_start": "2022-01-01", "test_end": "2022-12-31",
    },
    {
        "fold": "fold_1",
        "train_start": "2016-01-01", "train_end": "2021-12-31",
        "val_start": "2022-01-01", "val_end": "2022-12-31",
        "test_start": "2023-01-01", "test_end": "2023-12-31",
    },
)


def get_search_fold(name: str) -> SearchFold:
    try:
        return SEARCH_FOLDS[name]
    except KeyError:
        raise KeyError(
            f"Unknown search fold {name!r}. Available: {sorted(SEARCH_FOLDS)}"
        ) from None


def fold_config_for(name: str, base_config: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-ish copy of ``base_config`` with a search fold applied.

    The copy has the fold's train/val windows substituted and its ``test_*``
    window removed, because a search fold has no test period.  The returned
    config is firewall-validated before it is returned.
    """
    import copy

    fold = get_search_fold(name)
    cfg = copy.deepcopy(base_config)
    data = cfg.setdefault("data", {})
    data.update(fold.as_data_cfg())
    # A search fold has NO test window. Drop any inherited one so it can never
    # be evaluated by accident.
    data.pop("test_start", None)
    data.pop("test_end", None)
    cfg["experiment"] = copy.deepcopy(cfg.get("experiment", {}))
    cfg["experiment"]["name"] = f"{cfg['experiment'].get('name', 'run')}_{name}"
    fold.assert_safe()
    return cfg


def search_fold_train_span(fold: SearchFold) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (pd.Timestamp(fold.train_start), pd.Timestamp(fold.train_end))


def search_fold_val_span(fold: SearchFold) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (pd.Timestamp(fold.val_start), pd.Timestamp(fold.val_end))


def assert_fold_is_pre_test(name: str) -> None:
    """Defensive assertion used by the harness before any training starts."""
    fold = get_search_fold(name)
    if pd.Timestamp(fold.val_end) >= TEST_FIREWALL_START:
        raise AssertionError(
            f"{name} evaluates on/after {TEST_FIREWALL_START.date()}; it is not a "
            "legal search fold."
        )


def describe_folds() -> list[dict]:
    return [f.to_dict() for f in SEARCH_FOLDS.values()]
