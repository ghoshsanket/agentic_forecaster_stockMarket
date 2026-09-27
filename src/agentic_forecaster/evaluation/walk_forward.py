"""Walk-forward (anchored) evaluation folds."""

from __future__ import annotations

from datetime import date

import pandas as pd

PAPER_FOLD_DATES = [
    {
        "fold": 0,
        "train_start": date(2016, 1, 1),
        "train_end": date(2020, 12, 31),
        "val_start": date(2021, 1, 1),
        "val_end": date(2021, 12, 31),
        "test_start": date(2022, 1, 1),
        "test_end": date(2022, 12, 31),
    },
    {
        "fold": 1,
        "train_start": date(2016, 1, 1),
        "train_end": date(2021, 12, 31),
        "val_start": date(2022, 1, 1),
        "val_end": date(2022, 12, 31),
        "test_start": date(2023, 1, 1),
        "test_end": date(2023, 12, 31),
    },
]


def walk_forward_folds(
    n_samples: int,
    seq_len: int,
    step: int = 21,
    min_train: int = 252,
) -> list[dict]:
    """Generate anchored walk-forward train/test index ranges.

    Each fold trains on ``[0, train_end)`` and tests on
    ``[train_end, train_end + step)``.  The training window grows
    (anchored) as the test window advances.
    """
    folds = []
    train_end = min_train
    while train_end + step <= n_samples:
        folds.append(
            {
                "train_start": 0,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": train_end + step,
            }
        )
        train_end += step
    return folds


def paper_walk_forward_folds() -> list[dict]:
    """Return the paper's two specific walk-forward folds.

    Fold 0 trains on 2016-01-01 through 2020-12-31, validates on
    2021-01-01 through 2021-12-31, and tests on 2022-01-01 through
    2022-12-31.

    Fold 1 trains on 2016-01-01 through 2021-12-31, validates on
    2022-01-01 through 2022-12-31, and tests on 2023-01-01 through
    2023-12-31.
    """
    return [dict(fold) for fold in PAPER_FOLD_DATES]


def check_fold_date_coverage(
    df: pd.DataFrame,
    folds: list[dict],
    date_col: str = "date",
) -> list[dict]:
    """Check whether ``df`` contains the date ranges required by each fold.

    Parameters
    ----------
    df:
        DataFrame containing a date column.
    folds:
        Fold dicts as returned by :func:`paper_walk_forward_folds`, each
        with ``train_start``, ``train_end``, ``val_start``, ``val_end``,
        ``test_start``, and ``test_end`` keys.
    date_col:
        Name of the date column in ``df``.

    Returns
    -------
    list[dict]
        One dict per fold containing the fold index, whether each
        split's full date range is available (``train_available``,
        ``val_available``, ``test_available``), and the observed row
        counts and date bounds for each split.
    """
    if date_col not in df.columns:
        raise ValueError(f"date column {date_col!r} not found in DataFrame")

    dates = pd.to_datetime(df[date_col]).dt.date

    coverage = []
    for fold in folds:
        available_info = {}
        counts_info = {}
        bounds_info = {}
        for split in ("train", "val", "test"):
            start = pd.Timestamp(fold[f"{split}_start"]).date()
            end = pd.Timestamp(fold[f"{split}_end"]).date()
            mask = (dates >= start) & (dates <= end)
            split_dates = dates[mask]
            available_info[f"{split}_available"] = (
                len(split_dates) >= (end - start).days + 1
            )
            counts_info[f"{split}_count"] = int(mask.sum())
            bounds_info[f"{split}_actual_start"] = split_dates.min() if len(split_dates) else None
            bounds_info[f"{split}_actual_end"] = split_dates.max() if len(split_dates) else None
        coverage.append(
            {
                "fold": fold["fold"],
                "train_available": available_info["train_available"],
                "val_available": available_info["val_available"],
                "test_available": available_info["test_available"],
                "train_count": counts_info["train_count"],
                "val_count": counts_info["val_count"],
                "test_count": counts_info["test_count"],
                "train_actual_start": bounds_info["train_actual_start"],
                "train_actual_end": bounds_info["train_actual_end"],
                "val_actual_start": bounds_info["val_actual_start"],
                "val_actual_end": bounds_info["val_actual_end"],
                "test_actual_start": bounds_info["test_actual_start"],
                "test_actual_end": bounds_info["test_actual_end"],
                "all_available": (
                    available_info["train_available"]
                    and available_info["val_available"]
                    and available_info["test_available"]
                ),
            }
        )
    return coverage


def filter_df_by_fold(
    df: pd.DataFrame,
    fold: dict,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter ``df`` into train, validation, and test sets for a fold.

    Parameters
    ----------
    df:
        DataFrame containing a date column.
    fold:
        Fold dict as returned by :func:`paper_walk_forward_folds`, each
        with ``train_start``, ``train_end``, ``val_start``, ``val_end``,
        ``test_start``, and ``test_end`` keys.
    date_col:
        Name of the date column in ``df``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_df, val_df, test_df)`` filtered to the fold's
        respective date ranges (inclusive on both ends).
    """
    if date_col not in df.columns:
        raise ValueError(f"date column {date_col!r} not found in DataFrame")

    dates = pd.to_datetime(df[date_col]).dt.date

    def _filter(split: str) -> pd.DataFrame:
        start = pd.Timestamp(fold[f"{split}_start"]).date()
        end = pd.Timestamp(fold[f"{split}_end"]).date()
        return df.loc[(dates >= start) & (dates <= end)].copy()

    return _filter("train"), _filter("val"), _filter("test")
