import pandas as pd


def temporal_split_by_date(
    df: pd.DataFrame,
    train_start,
    train_end,
    val_start,
    val_end,
    test_start,
    test_end,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(df[date_col])

    train_mask = (dates >= pd.Timestamp(train_start)) & (dates <= pd.Timestamp(train_end))
    val_mask = (dates >= pd.Timestamp(val_start)) & (dates <= pd.Timestamp(val_end))
    test_mask = (dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    return train_df, val_df, test_df


def temporal_split_by_fraction(
    df: pd.DataFrame,
    train_frac: float = 0.85,
    val_frac: float = 0.10,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1.0:
        raise ValueError("Fractions must be positive and sum to less than 1.0")

    sorted_df = df.sort_values(by=date_col).reset_index(drop=True)
    dates = pd.to_datetime(sorted_df[date_col])

    min_date = dates.min()
    max_date = dates.max()
    total_days = (max_date - min_date).days

    train_region_end = min_date + pd.Timedelta(days=total_days * train_frac)
    test_start = train_region_end + pd.Timedelta(days=1)

    train_region = sorted_df[dates <= train_region_end]
    train_region_dates = pd.to_datetime(train_region[date_col])
    train_region_days = (train_region_dates.max() - train_region_dates.min()).days

    val_start = train_region_dates.min() + pd.Timedelta(
        days=train_region_days * (1.0 - val_frac)
    )

    train_df = train_region[train_region_dates < val_start].copy()
    val_df = train_region[train_region_dates >= val_start].copy()
    test_df = sorted_df[dates >= test_start].copy()

    return train_df, val_df, test_df


def walk_forward_folds(df: pd.DataFrame, date_col: str = "date") -> list[dict]:
    dates = pd.to_datetime(df[date_col])
    data_start = dates.min()
    data_end = dates.max()

    folds = []
    fold_index = 0

    train_start = pd.Timestamp("2016-01-01")
    val_start = pd.Timestamp("2021-01-01")
    test_start = pd.Timestamp("2022-01-01")
    test_end = pd.Timestamp("2022-12-31")

    while True:
        train_end = val_start - pd.Timedelta(days=1)
        val_end = test_start - pd.Timedelta(days=1)

        fold = {
            "fold": fold_index,
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
        }

        if train_start < data_start:
            fold["available"] = False
            fold["reason"] = (
                f"Data starts {data_start.date()}, after required train start {train_start.date()}"
            )
        elif test_end > data_end:
            fold["available"] = False
            fold["reason"] = (
                f"Data ends {data_end.date()}, before required test end {test_end.date()}"
            )
        else:
            fold["available"] = True
            fold["reason"] = None

        folds.append(fold)

        if test_end >= data_end:
            break

        val_start = test_start
        test_start = test_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        fold_index += 1

    return folds


def check_fold_availability(df: pd.DataFrame, folds: list[dict], date_col: str = "date") -> list[dict]:
    dates = pd.to_datetime(df[date_col])

    ticker_col = "ticker" if "ticker" in df.columns else None

    availability = []
    for fold in folds:
        fold_info = {
            "fold": fold["fold"],
            "available": fold.get("available", False),
            "reason": fold.get("reason"),
            "train_start": fold["train_start"],
            "train_end": fold["train_end"],
            "val_start": fold["val_start"],
            "val_end": fold["val_end"],
            "test_start": fold["test_start"],
            "test_end": fold["test_end"],
        }

        for split_name in ["train", "val", "test"]:
            split_start = fold[f"{split_name}_start"]
            split_end = fold[f"{split_name}_end"]
            split_mask = (dates >= split_start) & (dates <= split_end)
            split_df = df[split_mask]

            split_info = {
                "observation_count": len(split_df),
                "ticker_count": (
                    split_df[ticker_col].nunique() if ticker_col and len(split_df) > 0 else 0
                ),
                "observations_per_year": {},
            }

            if len(split_df) > 0 and ticker_col:
                split_years = pd.to_datetime(split_df[date_col]).dt.year
                for year in sorted(split_years.unique()):
                    year_df = split_df[split_years == year]
                    split_info["observations_per_year"][int(year)] = {
                        "observation_count": len(year_df),
                        "ticker_count": year_df[ticker_col].nunique(),
                    }

            fold_info[split_name] = split_info

        availability.append(fold_info)

    return availability
