import pandas as pd


def validate_ohlcv(df: pd.DataFrame, date_col: str = "date") -> list[str]:
    errors: list[str] = []

    required_cols = {"ticker", date_col, "open", "high", "low", "close", "volume"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(missing_cols)}")
        return errors

    nan_cols = [col for col in required_cols if df[col].isna().any()]
    if nan_cols:
        errors.append(f"NaN values found in columns: {sorted(nan_cols)}")

    if not df[date_col].is_monotonic_increasing:
        errors.append(f"Dates in column '{date_col}' are not sorted in chronological order")

    duplicated = df.duplicated(subset=["ticker", date_col], keep=False)
    if duplicated.any():
        dup_count = int(duplicated.sum())
        errors.append(f"Found {dup_count} duplicate ticker/date rows")

    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        non_positive = df[col] <= 0
        if non_positive.any():
            count = int(non_positive.sum())
            errors.append(f"Non-positive values found in column '{col}' ({count} rows)")

    non_negative_vol = df["volume"] < 0
    if non_negative_vol.any():
        count = int(non_negative_vol.sum())
        errors.append(f"Negative values found in column 'volume' ({count} rows)")

    for col in ["open", "close", "low"]:
        mask = df["high"] < df[col]
        if mask.any():
            count = int(mask.sum())
            errors.append(f"'high' is less than '{col}' in {count} rows")

    for col in ["open", "close"]:
        mask = df["low"] > df[col]
        if mask.any():
            count = int(mask.sum())
            errors.append(f"'low' is greater than '{col}' in {count} rows")

    today = pd.Timestamp.now().normalize()
    future_dates = df[date_col] > today
    if future_dates.any():
        count = int(future_dates.sum())
        errors.append(f"Found {count} rows with dates in the future relative to {today.date()}")

    return errors


def validate_no_leakage(train_dates, val_dates, test_dates) -> list[str]:
    errors: list[str] = []

    train_set = set(train_dates)
    val_set = set(val_dates)
    test_set = set(test_dates)

    train_val_overlap = train_set & val_set
    if train_val_overlap:
        errors.append(f"Date overlap between train and val sets: {len(train_val_overlap)} dates")

    train_test_overlap = train_set & test_set
    if train_test_overlap:
        errors.append(f"Date overlap between train and test sets: {len(train_test_overlap)} dates")

    val_test_overlap = val_set & test_set
    if val_test_overlap:
        errors.append(f"Date overlap between val and test sets: {len(val_test_overlap)} dates")

    train_max = max(train_dates) if len(train_dates) > 0 else None
    val_min = min(val_dates) if len(val_dates) > 0 else None
    val_max = max(val_dates) if len(val_dates) > 0 else None
    test_min = min(test_dates) if len(test_dates) > 0 else None

    if train_max is not None and val_min is not None and train_max >= val_min:
        errors.append(f"Train dates overlap or are not before val dates (train_max={train_max}, val_min={val_min})")

    if val_max is not None and test_min is not None and val_max >= test_min:
        errors.append(f"Val dates overlap or are not before test dates (val_max={val_max}, test_min={test_min})")

    return errors


def validate_target_alignment(dates, targets, seq_len) -> list[str]:
    errors: list[str] = []

    n = len(dates)
    if len(targets) != n:
        errors.append(f"Length mismatch: dates has {n} entries, targets has {len(targets)} entries")
        return errors

    if seq_len < 1:
        errors.append(f"seq_len must be >= 1, got {seq_len}")
        return errors

    if n < seq_len:
        errors.append(f"Not enough samples ({n}) for seq_len={seq_len}")
        return errors

    date_list = list(dates)
    target_list = list(targets)

    for i in range(seq_len - 1, n - 1):
        input_dates = date_list[i - seq_len + 1 : i + 1]
        target_date = target_list[i]
        max_input_date = max(input_dates)
        if target_date <= max_input_date:
            errors.append(
                f"Target date {target_date} is not strictly later than latest input date {max_input_date} at index {i}"
            )
            break

    return errors
