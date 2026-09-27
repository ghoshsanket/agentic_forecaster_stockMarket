import pandas as pd

_DATE_VARIANTS = ("Date", "date", "Timestamp", "timestamp", "Datetime", "datetime", "Time", "time")
_OPEN_VARIANTS = ("Open", "open", "OPEN", "o", "O")
_HIGH_VARIANTS = ("High", "high", "HIGH", "h", "H")
_LOW_VARIANTS = ("Low", "low", "LOW", "l", "L")
_CLOSE_VARIANTS = ("Close", "close", "CLOSE", "c", "C")
_VOLUME_VARIANTS = ("Volume", "volume", "VOLUME", "Vol", "vol", "v", "V")


def _find_column(df: pd.DataFrame, variants: tuple) -> str:
    for variant in variants:
        if variant in df.columns:
            return variant
    return None


def resample_intraday_to_daily(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[date_col, "open", "high", "low", "close", "volume"])

    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found in DataFrame columns: {list(df.columns)}")

    required = ("open", "high", "low", "close", "volume")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[[date_col] + list(required)].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    df["__day__"] = df[date_col].dt.date

    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    result = (
        df.groupby("__day__", sort=True)
        .agg(
            open=("open", lambda s: s.dropna().iloc[0] if s.dropna().size else None),
            high=("high", lambda s: s.dropna().max() if s.dropna().size else None),
            low=("low", lambda s: s.dropna().min() if s.dropna().size else None),
            close=("close", lambda s: s.dropna().iloc[-1] if s.dropna().size else None),
            volume=("volume", lambda s: s.dropna().sum() if s.dropna().size else 0.0),
        )
        .reset_index()
    )

    result = result.rename(columns={"__day__": date_col})
    result[date_col] = pd.to_datetime(result[date_col])

    return result


def resample_ohlcv(df: pd.DataFrame, date_col: str | None = None) -> pd.DataFrame:
    date_column = date_col if date_col is not None else _find_column(df, _DATE_VARIANTS)
    if date_column is None:
        raise ValueError(
            f"Could not auto-detect a date column. Tried: {_DATE_VARIANTS}"
        )

    rename_map = {}
    for variants in (
        _OPEN_VARIANTS,
        _HIGH_VARIANTS,
        _LOW_VARIANTS,
        _CLOSE_VARIANTS,
        _VOLUME_VARIANTS,
    ):
        col = _find_column(df, variants)
        if col is not None:
            canonical = variants[0].lower()
            rename_map[col] = canonical

    missing_canonical = {"open", "high", "low", "close", "volume"} - set(rename_map.values())
    if missing_canonical:
        raise ValueError(
            f"Could not auto-detect columns for: {sorted(missing_canonical)}"
        )

    df_renamed = df.rename(columns=rename_map)
    return resample_intraday_to_daily(df_renamed, date_col=date_column)
