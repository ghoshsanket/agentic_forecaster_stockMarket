from __future__ import annotations

import numpy as np
import pandas as pd


def build_sequences(
    df: pd.DataFrame,
    seq_length: int = 30,
    feature_cols: list[str] | None = None,
    target_col: str = "target",
    date_col: str = "date",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in (target_col, date_col, "ticker")]

    values = df[feature_cols].to_numpy(dtype=np.float64)
    target = df[target_col].to_numpy(dtype=np.int64)
    dates = df[date_col].to_numpy()

    X, y, seq_dates = [], [], []
    for i in range(seq_length, len(df)):
        window = values[i - seq_length : i]
        if np.isnan(window).any():
            continue
        X.append(window)
        y.append(target[i])
        seq_dates.append(dates[i])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(seq_dates),
    )


def build_sequences_with_tickers(
    df: pd.DataFrame,
    seq_length: int = 30,
    feature_cols: list[str] | None = None,
    target_col: str = "target",
    date_col: str = "date",
    ticker_col: str= "ticker",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in (target_col, date_col, ticker_col)]

    values = df[feature_cols].to_numpy(dtype=np.float64)
    target = df[target_col].to_numpy(dtype=np.int64)
    dates = df[date_col].to_numpy()
    tickers = df[ticker_col].to_numpy() if ticker_col in df.columns else np.array([""] * len(df))

    X, y, seq_dates, seq_tickers = [], [], [], []
    for i in range(seq_length, len(df)):
        window = values[i - seq_length : i]
        if np.isnan(window).any():
            continue
        X.append(window)
        y.append(target[i])
        seq_dates.append(dates[i])
        seq_tickers.append(tickers[i])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(seq_dates),
        np.asarray(seq_tickers),
    )
