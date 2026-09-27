"""Data Agent — first agent in the five-agent workflow.

Responsibilities:
  1. locate / download the raw dataset (Category B, outside Git);
  2. engineer features (RSI, MACD, ATR, volatility, ...);
  3. construct the target (next-day direction);
  4. build 30-day sequences;
  5. fit the StandardScaler on the training split only (no leakage);
  6. persist the processed dataset under ``$AGENTIC_PROCESSED_DATA_ROOT``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from agentic_forecaster.data.dataset import (
    discover_ticker_files,
    load_ticker_frame,
    make_synthetic_dataset,
)
from agentic_forecaster.features.engineer import build_feature_frame
from agentic_forecaster.utils import ensure_dir

logger = logging.getLogger("agentic_forecaster.data.agent")


@dataclass
class ProcessedSplit:
    X: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    tickers: np.ndarray
    feature_names: list[str] = field(default_factory=list)


@dataclass
class ProcessedDataset:
    train: ProcessedSplit
    val: ProcessedSplit
    test: ProcessedSplit
    scaler: StandardScaler
    feature_names: list[str]
    ticker_files: dict[str, Path] = field(default_factory=dict)

    def save(self, root: str | Path) -> Path:
        root = ensure_dir(root)
        for name in ("train", "val", "test"):
            split = getattr(self, name)
            np.savez_compressed(
                root / f"{name}.npz",
                X=split.X,
                y=split.y,
                dates=split.dates.astype("datetime64[D]").astype(str),
                tickers=split.tickers.astype(str),
            )
        joblib.dump(self.scaler, root / "scaler.joblib")
        (root / "feature_names.txt").write_text("\n".join(self.feature_names))
        logger.info("Processed dataset saved to %s", root)
        return root

    @classmethod
    def load(cls, root: str | Path) -> ProcessedDataset:
        root = Path(root)
        splits = {}
        for name in ("train", "val", "test"):
            z = np.load(root / f"{name}.npz", allow_pickle=False)
            splits[name] = ProcessedSplit(
                X=z["X"],
                y=z["y"],
                dates=z["dates"],
                tickers=z["tickers"],
            )
        scaler = joblib.load(root / "scaler.joblib")
        feature_names = (root / "feature_names.txt").read_text().splitlines()
        for name, split in splits.items():
            split.feature_names = feature_names
        return cls(
            train=splits["train"],
            val=splits["val"],
            test=splits["test"],
            scaler=scaler,
            feature_names=feature_names,
        )


class DataAgent:
    """Orchestrates dataset loading, feature engineering and sequencing."""

    def __init__(self, config: dict):
        self.config = config
        self.data_cfg = config["data"]
        self.feat_cfg = config.get("features", {})
        self.seq_len = int(self.data_cfg.get("sequence_length", 30))

    def _load_ticker_config(self) -> list[str] | None:
        ticker_path = self.data_cfg.get("tickers")
        if not ticker_path:
            return None
        path = Path(ticker_path)
        if not path.is_absolute():
            path = Path(self.config.get("_config_dir", ".")) / path
        if path.exists():
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
            return data.get("tickers", [])
        return None

    def run(self) -> ProcessedDataset:
        if self.data_cfg.get("synthetic"):
            frames = make_synthetic_dataset(
                n_tickers=int(self.data_cfg.get("n_tickers", 3)),
                n_days=int(self.data_cfg.get("n_days", 600)),
                seed=int(self.config.get("experiment", {}).get("seed", 42)),
            )
            ticker_files = {}
        else:
            raw_root = self.data_cfg["raw_root"]
            ticker_files = discover_ticker_files(raw_root)
            selected = self._load_ticker_config()
            if selected:
                ticker_files = {k: v for k, v in ticker_files.items() if k in selected}
                missing = set(selected) - set(ticker_files.keys())
                if missing:
                    logger.warning("Tickers in config but not found in raw data: %s", sorted(missing))
            frames = {}
            for ticker, path in ticker_files.items():
                df = load_ticker_frame(path)
                frames[ticker] = df

        feature_frames: dict[str, pd.DataFrame] = {}
        for ticker, df in frames.items():
            ff = build_feature_frame(df, indicators=self.feat_cfg.get("indicators"))
            if self.feat_cfg.get("drop_na", True):
                ff = ff.dropna().reset_index(drop=True)
            feature_frames[ticker] = ff

        train_frac = float(self.data_cfg.get("train_fraction", 0.7))
        val_frac = float(self.data_cfg.get("val_fraction", 0.15))
        train_start = self.data_cfg.get("train_start")
        train_end = self.data_cfg.get("train_end")
        val_start = self.data_cfg.get("val_start")
        val_end = self.data_cfg.get("val_end")
        test_start = self.data_cfg.get("test_start")
        test_end = self.data_cfg.get("test_end")
        use_date_splits = all([train_start, train_end, val_start, val_end, test_start, test_end])

        train_X, train_y, train_dates, train_tickers = [], [], [], []
        val_X, val_y, val_dates, val_tickers = [], [], [], []
        test_X, test_y, test_dates, test_tickers = [], [], [], []

        for ticker, ff in feature_frames.items():
            n = len(ff)
            feat_cols = [c for c in ff.columns if c not in ("date", "target")]
            values = ff[feat_cols].to_numpy(dtype=np.float64)
            target = ff["target"].to_numpy(dtype=np.int64)
            dates = ff["date"].to_numpy()

            if use_date_splits:
                for i in range(self.seq_len, n):
                    window = values[i - self.seq_len : i]
                    if np.isnan(window).any():
                        continue
                    d = pd.Timestamp(dates[i])
                    if pd.Timestamp(train_start) <= d <= pd.Timestamp(train_end):
                        train_X.append(window)
                        train_y.append(target[i])
                        train_dates.append(dates[i])
                        train_tickers.append(ticker)
                    elif pd.Timestamp(val_start) <= d <= pd.Timestamp(val_end):
                        val_X.append(window)
                        val_y.append(target[i])
                        val_dates.append(dates[i])
                        val_tickers.append(ticker)
                    elif pd.Timestamp(test_start) <= d <= pd.Timestamp(test_end):
                        test_X.append(window)
                        test_y.append(target[i])
                        test_dates.append(dates[i])
                        test_tickers.append(ticker)
            else:
                n_train = int(n * train_frac)
                n_val = int(n * val_frac)
                for i in range(self.seq_len, n):
                    window = values[i - self.seq_len : i]
                    if np.isnan(window).any():
                        continue
                    if i < n_train:
                        train_X.append(window)
                        train_y.append(target[i])
                        train_dates.append(dates[i])
                        train_tickers.append(ticker)
                    elif i < n_train + n_val:
                        val_X.append(window)
                        val_y.append(target[i])
                        val_dates.append(dates[i])
                        val_tickers.append(ticker)
                    else:
                        test_X.append(window)
                        test_y.append(target[i])
                        test_dates.append(dates[i])
                        test_tickers.append(ticker)

        def _mk(X, y, d, t) -> ProcessedSplit:
            return ProcessedSplit(
                X=np.asarray(X, dtype=np.float32),
                y=np.asarray(y, dtype=np.int64),
                dates=np.asarray(d),
                tickers=np.asarray(t),
                feature_names=feat_cols,
            )

        train = _mk(train_X, train_y, train_dates, train_tickers)
        val = _mk(val_X, val_y, val_dates, val_tickers)
        test = _mk(test_X, test_y, test_dates, test_tickers)

        scaler = StandardScaler()
        train.X = scaler.fit_transform(train.X.reshape(-1, train.X.shape[-1])).reshape(
            train.X.shape
        )
        val.X = scaler.transform(val.X.reshape(-1, val.X.shape[-1])).reshape(val.X.shape)
        test.X = scaler.transform(test.X.reshape(-1, test.X.shape[-1])).reshape(test.X.shape)

        logger.info(
            "Processed dataset: train=%d val=%d test=%d features=%d seq_len=%d",
            len(train.y), len(val.y), len(test.y), len(feat_cols), self.seq_len,
        )
        return ProcessedDataset(
            train=train, val=val, test=test,
            scaler=scaler, feature_names=feat_cols,
            ticker_files=ticker_files,
        )
