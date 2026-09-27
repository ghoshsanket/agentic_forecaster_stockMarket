"""Data Agent — first agent in the five-agent workflow.

Responsibilities:
  1. locate / download the raw dataset (Category B, outside Git);
  2. resample intraday (1-minute) OHLCV to DAILY OHLCV;
  3. engineer the Phase-1 feature set;
  4. construct the next-DAY direction target;
  5. build 30-day sequences with the prediction-origin row included;
  6. fit one StandardScaler per ticker per fold on TRAIN rows only;
  7. persist processed daily data under ``$AGENTIC_PROCESSED_DATA_ROOT/daily/``.

One-model-per-stock: the DataAgent exposes ``run_ticker(ticker)`` which
returns a ``ProcessedDataset`` for a SINGLE ticker.  The orchestration layer
calls this once per ticker, guaranteeing independent models and scalers.
"""

from __future__ import annotations

import hashlib
import json
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
from agentic_forecaster.data.resampling import resample_intraday_to_daily
from agentic_forecaster.features.engineer import build_feature_frame
from agentic_forecaster.utils import ensure_dir

logger = logging.getLogger("agentic_forecaster.data.agent")


@dataclass
class ProcessedSplit:
    X: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    target_dates: np.ndarray
    tickers: np.ndarray
    feature_names: list[str] = field(default_factory=list)


@dataclass
class ProcessedDataset:
    train: ProcessedSplit
    val: ProcessedSplit
    test: ProcessedSplit
    scaler: StandardScaler
    feature_names: list[str]
    ticker: str = ""
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
                target_dates=split.target_dates.astype("datetime64[D]").astype(str),
                tickers=split.tickers.astype(str),
            )
        joblib.dump(self.scaler, root / "scaler.joblib")
        (root / "feature_names.txt").write_text("\n".join(self.feature_names))
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
                target_dates=z["target_dates"],
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


def _fingerprint_file(path: Path) -> str:
    """Return a short fingerprint (size + mtime + sha256 of first 1MB) for cache validation."""
    stat = path.stat()
    h = hashlib.sha256()
    h.update(str(stat.st_size).encode())
    h.update(str(stat.st_mtime).encode())
    with open(path, "rb") as f:
        h.update(f.read(1_000_000))
    return h.hexdigest()[:16]


class DataAgent:
    """Orchestrates per-ticker dataset loading, resampling, feature engineering
    and sequencing for the one-model-per-stock design."""

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

    def _discover(self) -> dict[str, Path]:
        raw_root = self.data_cfg["raw_root"]
        ticker_files = discover_ticker_files(raw_root)
        selected = self._load_ticker_config()
        if selected:
            selected_upper = {s.upper() for s in selected}
            ticker_files = {k: v for k, v in ticker_files.items() if k in selected_upper}
        return ticker_files

    def _resample_to_daily(self, ticker: str, path: Path, processed_root: Path) -> pd.DataFrame:
        """Resample intraday CSV to daily OHLCV with caching and source-hash validation."""
        daily_dir = ensure_dir(processed_root / "daily")
        cache_path = daily_dir / f"{ticker}.parquet"
        meta_path = daily_dir / f"{ticker}.meta.json"
        source_fp = _fingerprint_file(path)

        if cache_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                if meta.get("source_fingerprint") == source_fp:
                    logger.info("Using cached daily data for %s", ticker)
                    return pd.read_parquet(cache_path)
                logger.info("Source changed for %s; re-resampling", ticker)
            except Exception:
                logger.warning("Corrupt cache for %s; re-resampling", ticker)

        logger.info("Resampling %s intraday -> daily", ticker)
        intraday = load_ticker_frame(path)
        daily = resample_intraday_to_daily(intraday, date_col="date")
        daily["ticker"] = ticker

        daily.to_parquet(cache_path, index=False)
        meta_path.write_text(json.dumps({
            "source_fingerprint": source_fp,
            "source_file": str(path),
            "source_rows": len(intraday),
            "daily_rows": len(daily),
            "start": str(daily["date"].min()),
            "end": str(daily["date"].max()),
        }))
        return daily

    def run_ticker(self, ticker: str, df: pd.DataFrame | None = None) -> ProcessedDataset:
        """Process a SINGLE ticker end-to-end: features, target, sequences, scaler.

        This is the primary entry point for the one-model-per-stock design.
        """
        feat_cfg = self.feat_cfg
        indicators = feat_cfg.get("indicators")
        use_ohlcv = feat_cfg.get("use_ohlcv", True)

        ff = build_feature_frame(df, indicators=indicators, use_ohlcv=use_ohlcv)
        if feat_cfg.get("drop_na", True):
            ff = ff.dropna().reset_index(drop=True)

        train_start = self.data_cfg.get("train_start")
        train_end = self.data_cfg.get("train_end")
        val_start = self.data_cfg.get("val_start")
        val_end = self.data_cfg.get("val_end")
        test_start = self.data_cfg.get("test_start")
        test_end = self.data_cfg.get("test_end")
        use_date_splits = all([train_start, train_end, val_start, val_end, test_start, test_end])
        train_frac = float(self.data_cfg.get("train_fraction", 0.7))
        val_frac = float(self.data_cfg.get("val_fraction", 0.15))

        train_X, train_y, train_dates, train_tdates = [], [], [], []
        val_X, val_y, val_dates, val_tdates = [], [], [], []
        test_X, test_y, test_dates, test_tdates = [], [], [], []

        feat_cols = [c for c in ff.columns if c not in ("date", "target")]
        values = ff[feat_cols].to_numpy(dtype=np.float64)
        target = ff["target"].to_numpy(dtype=np.float64)
        dates = ff["date"].to_numpy()

        n = len(ff)

        for i in range(self.seq_len - 1, n):
            window = values[i - self.seq_len + 1 : i + 1]
            if np.isnan(window).any():
                continue
            if np.isnan(target[i]):
                continue
            d = pd.Timestamp(dates[i])
            td = pd.Timestamp(dates[i + 1]) if i + 1 < n else pd.NaT

            if use_date_splits:
                if pd.Timestamp(train_start) <= d <= pd.Timestamp(train_end):
                    train_X.append(window)
                    train_y.append(target[i])
                    train_dates.append(d)
                    train_tdates.append(td)
                elif pd.Timestamp(val_start) <= d <= pd.Timestamp(val_end):
                    val_X.append(window)
                    val_y.append(target[i])
                    val_dates.append(d)
                    val_tdates.append(td)
                elif pd.Timestamp(test_start) <= d <= pd.Timestamp(test_end):
                    test_X.append(window)
                    test_y.append(target[i])
                    test_dates.append(d)
                    test_tdates.append(td)
            else:
                n_train = int(n * train_frac)
                n_val = int(n * val_frac)
                if i < n_train:
                    train_X.append(window)
                    train_y.append(target[i])
                    train_dates.append(d)
                    train_tdates.append(td)
                elif i < n_train + n_val:
                    val_X.append(window)
                    val_y.append(target[i])
                    val_dates.append(d)
                    val_tdates.append(td)
                else:
                    test_X.append(window)
                    test_y.append(target[i])
                    test_dates.append(d)
                    test_tdates.append(td)

        def _mk(X, y, d, td) -> ProcessedSplit:
            return ProcessedSplit(
                X=np.asarray(X, dtype=np.float32),
                y=np.asarray(y, dtype=np.float32),
                dates=np.asarray(d),
                target_dates=np.asarray(td),
                tickers=np.asarray([ticker] * len(X)),
                feature_names=feat_cols,
            )

        train = _mk(train_X, train_y, train_dates, train_tdates)
        val = _mk(val_X, val_y, val_dates, val_tdates)
        test = _mk(test_X, test_y, test_dates, test_tdates)

        scaler = StandardScaler()
        train.X = scaler.fit_transform(train.X.reshape(-1, train.X.shape[-1])).reshape(train.X.shape)
        val.X = scaler.transform(val.X.reshape(-1, val.X.shape[-1])).reshape(val.X.shape)
        test.X = scaler.transform(test.X.reshape(-1, test.X.shape[-1])).reshape(test.X.shape)

        return ProcessedDataset(
            train=train, val=val, test=test,
            scaler=scaler, feature_names=feat_cols, ticker=ticker,
        )

    def run(self, ticker: str | None = None) -> ProcessedDataset:
        """Process one ticker (or all tickers for backward compat / demo)."""
        if self.data_cfg.get("synthetic"):
            frames = make_synthetic_dataset(
                n_tickers=int(self.data_cfg.get("n_tickers", 3)),
                n_days=int(self.data_cfg.get("n_days", 600)),
                seed=int(self.config.get("experiment", {}).get("seed", 42)),
            )
            if ticker:
                if ticker in frames:
                    return self.run_ticker(ticker, frames[ticker])
                raise KeyError(f"Synthetic ticker {ticker!r} not found")
            first = next(iter(frames))
            return self.run_ticker(first, frames[first])

        ticker_files = self._discover()
        if ticker:
            if ticker not in ticker_files:
                raise FileNotFoundError(
                    f"Ticker {ticker!r} not found in raw data. "
                    f"Available: {sorted(ticker_files)[:10]}..."
                )
            path = ticker_files[ticker]
            processed_root = Path(self.data_cfg.get("processed_root", "./data/processed"))
            daily = self._resample_to_daily(ticker, path, processed_root)
            return self.run_ticker(ticker, daily)

        raise ValueError(
            "run() without a ticker is not supported in the one-model-per-stock design. "
            "Use run_ticker(ticker) or the orchestration layer."
        )
