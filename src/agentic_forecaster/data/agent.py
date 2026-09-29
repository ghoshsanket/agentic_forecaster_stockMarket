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
    find_file_name_map,
    load_file_name_map,
    load_ticker_frame,
    make_synthetic_dataset,
)
from agentic_forecaster.data.resampling import (
    normalize_daily_frame,
    resample_intraday_to_daily,
)
from agentic_forecaster.data.universe import Universe
from agentic_forecaster.features.engineer import build_feature_frame
from agentic_forecaster.utils import ensure_dir

logger = logging.getLogger("agentic_forecaster.data.agent")


@dataclass
class ProcessedSplit:
    X: np.ndarray
    y: np.ndarray
    dates: np.ndarray            # sequence_end_date == prediction-origin date
    target_dates: np.ndarray     # next actual trading date after the origin
    tickers: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    # Unscaled (real-world) feature values at the prediction-origin date,
    # retained verbatim for reporting.  Never reconstructed by inverse transform.
    unscaled: np.ndarray | None = None   # (n_samples, n_features) float64
    unscaled_feature_names: list[str] = field(default_factory=list)


@dataclass
class ProcessedDataset:
    train: ProcessedSplit
    val: ProcessedSplit
    test: ProcessedSplit
    scaler: StandardScaler
    feature_names: list[str]
    ticker: str = ""
    ticker_files: dict[str, Path] = field(default_factory=dict)
    # Scientific identity of the security, kept SEPARATE from the
    # filesystem-safe stem used for directories.  ``ticker`` always holds the
    # ORIGINAL requested label (e.g. "BRITISH OXYGEN (BOC)"), never a sanitised
    # filename, so reports and model ids are readable.
    requested_label: str = ""
    source_file: str = ""
    yahoo_symbol: str = ""
    historical_company_name: str = ""
    source_level: str = ""
    resampled: bool | None = None

    def save(self, root: str | Path) -> Path:
        root = ensure_dir(root)
        for name in ("train", "val", "test"):
            split = getattr(self, name)
            payload = {
                "X": split.X,
                "y": split.y,
                "dates": split.dates.astype("datetime64[D]").astype(str),
                "target_dates": split.target_dates.astype("datetime64[D]").astype(str),
                "tickers": split.tickers.astype(str),
            }
            if split.unscaled is not None and len(split.unscaled):
                payload["unscaled"] = split.unscaled.astype(np.float64)
            np.savez_compressed(root / f"{name}.npz", **payload)
        joblib.dump(self.scaler, root / "scaler.joblib")
        (root / "feature_names.txt").write_text("\n".join(self.feature_names))
        (root / "meta.json").write_text(json.dumps({
            "ticker": self.ticker,
            "requested_label": self.requested_label or self.ticker,
            "source_file": self.source_file,
            "yahoo_symbol": self.yahoo_symbol,
            "historical_company_name": self.historical_company_name,
            "source_level": self.source_level,
            "resampled": self.resampled,
        }, indent=2))
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
                unscaled=z["unscaled"] if "unscaled" in z.files else None,
            )
        scaler = joblib.load(root / "scaler.joblib")
        feature_names = (root / "feature_names.txt").read_text().splitlines()
        for split in splits.values():
            split.feature_names = feature_names
            split.unscaled_feature_names = feature_names
        meta_path = root / "meta.json"
        ticker = json.loads(meta_path.read_text()).get("ticker", "") if meta_path.exists() else ""
        return cls(
            train=splits["train"],
            val=splits["val"],
            test=splits["test"],
            scaler=scaler,
            feature_names=feature_names,
            ticker=ticker,
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
            from agentic_forecaster.data.universe import load_universe_config

            return load_universe_config(path).requested
        return None

    def universe(self) -> Universe:
        """Resolve the configured NIFTY-50 universe against the raw dataset."""
        from agentic_forecaster.data.universe import load_universe_config, resolve_universe

        ticker_path = self.data_cfg.get("tickers")
        if not ticker_path:
            return Universe()
        path = Path(ticker_path)
        if not path.is_absolute():
            path = Path(self.config.get("_config_dir", ".")) / path
        declared = load_universe_config(path)
        return resolve_universe(declared,
                                discover_ticker_files(self.data_cfg["raw_root"],
                                                      file_name_map=self._load_file_map()))

    def _file_name_map(self) -> dict[str, str]:
        """Canonical ``{requested_label: file_stem}`` map for this dataset."""
        return self._load_file_map()

    def _load_file_map(self) -> dict[str, str]:
        cached = getattr(self, "_fname_map_cache", None)
        if cached is not None:
            return cached
        raw_root = self.data_cfg.get("raw_root")
        explicit = self.data_cfg.get("file_name_map")
        if explicit:
            fmap = load_file_name_map(explicit)
        elif raw_root:
            map_path = find_file_name_map(raw_root)
            fmap = load_file_name_map(map_path) if map_path else {}
        else:
            fmap = {}
        self._fname_map_cache = fmap
        return fmap

    def _discover(self) -> dict[str, Path]:
        """Return ``{requested_label: path}`` for every available universe member.

        Keys are the ORIGINAL requested labels (e.g. ``BRITISH OXYGEN (BOC)``),
        not the filesystem-safe stems of the corresponding files.
        """
        raw_root = self.data_cfg["raw_root"]
        ticker_files = discover_ticker_files(raw_root, file_name_map=self._load_file_map())
        if not self.data_cfg.get("tickers"):
            return ticker_files
        resolved = self.universe()
        return {sym: ticker_files[sym] for sym in resolved.available.values()}

    def _label_provenance(self, label: str, path: Path) -> dict:
        """Scientific identity for a requested label, for the saved manifest.

        Filesystem-safe naming and security identity are separate concepts: the
        model may be stored under a safe directory name, but the manifest must
        record the ORIGINAL label, the source file it came from, the Yahoo
        symbol and the historical company name.
        """
        meta: dict = {
            "requested_label": label,
            "source_file": str(path),
            "yahoo_symbol": "",
            "historical_company_name": "",
        }
        lineage = self.data_cfg.get("lineage_config")
        if not lineage:
            return meta
        import yaml as _yaml
        p = Path(lineage)
        if not p.is_absolute():
            p = Path(self.config.get("_config_dir", ".")) / p
        if not p.is_file():
            return meta
        try:
            doc = _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError):
            return meta
        for entry in doc.get("securities", []):
            if str(entry.get("legacy_label")) == label:
                meta["yahoo_symbol"] = str(entry.get("primary_yahoo_candidate") or "")
                meta["historical_company_name"] = str(
                    entry.get("historical_company_name") or "")
                break
        return meta

    def _is_daily_source(self) -> bool:
        """True when the configured source is ALREADY daily bars.

        ``source_level: daily`` or ``resample_to_daily: false`` means the files
        hold one row per trading day, so the intraday resampler must be skipped.
        Anything else (notably the Kaggle minute feed) keeps the existing
        minute -> daily behaviour.
        """
        if str(self.data_cfg.get("source_level", "")).strip().lower() == "daily":
            return True
        return self.data_cfg.get("resample_to_daily") is False

    def _to_daily(self, ticker: str, path: Path, processed_root: Path) -> pd.DataFrame:
        """Route to the daily loader or the intraday resampler as appropriate."""
        if self._is_daily_source():
            return self._load_daily_source(ticker, path, processed_root)
        return self._resample_to_daily(ticker, path, processed_root)

    def _load_daily_source(self, ticker: str, path: Path,
                           processed_root: Path) -> pd.DataFrame:
        """Load an ALREADY-DAILY OHLCV file, with no intraday resampling.

        Used when the data config declares ``source_level: daily`` or
        ``resample_to_daily: false``.  The Yahoo legacy dataset is already one
        row per trading day, so calling ``resample_intraday_to_daily`` on it
        would be wrong (and would lose the vendor's exact daily open/close).

        Behaviour:
          * read the CSV,
          * normalise columns to the canonical daily schema,
          * validate that dates are one-per-trading-day, ascending, unique,
          * cache to Parquet with the same fingerprint-based invalidation used
            by the resampling path.

        The row count MUST equal the number of distinct dates in the source, so
        a test can assert that no aggregation occurred.
        """
        daily_dir = ensure_dir(processed_root / "daily")
        cache_path = daily_dir / f"{ticker}.parquet"
        meta_path = daily_dir / f"{ticker}.meta.json"
        source_fp = _fingerprint_file(path)

        if cache_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                if (meta.get("source_fingerprint") == source_fp
                        and meta.get("source_level") == "daily"):
                    logger.info("Using cached daily data for %s", ticker)
                    return pd.read_parquet(cache_path)
                logger.info("Source changed for %s; reloading daily", ticker)
            except Exception:
                logger.warning("Corrupt cache for %s; reloading daily", ticker)

        logger.info("Loading %s as already-daily OHLCV (no resampling)", ticker)
        raw = load_ticker_frame(path)
        daily = normalize_daily_frame(raw)

        n_source_dates = int(pd.to_datetime(raw[raw.columns[0]]).dt.normalize().nunique())
        if len(daily) != n_source_dates:
            raise ValueError(
                f"{ticker}: expected one row per source trading date "
                f"({n_source_dates}) but produced {len(daily)}. "
                "A daily source must not be resampled or aggregated."
            )

        daily.to_parquet(cache_path, index=False)
        meta_path.write_text(json.dumps({
            "source_fingerprint": source_fp,
            "source_file": str(path),
            "source_level": "daily",
            "resampled": False,
            "source_rows": len(raw),
            "daily_rows": len(daily),
            "start": str(daily["date"].min()),
            "end": str(daily["date"].max()),
        }))
        return daily

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

    def run_ticker(self, ticker: str, df: pd.DataFrame | None = None,
                   indicators: list[str] | None = None) -> ProcessedDataset:
        """Process a SINGLE ticker end-to-end: features, target, sequences, scaler.

        This is the primary entry point for the one-model-per-stock design.

        Parameters
        ----------
        indicators : optional override of ``features.indicators`` (used by the
            OHLCV-only ablation to pass ``[]``).
        """
        feat_cfg = self.feat_cfg
        selected = indicators if indicators is not None else feat_cfg.get("indicators")
        use_ohlcv = feat_cfg.get("use_ohlcv", True)

        # target_date[t] = next ACTUAL trading date, recorded BEFORE dropping
        # the final (unlabelled) row.  Without this the last valid sample would
        # have target_date = NaT.
        df = df.sort_values("date").reset_index(drop=True)
        nxt = pd.to_datetime(df["date"]).shift(-1)
        ff = build_feature_frame(df, indicators=selected, use_ohlcv=use_ohlcv)
        ff["target_date"] = nxt.to_numpy()
        if feat_cfg.get("drop_na", True):
            ff = ff.dropna(subset=[c for c in ff.columns if c != "target_date"]).reset_index(drop=True)

        train_start = self.data_cfg.get("train_start")
        train_end = self.data_cfg.get("train_end")
        val_start = self.data_cfg.get("val_start")
        val_end = self.data_cfg.get("val_end")
        test_start = self.data_cfg.get("test_start")
        test_end = self.data_cfg.get("test_end")
        use_date_splits = all([train_start, train_end, val_start, val_end, test_start, test_end])
        train_frac = float(self.data_cfg.get("train_fraction", 0.7))
        val_frac = float(self.data_cfg.get("val_fraction", 0.15))

        feat_cols = [c for c in ff.columns if c not in ("date", "target", "target_date")]
        values = ff[feat_cols].to_numpy(dtype=np.float64)
        target = ff["target"].to_numpy(dtype=np.float64)
        dates = ff["date"].to_numpy()
        target_dates = ff["target_date"].to_numpy()
        n = len(ff)

        buckets: dict[str, dict[str, list]] = {
            name: {"X": [], "y": [], "d": [], "td": [], "raw": []}
            for name in ("train", "val", "test")
        }

        for i in range(self.seq_len - 1, n):
            window = values[i - self.seq_len + 1 : i + 1]
            if np.isnan(window).any():
                continue
            if np.isnan(target[i]):
                continue
            origin = pd.Timestamp(dates[i])
            tdate = pd.Timestamp(target_dates[i])
            if pd.isna(tdate) or tdate <= origin:
                continue

            if use_date_splits:
                # Leakage guard: a sample belongs to a window only when BOTH
                # the origin date AND the label's target date fall inside it.
                if pd.Timestamp(train_start) <= origin <= pd.Timestamp(train_end) and \
                   pd.Timestamp(train_start) <= tdate <= pd.Timestamp(train_end):
                    name = "train"
                elif pd.Timestamp(val_start) <= origin <= pd.Timestamp(val_end) and \
                     pd.Timestamp(val_start) <= tdate <= pd.Timestamp(val_end):
                    name = "val"
                elif pd.Timestamp(test_start) <= origin <= pd.Timestamp(test_end) and \
                     pd.Timestamp(test_start) <= tdate <= pd.Timestamp(test_end):
                    name = "test"
                else:
                    continue
            else:
                n_train = int(n * train_frac)
                n_val = int(n * val_frac)
                if i < n_train:
                    name = "train"
                elif i < n_train + n_val:
                    name = "val"
                else:
                    name = "test"

            b = buckets[name]
            b["X"].append(window)
            b["y"].append(target[i])
            b["d"].append(dates[i])
            b["td"].append(target_dates[i])
            b["raw"].append(values[i])  # unscaled values AT the origin row

        def _mk(b: dict) -> ProcessedSplit:
            raw = np.asarray(b["raw"], dtype=np.float64)
            return ProcessedSplit(
                X=np.asarray(b["X"], dtype=np.float32),
                y=np.asarray(b["y"], dtype=np.float32),
                dates=np.asarray(b["d"]),
                target_dates=np.asarray(b["td"]),
                tickers=np.asarray([ticker] * len(b["X"])),
                feature_names=feat_cols,
                unscaled=raw if raw.size else None,
                unscaled_feature_names=feat_cols,
            )

        train = _mk(buckets["train"])
        val = _mk(buckets["val"])
        test = _mk(buckets["test"])

        if len(train.X) == 0:
            raise ValueError(
                f"{ticker}: no training samples in window "
                f"{train_start}..{train_end} (target dates must also fall inside)"
            )

        # One StandardScaler per ticker, fit on TRAIN rows only.
        scaler = StandardScaler()
        train.X = scaler.fit_transform(train.X.reshape(-1, train.X.shape[-1])).reshape(train.X.shape)
        if len(val.X):
            val.X = scaler.transform(val.X.reshape(-1, val.X.shape[-1])).reshape(val.X.shape)
        if len(test.X):
            test.X = scaler.transform(test.X.reshape(-1, test.X.shape[-1])).reshape(test.X.shape)

        return ProcessedDataset(
            train=train, val=val, test=test,
            scaler=scaler, feature_names=feat_cols, ticker=ticker,
        )

    def origin_snapshot(self, split: ProcessedSplit, index: int) -> dict:
        """Return the real (unscaled) feature values at one prediction origin.

        Values come from the retained unscaled snapshot, never from an
        inverse transform of the scaled data.
        """
        if split.unscaled is None or index >= len(split.unscaled):
            return {}
        row = split.unscaled[index]
        return {
            name: float(row[i])
            for i, name in enumerate(split.unscaled_feature_names)
        }

    def run(self, ticker: str | None = None,
            indicators: list[str] | None = None) -> ProcessedDataset:
        """Process one ticker.

        ``indicators`` overrides ``features.indicators`` (used by the
        OHLCV-only ablation, which passes ``[]``).
        """
        if self.data_cfg.get("synthetic"):
            frames = make_synthetic_dataset(
                n_tickers=int(self.data_cfg.get("n_tickers", 3)),
                n_days=int(self.data_cfg.get("n_days", 600)),
                seed=int(self.config.get("experiment", {}).get("seed", 42)),
            )
            if ticker:
                if ticker in frames:
                    return self.run_ticker(ticker, frames[ticker], indicators=indicators)
                raise KeyError(f"Synthetic ticker {ticker!r} not found")
            first = next(iter(frames))
            return self.run_ticker(first, frames[first], indicators=indicators)

        ticker_files = self._discover()
        if ticker:
            if ticker not in ticker_files:
                raise FileNotFoundError(
                    f"Ticker {ticker!r} not found in raw data. "
                    f"Available: {sorted(ticker_files)[:10]}..."
                )
            path = ticker_files[ticker]
            processed_root = Path(self.data_cfg.get("processed_root", "./data/processed"))
            daily = self._to_daily(ticker, path, processed_root)
            processed = self.run_ticker(ticker, daily, indicators=indicators)
            prov = self._label_provenance(ticker, path)
            processed.requested_label = prov["requested_label"]
            processed.source_file = prov["source_file"]
            processed.yahoo_symbol = prov["yahoo_symbol"]
            processed.historical_company_name = prov["historical_company_name"]
            processed.source_level = "daily" if self._is_daily_source() else "intraday"
            processed.resampled = not self._is_daily_source()
            return processed

        raise ValueError(
            "run() without a ticker is not supported in the one-model-per-stock design. "
            "Use run_ticker(ticker) or the orchestration layer."
        )
