"""Shared synthetic fixtures for the V2 unit tests.

No V2 test uses the real market dataset: causality, model, meta-learning and
firewall behaviour are all properties of the CODE, and testing them on synthetic
inputs keeps them fast, deterministic and independent of the downloaded data.
The real-data behaviour is covered by the pipeline's own artefacts
(``results/v2/sanity/`` and the per-experiment manifests).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_forecaster.v2 import context as ctx
from agentic_forecaster.v2 import features as feat
from agentic_forecaster.v2.dataset import (
    FeatureArrays,
    TickerMatrices,
)
from agentic_forecaster.v2.model import V2ModelConfig
from agentic_forecaster.v2.sectors import (
    SectorMap,
)

#: Six securities in two sectors of three.  Sector groups of three (rather than
#: two) are deliberate: a leave-one-out sector dispersion needs at least three
#: available members, exactly as in the real universe, whose smallest broad
#: sector has three members.
TICKERS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
SECTORS = {"AAA": "TECH", "BBB": "TECH", "CCC": "TECH",
           "DDD": "FIN", "EEE": "FIN", "FFF": "FIN"}
#: The security that lists late; its history before this offset does not exist.
LATE_TICKER = "FFF"
LATE_OFFSET = 120
SEQUENCE_LENGTH = 8
N_STOCK_FEATURES = 6


def synthetic_bars(n_days: int = 400, seed: int = 1, *, start: str = "2018-01-01",
                   drift: float = 0.0004, vol: float = 0.015) -> pd.DataFrame:
    """Daily OHLCV from a geometric random walk, with strictly positive prices."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, vol, n_days)))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, n_days))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, n_days)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, n_days)))
    volume = rng.integers(100_000, 5_000_000, n_days).astype(float)
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low,
                         "close": close, "volume": volume})












def make_model_config(**overrides) -> V2ModelConfig:
    """A small but structurally complete V2 model config for unit tests."""
    config = V2ModelConfig(
        n_stock_features=N_STOCK_FEATURES,
        n_context_features=len(ctx.CONTEXT_FEATURES) + len(ctx.RANK_FEATURES),
        n_regime_features=len(ctx.REGIME_FEATURES),
        n_tickers=len(TICKERS) + 1,
        n_sectors=3,
        ticker_vocab=["UNKNOWN", *TICKERS],
        sector_vocab=["FIN", "TECH", "UNKNOWN"],
        hidden_size=16,
        lstm_layers=2,
        lstm_dropout=0.0,
        context_dim=8,
        d_model=8,
        fusion_temporal_dim=8,
        transformer_layers=2,
        n_heads=2,
        dim_feedforward=16,
        max_sequence_length=32,
        ticker_embedding_dim=4,
        sector_embedding_dim=4,
        regime_embedding_dim=4,
        regime_hidden=8,
        fusion_hidden=16,
        z_dim=8,
        head_hidden=8,
        use_transformer=True,
        use_context=True,
        use_sector_embedding=True,
        use_regime=True,
        use_multitask=True,
        use_film=True,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config




def make_arrays(universe_stock_features: pd.DataFrame, context: pd.DataFrame,
                sector_map: SectorMap, *, use_context: bool = True) -> FeatureArrays:
    """Per-security matrices for the synthetic universe (mirrors the real path)."""
    context_columns = (list(ctx.CONTEXT_FEATURES) + list(ctx.RANK_FEATURES)
                       if use_context else [])
    stock = universe_stock_features.loc[
        universe_stock_features["ticker"].isin(TICKERS)]
    context_subset = context.loc[:, ["ticker", "date", *context_columns]]
    merged = stock.merge(context_subset, on=["ticker", "date"], how="left",
                         validate="one_to_one")
    matrices: dict[str, TickerMatrices] = {}
    for ticker, group in merged.groupby("ticker", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        matrices[str(ticker)] = TickerMatrices(
            ticker=str(ticker),
            dates=pd.to_datetime(group["date"]).to_numpy(dtype="datetime64[ns]"),
            stock=group.loc[:, list(feat.STOCK_FEATURE_NAMES)].to_numpy(dtype=np.float32),
            context=(group.loc[:, context_columns].to_numpy(dtype=np.float32)
                     if context_columns else np.zeros((len(group), 0), dtype=np.float32)),
            regime=(group.loc[:, list(ctx.REGIME_FEATURES)].to_numpy(dtype=np.float32)
                    if use_context else np.zeros((len(group), 0), dtype=np.float32)),
            position_of={str(pd.Timestamp(d).date()): i
                         for i, d in enumerate(group["date"])},
        )
    return FeatureArrays(
        matrices=matrices,
        ticker_vocab=["UNKNOWN", *TICKERS],
        sector_vocab=["FIN", "TECH", "UNKNOWN"],
        sector_of=sector_map.sector_series(),
        stock_features=list(feat.STOCK_FEATURE_NAMES),
        context_features=context_columns,
        percentile_features=list(ctx.RANK_FEATURES) if use_context else [],
        regime_features=list(ctx.REGIME_FEATURES) if use_context else [],
    )


def make_targets(universe_stock_features: pd.DataFrame, context: pd.DataFrame,
                 arrays: FeatureArrays, *, shuffle: bool = False,
                 seed: int = 3) -> pd.DataFrame:
    """A target frame whose labels follow the features (so it is learnable)."""
    ranked = context.sort_values(["ticker", "date"]).copy()
    rows = []
    for ticker, group in ranked.groupby("ticker", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        up = (group["log_return_1"].shift(-1) > 0).astype(float)
        for i in range(len(group) - 1):
            if not np.isfinite(group["log_return_1"].iloc[i]):
                continue
            vol = float(group["realized_vol_20"].iloc[i])
            rows.append({
                "ticker": str(ticker),
                "origin_date": group["date"].iloc[i],
                "target_date": group["date"].iloc[i + 1],
                "realized_vol_20": vol,
                "raw_next_return": float(group["log_return_1"].iloc[i + 1]),
                "y_direction": int(up.iloc[i + 1]),
                "y_return": (float(up.iloc[i + 1]) if not np.isnan(vol)
                             else float("nan")),
                "y_rank": float(group["log_return_1"].rank(pct=True).iloc[i + 1]),
            })
    frame = pd.DataFrame(rows)
    if shuffle:
        rng = np.random.default_rng(seed)
        frame["y_direction"] = rng.permutation(frame["y_direction"].to_numpy())
    return frame.reset_index(drop=True)








def batch_from(dataset, indices, device="cpu"):
    """Stack dataset items into a batch dict on ``device``."""
    import torch

    items = [dataset[int(i)] for i in indices]
    return {k: torch.stack([item[k] for item in items]).to(device) for k in items[0]}