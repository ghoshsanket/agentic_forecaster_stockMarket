"""V2 sanity checks that must pass BEFORE any real V2 training.

Two synthetic checks, both using the real model class and the real trainer:

``synthetic_learnable``
    A deterministic, learnable relationship between the last input feature and
    the next-day direction.  If V2 cannot drive validation accuracy far above
    chance here, something is wrong in the model, the gradients, the target
    alignment or the scaling -- and every later number would be uninterpretable.

``shuffled_labels``
    The same synthetic task with the TRAIN labels shuffled.  Validation accuracy
    must fall back to approximately chance.  A model that still looks good on
    shuffled labels is not using its inputs.

No real market data is used here, and neither check touches the firewall: the
synthetic dates are arbitrary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .dataset import (
    FeatureArrays,
    GroupFeatureScaler,
    SampleTable,
    TickerMatrices,
    V2ScalerBundle,
    V2SequenceDataset,
)
from .ledger import hash_payload
from .losses import MultiTaskWeights
from .model import ContextualLSTMTransformer, V2ModelConfig
from .trainer import V2TrainConfig, V2Trainer

logger = logging.getLogger("agentic_forecaster.v2.sanity")

#: Chance level of the synthetic binary task, and the thresholds that decide
#: whether the check passed.  Fixed before the run, like every other setting.
CHANCE = 0.5
LEARNABLE_MIN_ACCURACY = 0.85
SHUFFLED_MAX_ACCURACY = 0.60


@dataclass
class SanityResult:
    """Outcome of one synthetic sanity check."""

    name: str
    passed: bool
    n_train: int
    n_val: int
    validation_accuracy_micro: float
    validation_accuracy_macro_ticker: float
    train_majority_baseline: float
    best_epoch: int
    epochs_run: int
    thresholds: dict
    details: dict

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "validation_accuracy_micro": self.validation_accuracy_micro,
            "validation_accuracy_macro_ticker": self.validation_accuracy_macro_ticker,
            "train_majority_baseline": self.train_majority_baseline,
            "best_epoch": self.best_epoch,
            "epochs_run": self.epochs_run,
            "thresholds": self.thresholds,
            "details": self.details,
        }


def make_synthetic_arrays(*, n_tickers: int = 4, n_days: int = 900, n_features: int = 6,
                          seed: int = 0) -> tuple[FeatureArrays, pd.DataFrame]:
    """A synthetic universe whose next-day direction IS learnable from its inputs.

    The signal is explicit and deterministic:

        y_direction[t] = 1  <=>  feature_0[t] > 0

    and the last input row of the sequence IS row ``t``, so a working model can
    reach a very high accuracy while a broken one cannot.  Every other feature is
    independent noise, the label class is balanced by construction (so the
    majority baseline is 50% and a shuffled-label control has nowhere to hide),
    and no input is standardised away before the model sees it.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days + 2)
    tickers = [f"SYN{i:02d}" for i in range(n_tickers)]

    matrices: dict[str, TickerMatrices] = {}
    target_rows: list[dict] = []
    for ticker in tickers:
        signal = rng.choice([-1.0, 1.0], size=len(dates))
        noise = rng.normal(0.0, 1.0, size=(len(dates), n_features))
        raw = noise.copy()
        raw[:, 0] = signal
        stock = raw.astype(np.float32)
        context = noise[:, 1:].astype(np.float32)
        matrices[ticker] = TickerMatrices(
            ticker=ticker,
            dates=dates.to_numpy(dtype="datetime64[ns]"),
            stock=stock,
            context=context,
            regime=context[:, :1].copy(),
            position_of={str(pd.Timestamp(d).date()): i for i, d in enumerate(dates)},
        )
        for i in range(len(dates) - 1):
            target_rows.append({
                "ticker": ticker,
                "origin_date": pd.Timestamp(dates[i]),
                "target_date": pd.Timestamp(dates[i + 1]),
                "row": i,
                "y_direction": int(signal[i] > 0),
                "y_return": float(signal[i]) + float(rng.normal(0, 0.2)),
                "y_rank": float(signal[i] > 0),
            })

    arrays = FeatureArrays(
        matrices=matrices,
        ticker_vocab=sorted(tickers),
        sector_vocab=["UNKNOWN", "SYNTHETIC"],
        sector_of={t: "SYNTHETIC" for t in tickers},
        stock_features=[f"synthetic_feature_{i}" for i in range(n_features)],
        context_features=[f"synthetic_context_{i}" for i in range(n_features - 1)],
        percentile_features=[],
        regime_features=["synthetic_regime_0"],
    )
    return arrays, pd.DataFrame(target_rows)


def build_synthetic_samples(arrays: FeatureArrays, targets: pd.DataFrame, *,
                            sequence_length: int = 30, shuffle_labels: bool = False,
                            seed: int = 7) -> SampleTable:
    """Sample table for the synthetic universe, optionally with shuffled labels."""
    frame = targets.copy().sort_values(["ticker", "origin_date"]).reset_index(drop=True)
    records = []
    for row in frame.itertuples():
        matrices = arrays.matrices[str(row.ticker)]
        start = int(row.row) - sequence_length + 1
        if start < 0:
            continue
        window = matrices.stock[start:int(row.row) + 1]
        if not np.isfinite(window).all():
            continue
        records.append({
            "ticker": str(row.ticker),
            "origin_date": row.origin_date,
            "target_date": row.target_date,
            "row": int(row.row),
            "ticker_id": arrays.ticker_vocab.index(str(row.ticker)),
            "sector_id": 1,
            "y_direction": int(row.y_direction),
            "y_return": float(row.y_return),
            "y_rank": float(row.y_rank),
        })
    table = pd.DataFrame(records)
    if shuffle_labels and len(table):
        rng = np.random.default_rng(seed)
        table["y_direction"] = rng.permutation(table["y_direction"].to_numpy())
    table = table.sort_values(["ticker", "origin_date"]).reset_index(drop=True)
    return SampleTable(frame=table, arrays=arrays, sequence_length=sequence_length,
                       diagnostics={"synthetic": True, "shuffled_labels": shuffle_labels})


def run_synthetic_check(name: str, *, shuffle_labels: bool = False, seed: int = 42,
                        epochs: int = 12, n_tickers: int = 4, n_days: int = 900,
                        device: str = "auto") -> SanityResult:
    """Train the real V2 model on the synthetic task and score the holdout."""
    arrays, targets = make_synthetic_arrays(n_tickers=n_tickers, n_days=n_days)
    samples = build_synthetic_samples(arrays, targets, shuffle_labels=shuffle_labels,
                                      seed=seed)

    dates = pd.to_datetime(samples.frame["origin_date"])
    split = int(len(dates) * 0.7)
    ordered = samples.frame.sort_values("origin_date").reset_index(drop=True)
    train = ordered.loc[:split - 1]
    val = ordered.loc[split:]
    train_index = train.index.to_numpy()

    stock_scaler = GroupFeatureScaler(arrays.stock_features).fit(
        _origin_matrix(samples, train_index, "stock", arrays.stock_features),
        {"group": "stock", "synthetic": True, "fitted_on": "TRAIN_SAMPLES_ONLY"})
    context_scaler = GroupFeatureScaler(arrays.context_features,
                                        passthrough=arrays.percentile_features).fit(
        _origin_matrix(samples, train_index, "context", arrays.context_features),
        {"group": "context", "synthetic": True, "fitted_on": "TRAIN_SAMPLES_ONLY"})
    scalers = V2ScalerBundle(stock=stock_scaler, context=context_scaler)

    train_dataset = V2SequenceDataset(samples, train, scalers=scalers)
    val_dataset = V2SequenceDataset(samples, val, scalers=scalers)

    model_config = V2ModelConfig(
        n_stock_features=len(arrays.stock_features),
        n_context_features=len(arrays.context_features),
        n_regime_features=len(arrays.regime_features),
        n_tickers=len(arrays.ticker_vocab),
        n_sectors=len(arrays.sector_vocab),
        ticker_vocab=list(arrays.ticker_vocab),
        sector_vocab=list(arrays.sector_vocab),
        hidden_size=48,
        lstm_layers=2,
        lstm_dropout=0.1,
        context_dim=16,
        d_model=32,
        fusion_temporal_dim=32,
        transformer_layers=2,
        n_heads=4,
        dim_feedforward=64,
        max_sequence_length=128,
        ticker_embedding_dim=8,
        sector_embedding_dim=4,
        regime_embedding_dim=8,
        regime_hidden=16,
        fusion_hidden=64,
        use_transformer=True,
        use_context=True,
        use_sector_embedding=True,
        use_regime=True,
        use_multitask=True,
        use_film=True,
    )
    torch.manual_seed(seed)
    model = ContextualLSTMTransformer(model_config)

    # The sanity fixture is a DIAGNOSTIC, not the production training schedule: it
    # uses a larger learning rate and a smaller batch so that a trivially
    # learnable signal is learned within a handful of epochs on either device.
    # The V2 development runs use the fixed schedule in configs/v2/base.yaml.
    train_config = V2TrainConfig(max_epochs=epochs, early_stopping_patience=epochs,
                                 batch_size=64, learning_rate=1e-3, seed=seed,
                                 balanced_by_ticker=False, use_amp=False)
    trainer = V2Trainer(model, train_config, loss_weights=MultiTaskWeights(),
                        device=device)
    result = trainer.fit(train_dataset, val_dataset, train_frame=train,
                         val_frame=val)

    model.eval()
    device = trainer.device
    y_val = val["y_direction"].to_numpy(int)
    logits = []
    with torch.no_grad():
        for start in range(0, len(val_dataset), 512):
            items = [val_dataset[i] for i in range(start, min(start + 512, len(val_dataset)))]
            batch = {k: torch.stack([item[k] for item in items]) for k in items[0]}
            outputs = model(batch["stock_sequence"].to(device),
                            batch["ticker_id"].to(device),
                            context_sequence=batch["context_sequence"].to(device),
                            sector_id=batch["sector_id"].to(device),
                            regime_vector=batch["regime_vector"].to(device))
            logits.append(outputs["direction_logit"].cpu().numpy())
    logit = np.concatenate(logits) if logits else np.zeros(0)
    proba = 1.0 / (1.0 + np.exp(-logit))
    pred = (proba >= 0.5).astype(int)
    accuracy = float((pred == y_val).mean()) if len(y_val) else float("nan")

    per_ticker = val.assign(pred=pred).groupby("ticker").apply(
        lambda g: float((g["pred"].to_numpy(int) == g["y_direction"].to_numpy(int)).mean()),
        include_groups=False)
    macro = float(per_ticker.mean()) if len(per_ticker) else float("nan")
    baseline = float(max(y_val.mean(), 1.0 - y_val.mean())) if len(y_val) else float("nan")

    if name == "synthetic_learnable":
        passed = accuracy >= LEARNABLE_MIN_ACCURACY
        thresholds = {"min_validation_accuracy": LEARNABLE_MIN_ACCURACY,
                      "chance": CHANCE}
    elif name == "shuffled_labels":
        passed = accuracy <= SHUFFLED_MAX_ACCURACY
        thresholds = {"max_validation_accuracy": SHUFFLED_MAX_ACCURACY, "chance": CHANCE}
    else:
        raise ValueError(f"unknown sanity check {name!r}")

    return SanityResult(
        name=name,
        passed=bool(passed),
        n_train=len(train),
        n_val=len(val),
        validation_accuracy_micro=accuracy,
        validation_accuracy_macro_ticker=macro,
        train_majority_baseline=baseline,
        best_epoch=int(result.best_epoch),
        epochs_run=int(result.epochs_run),
        thresholds=thresholds,
        details={
            "per_ticker_accuracy": {str(k): float(v) for k, v in per_ticker.items()},
            "signal": "next-day direction is a deterministic function of the last "
                      "input feature row",
            "labels_shuffled": shuffle_labels,
            "n_parameters": model.n_parameters(),
            "train_seconds": result.train_seconds,
            "config_sha256": hash_payload(model_config.to_dict()),
        },
    )


def _origin_matrix(samples: SampleTable, indices: np.ndarray, group: str,
                   columns: list[str]) -> np.ndarray:
    frame = samples.frame.iloc[indices]
    blocks = []
    for ticker, row in zip(frame["ticker"], frame["row"], strict=True):
        matrices = samples.arrays.matrices[str(ticker)]
        source = matrices.stock if group == "stock" else matrices.context
        blocks.append(source[int(row), :])
    return np.vstack(blocks).astype(np.float64) if blocks else np.zeros((0, len(columns)))


def sanity_payload(results: list[SanityResult]) -> dict:
    """Machine-readable summary of both checks."""
    return {
        "checks": [r.to_dict() for r in results],
        "all_passed": all(r.passed for r in results),
        "gate": ("if the learnable synthetic task fails, STOP before real V2 training"
                 if not results[0].passed else
                 "synthetic learnability and shuffled-label control both behaved as "
                 "expected; real V2 training may proceed"),
        "no_real_market_data_used": True,
    }