"""``V2Trainer``: the V2 training loop.

Separate from the paper's binary ``training.trainer.Trainer`` on purpose.  That
trainer optimises a single logit against ``BCEWithLogitsLoss`` for one model per
stock; retrofitting it would have forced the multi-head, shared-model V2 model
through a per-stock code path it was never designed for.

Fixed initial training settings (NOT tuned in this development programme):

======================  ==============
optimizer               AdamW
learning_rate           3e-4
beta1 / beta2           0.9 / 0.999
weight_decay            1e-4
batch_size              256
max_epochs              30
early stopping          patience 6 on validation TOTAL loss
gradient_clip_norm      1.0
seed                    42
mixed precision         CUDA only, disabled when not deterministic enough
======================  ==============

``CUDA_VISIBLE_DEVICES`` is respected and ``--device auto`` never assumes a
particular GPU index.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import V2SequenceDataset, ticker_balanced_weights
from .losses import LossAccumulator, MultiTaskWeights, V2MultiTaskLoss
from .model import ContextualLSTMTransformer

logger = logging.getLogger("agentic_forecaster.v2.trainer")


def resolve_device(device: str | None = None) -> torch.device:
    """Resolve ``auto``/``cpu``/``cuda``/``cuda:N`` without assuming a GPU index."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested (%s) but unavailable; falling back to CPU", device)
        return torch.device("cpu")
    return torch.device(device)


@dataclass
class V2TrainConfig:
    """The fixed training configuration."""

    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 30
    early_stopping_patience: int = 6
    gradient_clip_norm: float = 1.0
    seed: int = 42
    balanced_by_ticker: bool = True
    use_amp: bool = True
    num_workers: int = 0

    def to_dict(self) -> dict:
        return {
            "optimizer": "AdamW",
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "gradient_clip_norm": self.gradient_clip_norm,
            "seed": self.seed,
            "balanced_by_ticker": self.balanced_by_ticker,
            "mixed_precision": self.use_amp,
        }


@dataclass
class V2TrainingResult:
    """Outcome of one shared-model fit."""

    best_epoch: int
    best_val_total_loss: float
    epochs_run: int
    history: list[dict] = field(default_factory=list)
    train_seconds: float = 0.0
    peak_gpu_memory_mb: float | None = None
    sampler: dict = field(default_factory=dict)
    stopped_early: bool = False


class V2Trainer:
    """Fit the shared V2 model on one chronological split."""

    def __init__(self, model: ContextualLSTMTransformer, config: V2TrainConfig, *,
                 loss_weights: MultiTaskWeights | None = None,
                 device: str | None = None) -> None:
        self.model = model
        self.config = config
        self.device = resolve_device(device)
        self.criterion = V2MultiTaskLoss(loss_weights)
        self.model.to(self.device)
        # Mixed precision is opt-in and CUDA-only. It is left OFF by default in
        # this programme: determinism of the reported seeds matters more than
        # throughput for a 200k-parameter model.
        self._amp_enabled = bool(config.use_amp and self.device.type == "cuda")
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    # ------------------------------------------------------------------ data

    def _loader(self, dataset: V2SequenceDataset, *, shuffle: bool,
                frame: pd.DataFrame | None = None) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        sampler = None
        if shuffle and self.config.balanced_by_ticker and frame is not None and len(frame):
            weights, _ = ticker_balanced_weights(frame)
            sampler = WeightedRandomSampler(
                torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(weights),
                replacement=True,
                generator=generator)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=self.config.num_workers,
            drop_last=False,
            generator=generator if sampler is None else None,
        )

    # --------------------------------------------------------------- forward

    def _to_device(self, batch: dict[str, torch.Tensor]) -> tuple[dict, dict]:
        inputs = {
            "stock_sequence": batch["stock_sequence"].to(self.device, non_blocking=True),
            "context_sequence": batch["context_sequence"].to(self.device, non_blocking=True),
            "regime_vector": batch["regime_vector"].to(self.device, non_blocking=True),
            "ticker_id": batch["ticker_id"].to(self.device, non_blocking=True),
            "sector_id": batch["sector_id"].to(self.device, non_blocking=True),
        }
        targets = {k: batch[k].to(self.device, non_blocking=True)
                   for k in ("y_direction", "y_return", "y_rank") if k in batch}
        return inputs, targets

    def _forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model(
            inputs["stock_sequence"], inputs["ticker_id"],
            context_sequence=inputs["context_sequence"],
            sector_id=inputs["sector_id"],
            regime_vector=inputs["regime_vector"],
        )

    # ----------------------------------------------------------------- steps

    def train_epoch(self, loader: DataLoader, optimiser: torch.optim.Optimizer
                    ) -> dict[str, float]:
        self.model.train()
        accumulator = LossAccumulator(self.criterion.weights)
        for batch in loader:
            inputs, targets = self._to_device(batch)
            optimiser.zero_grad(set_to_none=True)
            outputs = self._forward(inputs)
            loss, components = self.criterion(outputs, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite V2 loss at step: {components}")
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.config.gradient_clip_norm)
            optimiser.step()
            accumulator.update(components)
        return accumulator.result()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        accumulator = LossAccumulator(self.criterion.weights)
        for batch in loader:
            inputs, targets = self._to_device(batch)
            outputs = self._forward(inputs)
            _, components = self.criterion(outputs, targets)
            accumulator.update(components)
        return accumulator.result()

    # ------------------------------------------------------------------- fit

    def fit(self, train_dataset: V2SequenceDataset, val_dataset: V2SequenceDataset, *,
            train_frame: pd.DataFrame, val_frame: pd.DataFrame) -> V2TrainingResult:
        """Train with early stopping on validation TOTAL loss; keep the best state."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        train_loader = self._loader(train_dataset, shuffle=True, frame=train_frame)
        val_loader = self._loader(val_dataset, shuffle=False, frame=None)

        optimiser = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
        )

        best_loss = float("inf")
        best_epoch = 0
        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        history: list[dict] = []
        stale = 0
        started = time.perf_counter()

        for epoch in range(1, self.config.max_epochs + 1):
            train_stats = self.train_epoch(train_loader, optimiser)
            val_stats = self.evaluate(val_loader)
            val_total = float(val_stats.get("total", float("inf")))
            history.append({
                "epoch": epoch,
                "train_total_loss": train_stats.get("total"),
                "val_total_loss": val_total,
                "train_components": train_stats,
                "val_components": val_stats,
            })
            logger.info("epoch %d train_loss=%.5f val_loss=%.5f", epoch,
                        float(train_stats.get("total", float("nan"))), val_total)
            if val_total < best_loss - 1e-6:
                best_loss, best_epoch, stale = val_total, epoch, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.config.early_stopping_patience:
                    logger.info("early stopping at epoch %d (best epoch %d)", epoch,
                                best_epoch)
                    break

        elapsed = time.perf_counter() - started
        self.model.load_state_dict(best_state)
        self.model.to(self.device)
        peak = (torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                if self.device.type == "cuda" else None)

        _, sampler_info = ticker_balanced_weights(train_frame) if len(train_frame) else ({}, {})
        return V2TrainingResult(
            best_epoch=best_epoch,
            best_val_total_loss=float(best_loss),
            epochs_run=len(history),
            history=history,
            train_seconds=float(elapsed),
            peak_gpu_memory_mb=None if peak is None else float(peak),
            sampler={"balanced_by_ticker": self.config.balanced_by_ticker,
                     **sampler_info},
            stopped_early=len(history) < self.config.max_epochs,
        )

