"""PyTorch training loop with early stopping, gradient clipping, and
best-checkpoint tracking.

Uses ``BCEWithLogitsLoss`` for the single-binary-logit model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from agentic_forecaster.utils import atomic_torch_save

logger = logging.getLogger("agentic_forecaster.training")


@dataclass
class TrainingResult:
    best_epoch: int
    best_val_loss: float
    history: dict[str, list[float]] = field(default_factory=dict)
    checkpoint_path: str | None = None


def resolve_device(device: str | None = None) -> torch.device:
    """Resolve a device string to a torch.device.

    Supports: ``auto``, ``cpu``, ``cuda``, ``cuda:0``, ``cuda:1``, ...
    Respects ``CUDA_VISIBLE_DEVICES``.
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested (%s) but not available; falling back to CPU", device)
        return torch.device("cpu")
    return torch.device(device)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 3,
        max_epochs: int | None = None,
        patience: int = 10,
        restore_best_checkpoint: bool = True,
        beta1: float = 0.9,
        beta2: float = 0.999,
        gradient_clip_norm: float = 1.0,
        seed: int = 42,
        device: str | None = None,
    ):
        self.model = model
        self.lr = learning_rate
        self.wd = weight_decay
        self.batch_size = batch_size
        # `max_epochs` is an explicit alias for `epochs`.  The publication
        # specifies early-stopping patience 10 but does NOT specify an epoch
        # cap, so a performance-recovery config can set `max_epochs` without
        # the value being silently ignored.
        self.epochs = int(max_epochs) if max_epochs is not None else int(epochs)
        self.patience = patience
        self.restore_best_checkpoint = bool(restore_best_checkpoint)
        self.beta1 = beta1
        self.beta2 = beta2
        self.gradient_clip_norm = gradient_clip_norm
        self.seed = seed
        self.device = resolve_device(device)
        self.model.to(self.device)

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        checkpoint_path: str | None = None,
    ) -> TrainingResult:
        torch.manual_seed(self.seed)
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd,
            betas=(self.beta1, self.beta2),
        )

        best_val = float("inf")
        best_epoch = -1
        stale = 0
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                optimizer.step()
                train_losses.append(loss.item())

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                    val_losses.append(loss.item())

            tr = float(np.mean(train_losses))
            va = float(np.mean(val_losses))
            history["train_loss"].append(tr)
            history["val_loss"].append(va)

            if va < best_val - 1e-6:
                best_val = va
                best_epoch = epoch
                stale = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    logger.info("Early stopping at epoch %d (best=%d)", epoch, best_epoch)
                    break

        if best_state is not None and self.restore_best_checkpoint:
            self.model.load_state_dict(best_state)

        if checkpoint_path:
            atomic_torch_save(
                {
                    "model_state": self.model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                    "history": history,
                },
                checkpoint_path,
            )

        return TrainingResult(
            best_epoch=best_epoch,
            best_val_loss=best_val,
            history=history,
            checkpoint_path=checkpoint_path,
        )
