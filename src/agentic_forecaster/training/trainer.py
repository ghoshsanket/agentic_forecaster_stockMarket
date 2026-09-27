"""PyTorch training loop with early stopping and best-checkpoint tracking."""

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


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 100,
        patience: int = 15,
        class_weight: str | None = "balanced",
        seed: int = 42,
        device: str | None = None,
    ):
        self.model = model
        self.lr = learning_rate
        self.wd = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if class_weight == "balanced":
            self._class_weight = "balanced"
        else:
            self._class_weight = None

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
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

        counts = np.bincount(y_train.astype(int), minlength=2).astype(np.float64)
        weights = torch.tensor(
            counts.sum() / (2.0 * np.maximum(counts, 1.0)),
            dtype=torch.float32,
            device=self.device,
        )
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
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

        if best_state is not None:
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
