"""Model Agent — trains ONE independent Attention-LSTM per stock.

One-model-per-stock design (PAPER-DEFINED):
    for each selected ticker:
        load that ticker's daily data
        fit that ticker's own StandardScaler on TRAIN rows only
        build that ticker's sequences
        instantiate a fresh AttentionLSTM
        train it (max 3 epochs)
        fit temperature calibration on that ticker's validation data
        evaluate on that ticker's test data
        save model.pt, scaler.joblib, calibration.json, config.json,
              metrics.json, manifest.json

Runtime artifact layout:
    $AGENTIC_MODEL_ROOT/trained/<ticker>/<experiment-or-fold>/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import torch

from agentic_forecaster.calibration import TemperatureCalibrator
from agentic_forecaster.models import AttentionLSTM
from agentic_forecaster.training import Trainer
from agentic_forecaster.utils import atomic_json_dump, atomic_torch_save, ensure_dir

logger = logging.getLogger("agentic_forecaster.agents.model")


@dataclass
class FittedModel:
    name: str
    kind: str  # 'torch' | 'sklearn' | 'majority'
    model: object
    ticker: str = ""
    fold: str = ""
    metrics: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    temperature: float = 1.0
    checkpoint_path: str | None = None
    scaler_path: str | None = None
    feature_names: list[str] = field(default_factory=list)
    train_config: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)
    seed: int = 42

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return CALIBRATED p_up probabilities."""
        if self.kind == "torch":
            self.model.eval()
            device = next(self.model.parameters()).device
            with torch.no_grad():
                logits = self.model(torch.tensor(X, dtype=torch.float32, device=device))
                p_raw = torch.sigmoid(logits).cpu().numpy()
            cal = TemperatureCalibrator()
            cal.temperature = self.temperature
            return cal.calibrate_proba(p_raw)
        return self.model.predict_proba(X)

    def predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        """Return RAW (uncalibrated) p_up probabilities."""
        if self.kind == "torch":
            self.model.eval()
            device = next(self.model.parameters()).device
            with torch.no_grad():
                logits = self.model(torch.tensor(X, dtype=torch.float32, device=device))
                return torch.sigmoid(logits).cpu().numpy()
        return self.model.predict_proba(X)

    def save(self, root: str | Path) -> Path:
        root = ensure_dir(root)
        if self.kind == "torch":
            atomic_torch_save(self.model.state_dict(), root / "model.pt")
        elif self.kind != "majority":
            joblib.dump(self.model, root / "model.joblib")
        joblib.dump(self.scaler, root / "scaler.joblib")
        atomic_json_dump(self.metrics, root / "metrics.json")
        atomic_json_dump(self.calibration, root / "calibration.json")
        atomic_json_dump(self.train_config, root / "config.json")
        atomic_json_dump(
            {
                "name": self.name,
                "kind": self.kind,
                "seed": self.seed,
                "ticker": self.ticker,
                "fold": self.fold,
                "temperature": self.temperature,
            },
            root / "manifest.json",
        )
        if self.feature_names:
            (root / "feature_names.txt").write_text("\n".join(self.feature_names))
        return root

    @property
    def scaler(self):
        """Return the fitted scaler (stored on the object after training)."""
        return getattr(self, "_scaler", None)

    @scaler.setter
    def scaler(self, value):
        self._scaler = value


class ModelAgent:
    """Trains one independent model per ticker."""

    def __init__(self, config: dict):
        self.config = config
        self.model_cfg = config.get("models", {})
        self.cal_cfg = config.get("calibration", {})
        self.seed = int(config.get("experiment", {}).get("seed", 42))

    def train_ticker(self, ticker: str, dataset, fold: str = "fold_0",
                     device: str | None = None) -> FittedModel:
        """Train a single ticker's model. Returns the fitted model."""
        cfg = self.model_cfg["attention_lstm"]
        model = AttentionLSTM(
            input_size=dataset.train.X.shape[-1],
            hidden_size=int(cfg.get("hidden_size", 64)),
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.2)),
        )
        trainer = Trainer(
            model,
            learning_rate=float(cfg.get("learning_rate", 1e-3)),
            weight_decay=float(cfg.get("weight_decay", 1e-4)),
            batch_size=int(cfg.get("batch_size", 64)),
            epochs=int(cfg.get("epochs", 3)),
            patience=int(cfg.get("patience", 10)),
            beta1=float(cfg.get("beta1", 0.9)),
            beta2=float(cfg.get("beta2", 0.999)),
            gradient_clip_norm=float(cfg.get("gradient_clip_norm", 1.0)),
            seed=self.seed,
            device=device,
        )
        result = trainer.fit(
            dataset.train.X, dataset.train.y,
            dataset.val.X, dataset.val.y,
        )

        calibrator = TemperatureCalibrator()
        device = trainer.device
        model.eval()
        with torch.no_grad():
            val_logits = model(
                torch.tensor(dataset.val.X, dtype=torch.float32, device=device)
            ).cpu().numpy()
        calibrator.fit(val_logits, dataset.val.y)

        p_cal = model.eval() and self._predict_calibrated(model, calibrator, dataset.test.X, device)
        p_raw = self._predict_raw(model, dataset.test.X, device)

        from agentic_forecaster.evaluation import compute_metrics
        metrics = compute_metrics(dataset.test.y, p_cal)
        metrics["brier_raw"] = float(_brier(dataset.test.y, p_raw))
        metrics["ece_raw"] = float(_ece(dataset.test.y, p_raw, 10))
        metrics["brier_calibrated"] = metrics.pop("brier")
        metrics["ece_calibrated"] = metrics.pop("ece")

        fitted = FittedModel(
            name="attention_lstm", kind="torch", model=model,
            ticker=ticker, fold=fold,
            metrics=metrics,
            calibration=calibrator.to_dict(),
            temperature=calibrator.temperature,
            train_config={
                "hidden_size": int(cfg.get("hidden_size", 64)),
                "num_layers": int(cfg.get("num_layers", 2)),
                "dropout": float(cfg.get("dropout", 0.2)),
                "learning_rate": float(cfg.get("learning_rate", 1e-3)),
                "epochs": int(cfg.get("epochs", 3)),
                "best_epoch": result.best_epoch,
                "best_val_loss": result.best_val_loss,
                "gradient_clip_norm": float(cfg.get("gradient_clip_norm", 1.0)),
            },
            history=result.history,
            seed=self.seed,
            feature_names=dataset.feature_names,
        )
        fitted._scaler = dataset.scaler
        return fitted

    @staticmethod
    def _predict_raw(model, X: np.ndarray, device) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32, device=device))
            return torch.sigmoid(logits).cpu().numpy()

    @staticmethod
    def _predict_calibrated(model, calibrator, X: np.ndarray, device) -> np.ndarray:
        p_raw = ModelAgent._predict_raw(model, X, device)
        return calibrator.calibrate_proba(p_raw)


def _brier(y_true: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import brier_score_loss
    return brier_score_loss(y_true, p)


def _ece(y_true: np.ndarray, p: np.ndarray, n_bins: int) -> float:
    from agentic_forecaster.evaluation.metrics import expected_calibration_error
    return expected_calibration_error(y_true, p, n_bins)
