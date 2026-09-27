"""Model Agent — trains ONE independent model per stock, plus baselines.

Per ticker the agent trains and evaluates:

    Attention-LSTM  (primary)
    Plain LSTM      (baseline)
    Random Forest   (baseline)
    Logistic Regression (baseline)
    Majority Class  (baseline)

Every model sees the same temporal split and the same per-ticker scaler.
Runtime bundle layout::

    $AGENTIC_MODEL_ROOT/trained/<ticker>/<fold>/
        model.pt  scaler.joblib  calibration.json  config.json
        metrics.json  manifest.json  feature_names.txt  background.npy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import torch

from agentic_forecaster.calibration import TemperatureCalibrator
from agentic_forecaster.evaluation import compute_metrics
from agentic_forecaster.models import (
    AttentionLSTM,
    LogisticRegressionBaseline,
    MajorityBaseline,
    PlainLSTM,
    RandomForestBaseline,
)
from agentic_forecaster.training import Trainer
from agentic_forecaster.utils import atomic_json_dump, atomic_torch_save, ensure_dir

logger = logging.getLogger("agentic_forecaster.agents.model")

# Models that expose a single binary logit (and therefore a temperature).
_TORCH_MODELS = ("attention_lstm", "lstm")


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
    feature_names: list[str] = field(default_factory=list)
    train_config: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)
    seed: int = 42
    _scaler: object = field(default=None, repr=False)
    _background: np.ndarray | None = field(default=None, repr=False)

    # ------------------------------------------------------------ inference

    def _torch_raw(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32, device=device))
            return torch.sigmoid(logits).cpu().numpy()

    def predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        """Uncalibrated p_up."""
        if self.kind == "torch":
            return self._torch_raw(X)
        return np.asarray(self.model.predict_proba(X))[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """CALIBRATED p_up (the default used everywhere downstream)."""
        raw = self.predict_proba_raw(X)
        if self.kind == "torch" and self.temperature != 1.0:
            cal = TemperatureCalibrator()
            cal.temperature = self.temperature
            return cal.calibrate_proba(raw)
        return raw

    # --------------------------------------------------------- persistence

    def save(self, root: str | Path) -> Path:
        root = ensure_dir(root)
        if self.kind == "torch":
            atomic_torch_save(self.model.state_dict(), root / "model.pt")
        elif self.kind != "majority":
            joblib.dump(self.model, root / "model.joblib")
        if self._scaler is not None:
            joblib.dump(self._scaler, root / "scaler.joblib")
        if self._background is not None:
            np.save(root / "background.npy", self._background)
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
        (root / "feature_names.txt").write_text("\n".join(self.feature_names))
        return root


class ModelAgent:
    def __init__(self, config: dict):
        self.config = config
        self.model_cfg = config.get("models", {})
        self.baseline_cfg = config.get("baselines", {})
        self.seed = int(config.get("experiment", {}).get("seed", 42))

    # ------------------------------------------------------------- helpers

    def _trainer(self, model, cfg: dict, device: str | None) -> Trainer:
        return Trainer(
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

    def _fit_temperature(self, model, X_val, y_val, device) -> TemperatureCalibrator:
        model.eval()
        with torch.no_grad():
            logits = model(
                torch.tensor(X_val, dtype=torch.float32, device=device)
            ).cpu().numpy()
        return TemperatureCalibrator().fit(logits, y_val)

    @staticmethod
    def _evaluate(y_true, p_cal, p_raw) -> dict:
        """Metrics for one model, reporting raw AND calibrated Brier/ECE."""
        from sklearn.metrics import brier_score_loss

        from agentic_forecaster.evaluation.metrics import expected_calibration_error

        metrics = compute_metrics(y_true, p_cal)
        metrics.pop("brier", None)
        metrics["brier_raw"] = float(brier_score_loss(y_true, p_raw))
        metrics["brier_calibrated"] = float(brier_score_loss(y_true, p_cal))
        metrics["ece_raw"] = float(expected_calibration_error(y_true, p_raw, 10))
        metrics["ece_calibrated"] = float(expected_calibration_error(y_true, p_cal, 10))
        return metrics

    # --------------------------------------------------------- entry point

    def train_ticker(self, ticker: str, dataset, fold: str = "fold_0",
                     device: str | None = None) -> FittedModel:
        """Train the primary Attention-LSTM for one ticker."""
        from agentic_forecaster.explainability.shap_explainer import sample_background

        cfg = self.model_cfg["attention_lstm"]
        model = AttentionLSTM(
            input_size=dataset.train.X.shape[-1],
            hidden_size=int(cfg.get("hidden_size", 64)),
            num_layers=int(cfg.get("num_layers", 2)),
            dropout=float(cfg.get("dropout", 0.2)),
        )
        trainer = self._trainer(model, cfg, device)
        result = trainer.fit(
            dataset.train.X, dataset.train.y, dataset.val.X, dataset.val.y
        )

        calibrator = self._fit_temperature(
            model, dataset.val.X, dataset.val.y, trainer.device
        )
        p_raw = model.eval() and self._raw(model, dataset.test.X, trainer.device)
        p_cal = calibrator.calibrate_proba(p_raw)
        metrics = self._evaluate(dataset.test.y, p_cal, p_raw)

        background = sample_background(
            dataset.train.X,
            size=int(self.config.get("explainability", {}).get("background_samples", 64)),
        )

        fitted = FittedModel(
            name="attention_lstm", kind="torch", model=model,
            ticker=ticker, fold=fold, metrics=metrics,
            calibration=calibrator.to_dict(),
            temperature=calibrator.temperature,
            train_config={
                "hidden_size": int(cfg.get("hidden_size", 64)),
                "num_layers": int(cfg.get("num_layers", 2)),
                "dropout": float(cfg.get("dropout", 0.2)),
                "learning_rate": float(cfg.get("learning_rate", 1e-3)),
                "epochs": int(cfg.get("epochs", 3)),
                "batch_size": int(cfg.get("batch_size", 64)),
                "patience": int(cfg.get("patience", 10)),
                "gradient_clip_norm": float(cfg.get("gradient_clip_norm", 1.0)),
                "best_epoch": result.best_epoch,
                "best_val_loss": result.best_val_loss,
            },
            history=result.history,
            seed=self.seed,
            feature_names=list(dataset.feature_names),
        )
        fitted._scaler = dataset.scaler
        fitted._background = background
        return fitted

    def train_baselines(self, ticker: str, dataset, fold: str = "fold_0",
                        device: str | None = None) -> dict[str, FittedModel]:
        """Train every enabled baseline on the same per-ticker split."""
        out: dict[str, FittedModel] = {}
        if not self.baseline_cfg.get("enabled", True):
            return out

        lstm_cfg = dict(self.model_cfg.get("lstm", self.model_cfg["attention_lstm"]))
        if self.baseline_cfg.get("lstm", True):
            model = PlainLSTM(
                input_size=dataset.train.X.shape[-1],
                hidden_size=int(lstm_cfg.get("hidden_size", 64)),
                num_layers=int(lstm_cfg.get("num_layers", 2)),
                dropout=float(lstm_cfg.get("dropout", 0.2)),
            )
            trainer = self._trainer(model, lstm_cfg, device)
            trainer.fit(dataset.train.X, dataset.train.y, dataset.val.X, dataset.val.y)
            p_raw = self._raw(model, dataset.test.X, trainer.device)
            p_cal = self._fit_temperature(
                model, dataset.val.X, dataset.val.y, trainer.device
            ).calibrate_proba(p_raw)
            out["lstm"] = FittedModel(
                name="lstm", kind="torch", model=model, ticker=ticker, fold=fold,
                metrics=self._evaluate(dataset.test.y, p_cal, p_raw),
                feature_names=list(dataset.feature_names), seed=self.seed,
            )

        if self.baseline_cfg.get("random_forest", True):
            rf_cfg = self.model_cfg.get("random_forest", {})
            base = RandomForestBaseline(
                n_estimators=int(rf_cfg.get("n_estimators", 300)),
                max_depth=rf_cfg.get("max_depth", 8),
                min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 5)),
                seed=self.seed,
            ).fit(dataset.train.X, dataset.train.y)
            p = np.asarray(base.predict_proba(dataset.test.X))[:, 1]
            out["random_forest"] = FittedModel(
                name="random_forest", kind="sklearn", model=base.model,
                ticker=ticker, fold=fold,
                metrics=compute_metrics(dataset.test.y, p),
                feature_names=list(dataset.feature_names), seed=self.seed,
            )

        if self.baseline_cfg.get("logistic_regression", True):
            lr_cfg = self.model_cfg.get("logistic_regression", {})
            base = LogisticRegressionBaseline(
                C=float(lr_cfg.get("C", 1.0)),
                max_iter=int(lr_cfg.get("max_iter", 1000)),
                seed=self.seed,
            ).fit(dataset.train.X, dataset.train.y)
            p = np.asarray(base.predict_proba(dataset.test.X))[:, 1]
            out["logistic_regression"] = FittedModel(
                name="logistic_regression", kind="sklearn", model=base.model,
                ticker=ticker, fold=fold,
                metrics=compute_metrics(dataset.test.y, p),
                feature_names=list(dataset.feature_names), seed=self.seed,
            )

        if self.baseline_cfg.get("majority", True):
            base = MajorityBaseline().fit(dataset.train.X, dataset.train.y)
            p = np.asarray(base.predict_proba(dataset.test.X))[:, 1]
            out["majority"] = FittedModel(
                name="majority", kind="majority", model=base, ticker=ticker, fold=fold,
                metrics=compute_metrics(dataset.test.y, p),
                feature_names=list(dataset.feature_names), seed=self.seed,
            )
        return out

    @staticmethod
    def _raw(model, X, device) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32, device=device))
            return torch.sigmoid(logits).cpu().numpy()
