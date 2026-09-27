"""Model Agent — second agent in the five-agent workflow.

Trains all five model families (Attention-LSTM, plain LSTM, Random Forest,
Logistic Regression, Majority) and persists them under
``$AGENTIC_MODEL_ROOT`` (Category B).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import torch

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


@dataclass
class FittedModel:
    name: str
    kind: str  # 'torch' | 'sklearn' | 'majority'
    model: object
    metrics: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    checkpoint_path: str | None = None
    scaler_path: str | None = None
    feature_names: list[str] = field(default_factory=list)
    train_config: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)
    seed: int = 42

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "torch":
            self.model.eval()
            device = next(self.model.parameters()).device
            with torch.no_grad():
                logits = self.model(torch.tensor(X, dtype=torch.float32, device=device))
                return torch.softmax(logits, dim=1).cpu().numpy()
        return self.model.predict_proba(X)

    def save(self, root: str | Path) -> Path:
        root = ensure_dir(root)
        root.mkdir(parents=True, exist_ok=True)
        if self.kind == "torch":
            atomic_torch_save(self.model.state_dict(), root / "model.pt")
        elif self.kind != "majority":
            joblib.dump(self.model, root / "model.joblib")
        atomic_json_dump(self.metrics, root / "metrics.json")
        atomic_json_dump(self.calibration, root / "calibration.json")
        atomic_json_dump(self.train_config, root / "config.json")
        atomic_json_dump(
            {"name": self.name, "kind": self.kind, "seed": self.seed},
            root / "manifest.json",
        )
        if self.feature_names:
            (root / "feature_names.txt").write_text("\n".join(self.feature_names))
        return root


class ModelAgent:
    def __init__(self, config: dict):
        self.config = config
        self.model_cfg = config.get("models", {})
        self.cal_cfg = config.get("calibration", {})
        self.seed = int(config.get("experiment", {}).get("seed", 42))

    def run(self, dataset) -> dict[str, FittedModel]:
        fitted: dict[str, FittedModel] = {}
        fitted["attention_lstm"] = self._train_attention_lstm(dataset)
        fitted["lstm"] = self._train_lstm(dataset)
        fitted["random_forest"] = self._train_sklearn(
            RandomForestBaseline(
                n_estimators=int(self.model_cfg["random_forest"].get("n_estimators", 300)),
                max_depth=int(self.model_cfg["random_forest"].get("max_depth", 8)),
                min_samples_leaf=int(self.model_cfg["random_forest"].get("min_samples_leaf", 5)),
                seed=self.seed,
            ),
            dataset,
            "random_forest",
        )
        fitted["logistic_regression"] = self._train_sklearn(
            LogisticRegressionBaseline(
                C=float(self.model_cfg["logistic_regression"].get("C", 1.0)),
                max_iter=int(self.model_cfg["logistic_regression"].get("max_iter", 1000)),
                seed=self.seed,
            ),
            dataset,
            "logistic_regression",
        )
        fitted["majority"] = self._train_majority(dataset)
        return fitted

    def _train_attention_lstm(self, dataset) -> FittedModel:
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
            epochs=int(cfg.get("epochs", 100)),
            patience=int(cfg.get("patience", 15)),
            class_weight=cfg.get("class_weight", "balanced"),
            seed=self.seed,
        )
        result = trainer.fit(
            dataset.train.X, dataset.train.y,
            dataset.val.X, dataset.val.y,
        )
        proba = self._torch_predict(model, dataset.test.X)
        from agentic_forecaster.evaluation import compute_metrics
        metrics = compute_metrics(dataset.test.y, proba[:, 1])
        return FittedModel(
            name="attention_lstm", kind="torch", model=model,
            metrics=metrics,
            train_config={
                "hidden_size": int(cfg.get("hidden_size", 64)),
                "num_layers": int(cfg.get("num_layers", 2)),
                "dropout": float(cfg.get("dropout", 0.2)),
                "learning_rate": float(cfg.get("learning_rate", 1e-3)),
                "best_epoch": result.best_epoch,
                "best_val_loss": result.best_val_loss,
            },
            history=result.history,
            seed=self.seed, feature_names=dataset.feature_names,
        )

    def _train_lstm(self, dataset) -> FittedModel:
        cfg = self.model_cfg["lstm"]
        model = PlainLSTM(
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
            epochs=int(cfg.get("epochs", 100)),
            patience=int(cfg.get("patience", 15)),
            class_weight=cfg.get("class_weight", "balanced"),
            seed=self.seed,
        )
        trainer.fit(dataset.train.X, dataset.train.y, dataset.val.X, dataset.val.y)
        proba = self._torch_predict(model, dataset.test.X)
        from agentic_forecaster.evaluation import compute_metrics
        metrics = compute_metrics(dataset.test.y, proba[:, 1])
        return FittedModel(
            name="lstm", kind="torch", model=model, metrics=metrics,
            seed=self.seed, feature_names=dataset.feature_names,
        )

    def _train_sklearn(self, baseline, dataset, name: str) -> FittedModel:
        baseline.fit(dataset.train.X, dataset.train.y)
        proba = baseline.predict_proba(dataset.test.X)
        from agentic_forecaster.evaluation import compute_metrics
        metrics = compute_metrics(dataset.test.y, proba[:, 1])
        return FittedModel(
            name=name, kind="sklearn", model=baseline.model, metrics=metrics,
            seed=self.seed, feature_names=dataset.feature_names,
        )

    def _train_majority(self, dataset) -> FittedModel:
        baseline = MajorityBaseline().fit(dataset.train.X, dataset.train.y)
        proba = baseline.predict_proba(dataset.test.X)
        from agentic_forecaster.evaluation import compute_metrics
        metrics = compute_metrics(dataset.test.y, proba[:, 1])
        return FittedModel(
            name="majority", kind="majority", model=baseline, metrics=metrics,
            seed=self.seed, feature_names=dataset.feature_names,
        )

    @staticmethod
    def _torch_predict(model, X: np.ndarray) -> np.ndarray:
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32, device=device))
            return torch.softmax(logits, dim=1).cpu().numpy()
