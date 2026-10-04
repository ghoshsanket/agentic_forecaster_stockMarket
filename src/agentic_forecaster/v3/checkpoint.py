"""Persistence for V3 fits: scalers, PCA, feature schemas and the neural weights.

Every fitted transformation is saved WITH its experiment, so a result can be
re-scored without refitting and so an auditor can see exactly what was fitted on
TRAIN rows only.  The neural checkpoint records the two feature schemas separately,
because the stock and exogenous inputs are consumed by different encoders.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agentic_forecaster.utils import atomic_json_dump

logger = logging.getLogger("agentic_forecaster.v3.checkpoint")

CHECKPOINT_NAME = "checkpoint.pt"
TRANSFORM_NAME = "fitted_transform.json"


@dataclass
class TransformState:
    """A fitted, persisted TRAIN-only transformation."""

    scaler: object | None
    pca: object | None
    state: dict

    def save(self, path: Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "pca": self.pca,
                     "state": self.state}, path)
        return path

    @classmethod
    def load(cls, path: Path) -> TransformState:
        import joblib

        payload = joblib.load(Path(path))
        return cls(scaler=payload.get("scaler"), pca=payload.get("pca"),
                   state=payload.get("state", {}))


def save_transform(fitted: dict, state: dict, directory: Path) -> Path:
    """Persist a fitted scaler/PCA pair with its provenance."""
    path = Path(directory) / TRANSFORM_NAME
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": fitted.get("scaler"), "pca": fitted.get("pca")}, path)
    atomic_json_dump(state, Path(directory) / "transform_state.json")
    return path


def save_neural_checkpoint(model, directory: Path, *, extra: dict | None = None
                           ) -> dict:
    """Save the dual-encoder neural weights and their architecture record."""
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CHECKPOINT_NAME
    torch.save({"state_dict": model.state_dict(),
                "architecture": getattr(model, "architecture_record", dict)()},
               path)
    payload = {
        "checkpoint": str(path),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)),
        **(extra or {}),
    }
    atomic_json_dump(payload, directory / "checkpoint_manifest.json")
    return payload


def load_neural_checkpoint(path: Path, model) -> dict:
    """Restore neural weights into an already-constructed model."""
    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return payload.get("architecture", {})


def feature_schema_record(stock_columns: list[str], exogenous_columns: list[str]) -> dict:
    """The two input schemas, hashed separately because two encoders consume them."""
    import hashlib

    def _digest(columns: list[str]) -> str:
        return hashlib.sha256(json.dumps(sorted(columns)).encode()).hexdigest()

    return {
        "stock_feature_count": len(stock_columns),
        "exogenous_feature_count": len(exogenous_columns),
        "stock_features": list(stock_columns),
        "exogenous_features": list(exogenous_columns),
        "stock_schema_sha256": _digest(stock_columns),
        "exogenous_schema_sha256": _digest(exogenous_columns),
        "combined_schema_sha256": _digest(list(stock_columns) + list(exogenous_columns)),
        "note": ("the exogenous schema may be EMPTY, which is exactly the stock-only "
                 "control"),
    }


def finite_matrix(matrix: np.ndarray) -> bool:
    return bool(np.isfinite(np.asarray(matrix, dtype=float)).all())