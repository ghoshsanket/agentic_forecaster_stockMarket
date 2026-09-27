"""Canonical model-bundle save/load API.

A model bundle is a directory containing:
    model.pt          — state_dict
    scaler.joblib     — fitted StandardScaler
    calibration.json  — temperature + metadata
    config.json       — model architecture + training config
    metrics.json      — evaluation metrics
    manifest.json     — name, kind, ticker, fold, seed, temperature
    feature_names.txt — ordered feature names

``save_model_bundle`` writes all of these.  ``load_model_bundle`` reconstructs
the architecture from config, loads the state_dict, restores the scaler and
calibration, and returns a ``FittedModel`` ready for inference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import torch

from agentic_forecaster.agents.model_agent import FittedModel
from agentic_forecaster.calibration import TemperatureCalibrator
from agentic_forecaster.models import AttentionLSTM
from agentic_forecaster.utils import ensure_dir

logger = logging.getLogger("agentic_forecaster.models.checkpoint")


def save_model_bundle(fitted_model, root: str | Path) -> Path:
    """Save a complete model bundle to ``root``."""
    root = ensure_dir(root)
    fitted_model.save(root)
    logger.info("Model bundle saved to %s", root)
    return root


def load_model_bundle(root: str | Path, device: str | None = None) -> FittedModel:
    """Load a model bundle and return a ready-to-use FittedModel."""
    root = Path(root)

    manifest = json.loads((root / "manifest.json").read_text())
    config = json.loads((root / "config.json").read_text())
    calibration_dict = json.loads((root / "calibration.json").read_text())
    metrics = json.loads((root / "metrics.json").read_text())
    feature_names = (root / "feature_names.txt").read_text().splitlines()

    model_cfg = config
    model = AttentionLSTM(
        input_size=len(feature_names),
        hidden_size=int(model_cfg.get("hidden_size", 64)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )

    state_dict = torch.load(root / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)

    if device is not None:
        model.to(device)
    model.eval()

    scaler = joblib.load(root / "scaler.joblib")
    calibrator = TemperatureCalibrator.from_dict(calibration_dict)

    fitted = FittedModel(
        name=manifest.get("name", "attention_lstm"),
        kind=manifest.get("kind", "torch"),
        model=model,
        ticker=manifest.get("ticker", ""),
        fold=manifest.get("fold", ""),
        metrics=metrics,
        calibration=calibration_dict,
        temperature=calibrator.temperature,
        train_config=config,
        seed=int(manifest.get("seed", 42)),
        feature_names=feature_names,
    )
    fitted._scaler = scaler
    logger.info("Model bundle loaded from %s (ticker=%s, fold=%s)",
                root, fitted.ticker, fitted.fold)
    return fitted
