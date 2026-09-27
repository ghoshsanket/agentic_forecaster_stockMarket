"""Canonical model-bundle save/load API.

A bundle directory contains::

    model.pt           state_dict
    scaler.joblib      fitted StandardScaler
    calibration.json   temperature + metadata
    config.json        architecture + training config
    metrics.json       evaluation metrics
    manifest.json      name, kind, ticker, fold, seed, temperature
    feature_names.txt  ordered feature names
    background.npy     SHAP background sampled from TRAINING sequences

``save_model_bundle`` writes all of these; ``load_model_bundle`` rebuilds the
architecture from ``config.json``, loads the state_dict, restores the scaler,
calibration and background, and returns a ready-to-use ``FittedModel``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import torch

from agentic_forecaster.agents.model_agent import FittedModel
from agentic_forecaster.calibration import TemperatureCalibrator
from agentic_forecaster.models import AttentionLSTM
from agentic_forecaster.utils import ensure_dir

logger = logging.getLogger("agentic_forecaster.models.checkpoint")


def save_model_bundle(fitted_model: FittedModel, root: str | Path) -> Path:
    """Write a complete model bundle to ``root``."""
    root = ensure_dir(root)
    fitted_model.save(root)
    logger.info("Model bundle saved to %s", root)
    return root


def load_model_bundle(root: str | Path, device: str | None = None) -> FittedModel:
    """Load a bundle and return a usable FittedModel.

    ``device`` accepts ``auto``/``cpu``/``cuda``/``cuda:N`` and is normalised
    through the same :func:`resolve_device` used for training, so a literal
    ``"auto"`` is never handed to ``torch``.
    """
    from agentic_forecaster.training import resolve_device

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    config = json.loads((root / "config.json").read_text())
    calibration_dict = json.loads((root / "calibration.json").read_text())
    metrics = json.loads((root / "metrics.json").read_text())
    feature_names = (root / "feature_names.txt").read_text().splitlines()

    model = AttentionLSTM(
        input_size=len(feature_names),
        hidden_size=int(config.get("hidden_size", 64)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.2)),
    )
    state_dict = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    torch_device = resolve_device(device)
    model.to(torch_device)
    model.eval()

    scaler_path = root / "scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.is_file() else None
    background_path = root / "background.npy"
    background = np.load(background_path) if background_path.is_file() else None

    fitted = FittedModel(
        name=manifest.get("name", "attention_lstm"),
        kind=manifest.get("kind", "torch"),
        model=model,
        ticker=manifest.get("ticker", ""),
        fold=manifest.get("fold", ""),
        metrics=metrics,
        calibration=calibration_dict,
        temperature=TemperatureCalibrator.from_dict(calibration_dict).temperature,
        train_config=config,
        seed=int(manifest.get("seed", 42)),
        feature_names=feature_names,
    )
    fitted._scaler = scaler
    fitted._background = background
    logger.info("Loaded bundle %s (ticker=%s fold=%s device=%s T=%.4f)",
                root, fitted.ticker, fitted.fold, torch_device, fitted.temperature)
    return fitted
