"""Temperature scaling for binary-logit probability calibration.

Fits a single scalar temperature ``T`` on validation logits by minimising
binary NLL, then divides logits by ``T`` at inference.  A scalar ``T``
preserves ranking (so accuracy / Precision@3 are unchanged) while improving
Brier and ECE.

CALIBRATION METHOD IS RECONSTRUCTION-ASSUMED.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TemperatureCalibrator:
    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, y: np.ndarray) -> TemperatureCalibrator:
        """Fit temperature on validation logits (shape ``(n,)``) and labels."""
        logits_t = torch.tensor(logits, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss()
        # Log-parameterisation keeps T positive.
        log_t = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = criterion(logits_t / torch.exp(log_t), y_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_t).item())
        return self

    def calibrate_proba(self, p_up: np.ndarray) -> np.ndarray:
        """Calibrate probabilities ``p_up`` using the fitted temperature.

        Converts probabilities back to logits, divides by T, re-sigmoids.
        """
        p = np.clip(p_up, 1e-7, 1 - 1e-7)
        logits = np.log(p / (1 - p))
        calibrated = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-calibrated))

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        return logits / self.temperature

    def to_dict(self) -> dict:
        return {
            "method": "temperature",
            "temperature": self.temperature,
            "classification": "reconstruction_assumed",
        }

    @classmethod
    def from_dict(cls, d: dict) -> TemperatureCalibrator:
        obj = cls()
        obj.temperature = float(d.get("temperature", 1.0))
        return obj
