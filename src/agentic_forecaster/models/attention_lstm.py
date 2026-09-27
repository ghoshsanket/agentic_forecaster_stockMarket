"""Attention-LSTM classifier (binary logit output).

Architecture (RECONSTRUCTION-ASSUMED hyperparameters):
    LSTM (2 layers, hidden 64) -> additive attention over time steps
    -> Linear(hidden_size, 1) -> single logit.

The model outputs a single logit consistent with the paper's binary
formulation.  Training uses ``BCEWithLogitsLoss`` and ``p_up = sigmoid(logit)``.

The attention weights are returned alongside the logit so the Explainer Agent
can surface *which days* the model attended to (attention evidence).
"""

from __future__ import annotations

import torch
from torch import nn


class AttentionLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)  # (B, T, H)
        scores = self.attention(lstm_out).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=1)  # (B, T)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (B, H)
        context = self.dropout(context)
        logit = self.classifier(context).squeeze(-1)  # (B,)
        if return_attention:
            return logit, weights
        return logit

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return p_up = sigmoid(logit)."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self(x))
