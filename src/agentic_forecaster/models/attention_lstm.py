"""Attention-LSTM classifier.

Architecture:
    LSTM (2 layers) -> additive attention over time steps -> linear head.

The attention weights are returned alongside logits so the Explainer Agent
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
        num_classes: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
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
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)  # (B, T, H)
        scores = self.attention(lstm_out).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=1)  # (B, T)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (B, H)
        context = self.dropout(context)
        logits = self.classifier(context)
        if return_attention:
            return logits, weights
        return logits
