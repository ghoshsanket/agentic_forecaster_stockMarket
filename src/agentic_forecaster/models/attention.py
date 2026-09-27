from __future__ import annotations

import torch
from torch import nn


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.W_a = nn.Linear(hidden_dim, attention_dim, bias=True)
        self.v_a = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.v_a(torch.tanh(self.W_a(hidden_states))).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights.unsqueeze(-1) * hidden_states, dim=1)
        return context, weights
