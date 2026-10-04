"""V2 multi-task loss with FIXED weights.

The three objectives and their weights are fixed before any experiment is run
and are NOT tuned in this development programme:

===========================  =========================  =======
task                         loss                       weight
===========================  =========================  =======
direction (headline)         ``BCEWithLogitsLoss``      1.00
volatility-normalised return ``SmoothL1Loss``          0.50
cross-sectional next-day rank ``SmoothL1Loss``         0.25
===========================  =========================  =======

``loss = 1.00 * direction + 0.50 * return + 0.25 * rank``

Each component is recorded separately in the training history and in the
experiment outputs, so it is always visible whether a multi-task model improved
the headline direction metric or only the auxiliary ones.

The return target was already clipped to a documented fixed range
(``[-5, +5]``) when the store was built; the same bound is re-applied here as a
guard so an out-of-range label can never dominate the gradient.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

#: Fixed clipping bound of the normalised return target (set before experiments).
RETURN_TARGET_CLIP = 5.0


@dataclass(frozen=True)
class MultiTaskWeights:
    """The fixed, pre-registered loss weights."""

    direction: float = 1.00
    return_value: float = 0.50
    rank: float = 0.25

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "return": self.return_value,
            "rank": self.rank,
        }


class V2MultiTaskLoss(nn.Module):
    """Weighted sum of the direction, return and rank objectives."""

    def __init__(self, weights: MultiTaskWeights | None = None, *,
                 return_clip: float = RETURN_TARGET_CLIP,
                 smooth_l1_beta: float = 1.0) -> None:
        super().__init__()
        self.weights = weights or MultiTaskWeights()
        self.return_clip = float(return_clip)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.direction_loss = nn.BCEWithLogitsLoss()
        self.return_loss = nn.SmoothL1Loss(beta=self.smooth_l1_beta)
        self.rank_loss = nn.SmoothL1Loss(beta=self.smooth_l1_beta)

    def forward(self, outputs: dict[str, torch.Tensor],
                targets: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Return ``(total_loss, components)``.

        ``components`` always contains the direction loss; return and rank
        components appear only when the corresponding head and target exist, so a
        single-task variant cannot silently pay for a head it does not have.
        """
        y_direction = targets["y_direction"].float().reshape(-1)
        direction = self.direction_loss(outputs["direction_logit"].reshape(-1), y_direction)
        total = self.weights.direction * direction
        components = {"direction": float(direction.detach())}
        components["direction_weighted"] = float(
            (self.weights.direction * direction).detach())

        if "return_prediction" in outputs and "y_return" in targets:
            y_return = targets["y_return"].float().reshape(-1)
            valid = torch.isfinite(y_return)
            if bool(valid.any()):
                clipped = y_return[valid].clamp(-self.return_clip, self.return_clip)
                value = self.return_loss(outputs["return_prediction"].reshape(-1)[valid],
                                         clipped)
                total = total + self.weights.return_value * value
                components["return"] = float(value.detach())
                components["return_weighted"] = float(
                    (self.weights.return_value * value).detach())
                components["return_n"] = int(valid.sum())

        if "rank_prediction" in outputs and "y_rank" in targets:
            y_rank = targets["y_rank"].float().reshape(-1)
            valid = torch.isfinite(y_rank)
            if bool(valid.any()):
                value = self.rank_loss(outputs["rank_prediction"].reshape(-1)[valid],
                                       y_rank[valid])
                total = total + self.weights.rank * value
                components["rank"] = float(value.detach())
                components["rank_weighted"] = float((self.weights.rank * value).detach())
                components["rank_n"] = int(valid.sum())

        components["total"] = float(total.detach())
        return total, components


@dataclass
class LossAccumulator:
    """Running mean of the loss components over an epoch."""

    weights: MultiTaskWeights = field(default_factory=MultiTaskWeights)
    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def update(self, components: dict[str, float]) -> None:
        for key, value in components.items():
            if key.endswith("_n"):
                self.counts[key] = self.counts.get(key, 0) + int(value)
                continue
            self.totals[key] = self.totals.get(key, 0.0) + float(value)
            self.counts[key] = self.counts.get(key, 0) + 1

    def result(self) -> dict[str, float]:
        n = max(self.counts.get("total", 0), 1)
        return {k: v / n for k, v in self.totals.items()}