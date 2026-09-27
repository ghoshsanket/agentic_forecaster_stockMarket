"""Risk Agent — converts conviction into a sized, risk-limited position.

Equations (documented in docs/IMPLEMENTATION_ASSUMPTIONS.md):

* Kelly fraction (half-Kelly by default):
      f* = kelly_fraction * (p * b - (1 - p)) / b
  where ``b`` is the win/loss odds ratio and ``p`` is the calibrated
  probability of an up-move.
* Volatility target sizing:
      vol_size = volatility_target / realised_vol
* Final size = min(kelly_size, vol_size, max_position_pct)
* Stop-loss triggers when adverse move exceeds ``stop_loss_pct``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskDecision:
    ticker: str
    date: str
    direction: str
    conviction: float
    kelly_fraction: float
    position_size: float
    stop_loss: float
    take_profit: float
    risk_level: str


class RiskAgent:
    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.10,
        stop_loss_pct: float = 0.05,
        volatility_target: float = 0.15,
    ):
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.volatility_target = volatility_target

    def kelly_size(self, p: float, b: float = 1.0) -> float:
        """Half-Kelly position size for win probability ``p`` and odds ``b``."""
        if p <= 0.0 or p >= 1.0:
            return 0.0
        raw = (p * b - (1.0 - p)) / b
        return max(0.0, self.kelly_fraction * raw)

    def vol_size(self, realised_vol: float) -> float:
        if realised_vol <= 0:
            return self.max_position_pct
        return min(self.volatility_target / realised_vol, self.max_position_pct)

    def decide(
        self,
        ticker: str,
        date: str,
        direction: str,
        conviction: float,
        realised_vol: float,
    ) -> RiskDecision:
        kelly = self.kelly_size(conviction)
        vol = self.vol_size(realised_vol)
        size = min(kelly, vol, self.max_position_pct)
        if size < 0.01:
            risk_level = "no_trade"
        elif size < 0.03:
            risk_level = "low"
        elif size < 0.06:
            risk_level = "medium"
        else:
            risk_level = "high"
        return RiskDecision(
            ticker=ticker,
            date=str(date),
            direction=direction,
            conviction=float(conviction),
            kelly_fraction=float(kelly),
            position_size=float(size),
            stop_loss=float(self.stop_loss_pct),
            take_profit=float(2.0 * self.stop_loss_pct),
            risk_level=risk_level,
        )
