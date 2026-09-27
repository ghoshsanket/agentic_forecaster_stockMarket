"""Risk Agent: ATR-based stop-loss / take-profit (Phase-1, PAPER-DEFINED).

The Phase-1 risk implementation is ATR-based.  There is deliberately NO
position-sizing logic (no Kelly fraction, no fixed-percentage stop).
"""

from agentic_forecaster.risk.risk_agent import RiskAgent, RiskDecision

__all__ = ["RiskAgent", "RiskDecision"]
