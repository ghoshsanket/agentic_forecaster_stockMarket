"""Unit tests for the risk agent."""

from __future__ import annotations

from agentic_forecaster.risk import RiskAgent


def test_kelly_zero_at_coin_flip():
    ra = RiskAgent(kelly_fraction=0.25)
    assert ra.kelly_size(0.5) == 0.0


def test_kelly_positive_when_edge():
    ra = RiskAgent(kelly_fraction=0.25)
    assert ra.kelly_size(0.7) > 0.0


def test_position_capped():
    ra = RiskAgent(max_position_pct=0.10)
    d = ra.decide("X", "2024-01-01", "UP", 0.99, 0.01)
    assert d.position_size <= 0.10


def test_no_trade_low_conviction():
    ra = RiskAgent()
    d = ra.decide("X", "2024-01-01", "UP", 0.51, 0.02)
    assert d.risk_level == "no_trade"
