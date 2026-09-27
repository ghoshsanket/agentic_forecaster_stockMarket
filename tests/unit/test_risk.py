"""Unit tests for the ATR-based risk agent."""

from __future__ import annotations

from agentic_forecaster.risk import RiskAgent


def test_up_direction_high_confidence():
    ra = RiskAgent()
    d = ra.decide("X", "2024-01-01", p_up=0.85, close=100.0, atr=2.0)
    assert d.direction == "UP"
    assert d.confidence_level == "HIGH"


def test_down_direction():
    ra = RiskAgent()
    d = ra.decide("X", "2024-01-01", p_up=0.30, close=100.0, atr=2.0)
    assert d.direction == "DOWN"
    assert d.confidence == 0.70


def test_stop_loss_is_price_not_percentage():
    ra = RiskAgent()
    d = ra.decide("X", "2024-01-01", p_up=0.85, close=100.0, atr=2.0)
    assert d.stop_loss == 100.0 - 0.8 * 2.0
    assert d.take_profit == 100.0 + 1.5 * 2.0


def test_risk_score_formula():
    ra = RiskAgent()
    d = ra.decide("X", "2024-01-01", p_up=0.85, close=100.0, atr=2.0)
    assert abs(d.risk_score - 0.85 / 2.0) < 1e-9
