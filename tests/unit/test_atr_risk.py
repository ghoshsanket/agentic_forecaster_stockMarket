"""Tests for ATR-based risk agent."""

from __future__ import annotations

from agentic_forecaster.risk.risk_agent import RiskAgent


def test_high_confidence_band():
    ra = RiskAgent()
    d = ra.decide("X", "2024-01-01", p_up=0.85, close=100.0, atr=2.0)
    assert d.confidence_level == "HIGH"
    assert d.lambda_sl == 0.8
    assert d.lambda_tp == 1.5
    assert d.stop_loss == 100.0 - 0.8 * 2.0
    assert d.take_profit == 100.0 + 1.5 * 2.0


def test_medium_confidence_band():
    ra = RiskAgent()
    d = ra.decide("X", "2024-01-01", p_up=0.70, close=100.0, atr=2.0)
    assert d.confidence_level == "MEDIUM"
    assert d.lambda_sl == 1.0
    assert d.lambda_tp == 1.0


def test_low_confidence_band():
    ra = RiskAgent()
    d = ra.decide("X", "2024-01-01", p_up=0.55, close=100.0, atr=2.0)
    assert d.confidence_level == "LOW"
    assert d.lambda_sl == 1.2
    assert d.lambda_tp == 0.8


def test_down_direction():
    ra = RiskAgent()
    d = ra.decide("X", "2024-01-01", p_up=0.30, close=100.0, atr=2.0)
    assert d.direction == "DOWN"
    assert d.confidence == 0.70
    assert d.stop_loss == 100.0 + 1.0 * 2.0
    assert d.take_profit == 100.0 - 1.0 * 2.0


def test_rrr_and_risk_score():
    ra = RiskAgent()
    d = ra.decide("X", "2024-01-01", p_up=0.85, close=100.0, atr=2.0)
    expected_rrr = abs(d.take_profit - 100.0) / abs(100.0 - d.stop_loss)
    assert abs(d.rrr - expected_rrr) < 1e-9
    assert abs(d.risk_score - 0.85 / 2.0) < 1e-9
