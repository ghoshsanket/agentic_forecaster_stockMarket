import pytest

from agentic_forecaster.risk import RiskAgent


@pytest.fixture
def risk_agent():
    return RiskAgent()


class TestRiskAgentBullish:
    def test_up_direction(self, risk_agent):
        decision = risk_agent.decide("TEST", "2024-01-01", "UP", 0.75, 0.02)
        assert decision.direction == "UP"
        assert decision.ticker == "TEST"

    def test_up_direction_with_high_confidence(self, risk_agent):
        decision = risk_agent.decide("TEST", "2024-01-01", "UP", 0.85, 0.02)
        assert decision.conviction == 0.85

    def test_up_direction_with_medium_confidence(self, risk_agent):
        decision = risk_agent.decide("TEST", "2024-01-01", "UP", 0.65, 0.02)
        assert decision.conviction == 0.65

    def test_up_direction_with_low_confidence(self, risk_agent):
        decision = risk_agent.decide("TEST", "2024-01-01", "UP", 0.55, 0.02)
        assert decision.conviction == 0.55

    def test_stop_loss_is_percentage(self, risk_agent):
        decision = risk_agent.decide("TEST", "2024-01-01", "UP", 0.75, 0.02)
        assert decision.stop_loss == 0.05

    def test_take_profit_is_percentage(self, risk_agent):
        decision = risk_agent.decide("TEST", "2024-01-01", "UP", 0.75, 0.02)
        assert decision.take_profit == 0.10


class TestRiskAgentBearish:
    def test_down_direction(self, risk_agent):
        decision = risk_agent.decide("TEST", "2024-01-01", "DOWN", 0.75, 0.02)
        assert decision.direction == "DOWN"

    def test_down_direction_with_high_confidence(self, risk_agent):
        decision = risk_agent.decide("TEST", "2024-01-01", "DOWN", 0.85, 0.02)
        assert decision.conviction == 0.85


class TestRiskAgentPositionSize:
    def test_position_size_positive(self, risk_agent):
        decision = risk_agent.decide("TEST", "2024-01-01", "UP", 0.75, 0.02)
        assert decision.position_size >= 0

    def test_position_size_bounded(self, risk_agent):
        decision = risk_agent.decide("TEST", "2024-01-01", "UP", 0.99, 0.01)
        assert decision.position_size <= risk_agent.max_position_pct

    def test_no_trade_low_conviction(self, risk_agent):
        decision = risk_agent.decide("TEST", "2024-01-01", "UP", 0.51, 0.02)
        assert decision.risk_level == "no_trade"


class TestRiskAgentRiskLevel:
    def test_risk_level_assigned(self, risk_agent):
        decision = risk_agent.decide("TEST", "2024-01-01", "UP", 0.75, 0.02)
        assert decision.risk_level in ["no_trade", "low", "medium", "high"]

    def test_kelly_fraction_computed(self, risk_agent):
        decision = risk_agent.decide("TEST", "2024-01-01", "UP", 0.75, 0.02)
        assert decision.kelly_fraction >= 0
