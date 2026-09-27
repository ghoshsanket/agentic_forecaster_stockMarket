from pathlib import Path

import pytest

from agentic_forecaster.reporting import HTMLReportGenerator, PDFReportGenerator
from agentic_forecaster.risk import RiskAgent


@pytest.fixture
def sample_explanation():
    return {
        "top_features": [
            {"feature": "rsi_14", "mean_abs_shap": 0.15},
            {"feature": "macd", "mean_abs_shap": 0.12},
        ],
        "attention_evidence": [{"date": "2024-01-01", "weight": 0.05}],
        "llm_narrative": "Test explanation",
        "shap_summary": {"n_samples": 64},
    }


@pytest.fixture
def sample_metrics():
    return {"accuracy": 0.65, "f1": 0.58, "brier": 0.22, "ece": 0.05}


@pytest.fixture
def sample_risk_decision():
    agent = RiskAgent()
    return agent.decide("TEST", "2024-01-01", "UP", 0.75, 0.02)


class TestHTMLReport:
    def test_generates_html_file(self, tmp_path, sample_explanation, sample_metrics, sample_risk_decision):
        gen = HTMLReportGenerator()
        path = gen.generate(
            ticker="TEST",
            date="2024-01-30",
            direction="UP",
            conviction=0.75,
            explanation=sample_explanation,
            risk_decision=sample_risk_decision,
            metrics=sample_metrics,
            output_path=tmp_path / "test_report.html",
        )
        assert Path(path).exists()
        assert str(path).endswith(".html")

    def test_html_contains_title(self, tmp_path, sample_explanation, sample_metrics, sample_risk_decision):
        gen = HTMLReportGenerator()
        path = gen.generate(
            ticker="TEST",
            date="2024-01-30",
            direction="UP",
            conviction=0.75,
            explanation=sample_explanation,
            risk_decision=sample_risk_decision,
            metrics=sample_metrics,
            output_path=tmp_path / "test_report.html",
        )
        content = Path(path).read_text()
        assert "<html" in content.lower()


class TestPDFReport:
    def test_generates_pdf_file(self, tmp_path, sample_explanation, sample_metrics, sample_risk_decision):
        gen = PDFReportGenerator()
        path = gen.generate(
            ticker="TEST",
            date="2024-01-30",
            direction="UP",
            conviction=0.75,
            explanation=sample_explanation,
            risk_decision=sample_risk_decision,
            metrics=sample_metrics,
            output_path=tmp_path / "test_pdf.pdf",
        )
        assert Path(path).exists()
        assert str(path).endswith(".pdf")

    def test_pdf_contains_content(self, tmp_path, sample_explanation, sample_metrics, sample_risk_decision):
        gen = PDFReportGenerator()
        path = gen.generate(
            ticker="TEST",
            date="2024-01-30",
            direction="UP",
            conviction=0.75,
            explanation=sample_explanation,
            risk_decision=sample_risk_decision,
            metrics=sample_metrics,
            output_path=tmp_path / "test_pdf.pdf",
        )
        content = Path(path).read_bytes()
        assert len(content) > 0
        assert content[:5] == b"%PDF-"
