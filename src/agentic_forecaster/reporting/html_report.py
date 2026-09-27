"""HTML report generation for forecast research reports."""

from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("agentic_forecaster.reporting.html")


class HTMLReportGenerator:
    """Generates a clean standalone HTML research report.

    The report includes forecast summary, technical indicators,
    calibration/confidence metrics, risk assessment, SHAP evidence,
    attention evidence, reason codes, explanation, and a
    methodology/reconstruction footer.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def generate(
        self,
        ticker: str,
        date: str,
        direction: str,
        conviction: float,
        explanation: dict[str, Any],
        risk_decision: Any,
        metrics: dict[str, float],
        technical_indicators: dict[str, Any] | None = None,
        calibration: dict[str, Any] | None = None,
        reason_codes: list[str] | None = None,
        output_path: str | Path | None = None,
    ) -> Path:
        """Generate a standalone HTML research report.

        Parameters
        ----------
        ticker : str
            The ticker symbol for the forecast.
        date : str
            The forecast date.
        direction : str
            Predicted direction (e.g., "up", "down").
        conviction : float
            Calibrated conviction probability (0-1).
        explanation : dict
            Explanation bundle from the ExplainerAgent containing
            top_features, attention_evidence, llm_narrative, etc.
        risk_decision : RiskDecision
            The risk decision object from the RiskAgent.
        metrics : dict
            Model performance metrics (e.g., brier, ece, precision_at_3).
        technical_indicators : dict, optional
            Technical indicator values (RSI, MACD, moving averages, etc.).
        calibration : dict, optional
            Calibration data (ECE, reliability bins, temperature).
        reason_codes : list of str, optional
            Reason codes explaining the model's decision.
        output_path : str or Path, optional
            Where to write the HTML file. If None, returns the HTML string
            wrapped in a temporary file path.

        Returns
        -------
        Path
            Path to the generated HTML file.
        """
        technical_indicators = technical_indicators or {}
        calibration = calibration or {}
        reason_codes = reason_codes or []

        narrative = explanation.get("llm_narrative", "")
        top_features = explanation.get("top_features", [])
        attention_evidence = explanation.get("attention_evidence")
        shap_summary = explanation.get("shap_summary", {})

        conviction_pct = f"{conviction * 100:.1f}%"
        risk = risk_decision

        sections = [
            self._header(ticker, date),
            self._forecast_summary(direction, conviction_pct, risk),
            self._technical_indicators(technical_indicators),
            self._calibration_section(calibration, metrics),
            self._risk_section(risk),
            self._shap_evidence(top_features, shap_summary),
            self._attention_evidence(attention_evidence),
            self._reason_codes(reason_codes),
            self._explanation(narrative),
            self._methodology_footer(),
        ]

        html_doc = "\n".join(sections)

        if output_path is None:
            output_path = Path(f"{ticker.replace('/', '_')}_{str(date)[:10]}_report.html")
        output_path = Path(output_path)
        output_path.write_text(html_doc, encoding="utf-8")
        logger.info("HTML report written to %s", output_path)
        return output_path

    def _header(self, ticker: str, date: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Forecast Report — {html.escape(ticker)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.6; }}
h1 {{ border-bottom: 2px solid #2c3e50; padding-bottom: .5rem; color: #2c3e50; }}
h2 {{ color: #34495e; margin-top: 2rem; border-bottom: 1px solid #eee; padding-bottom: .3rem; }}
.metric {{ display: inline-block; margin: .5rem 1rem .5rem 0; padding: .5rem 1rem; background: #f8f9fa; border-radius: 4px; }}
.badge {{ padding: .25rem .75rem; border-radius: 4px; color: #fff; font-weight: 600; }}
.high {{ background: #c0392b; }}
.medium {{ background: #e67e22; }}
.low {{ background: #27ae60; }}
.no_trade {{ background: #7f8c8d; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
td, th {{ border: 1px solid #ddd; padding: .5rem .8rem; text-align: left; }}
th {{ background: #f1f2f6; }}
.footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #ddd; font-size: .85rem; color: #666; }}
</style>
</head>
<body>
<h1>Forecast Report — {html.escape(ticker)}</h1>
<p><strong>Date:</strong> {html.escape(str(date))}</p>"""

    def _forecast_summary(self, direction: str, conviction_pct: str, risk: Any) -> str:
        return f"""
<h2>Forecast Summary</h2>
<p><strong>Direction:</strong> {html.escape(direction)}</p>
<p><strong>Conviction:</strong> {conviction_pct}</p>
<p><strong>Risk Level:</strong> <span class="badge {html.escape(risk.risk_level)}">{html.escape(risk.risk_level)}</span></p>
<p><strong>Position Size:</strong> {risk.position_size:.2%}</p>
<p><strong>Stop Loss:</strong> {risk.stop_loss:.2%} &nbsp; <strong>Take Profit:</strong> {risk.take_profit:.2%}</p>"""

    def _technical_indicators(self, indicators: dict[str, Any]) -> str:
        if not indicators:
            return """
<h2>Technical Indicators</h2>
<p>No technical indicators available.</p>"""
        rows = "".join(
            f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
            for k, v in indicators.items()
        )
        return f"""
<h2>Technical Indicators</h2>
<table>
<tr><th>Indicator</th><th>Value</th></tr>
{rows}
</table>"""

    def _calibration_section(self, calibration: dict[str, Any], metrics: dict[str, float]) -> str:
        cal_rows = ""
        if calibration:
            cal_rows = "".join(
                f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
                for k, v in calibration.items()
            )
            cal_table = f"""
<h3>Calibration</h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{cal_rows}
</table>"""
        else:
            cal_table = ""

        metric_items = "".join(
            f'<div class="metric"><strong>{html.escape(k)}</strong>: {v:.4f}</div>'
            for k, v in metrics.items()
        )
        return f"""
<h2>Calibration &amp; Confidence</h2>
{cal_table}
<h3>Model Metrics (Test Split)</h3>
{metric_items}"""

    def _risk_section(self, risk: Any) -> str:
        return f"""
<h2>Risk Assessment</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Risk Level</td><td><span class="badge {html.escape(risk.risk_level)}">{html.escape(risk.risk_level)}</span></td></tr>
<tr><td>Position Size</td><td>{risk.position_size:.2%}</td></tr>
<tr><td>Kelly Fraction</td><td>{risk.kelly_fraction:.4f}</td></tr>
<tr><td>Stop Loss</td><td>{risk.stop_loss:.2%}</td></tr>
<tr><td>Take Profit</td><td>{risk.take_profit:.2%}</td></tr>
</table>"""

    def _shap_evidence(self, top_features: list[dict[str, Any]], shap_summary: dict[str, Any]) -> str:
        if not top_features:
            return """
<h2>SHAP Evidence</h2>
<p>No SHAP feature attributions available.</p>"""
        feat_rows = "".join(
            f"<tr><td>{html.escape(f['feature'])}</td><td>{f['mean_abs_shap']:.4f}</td></tr>"
            for f in top_features
        )
        n_samples = shap_summary.get("n_samples", "N/A")
        return f"""
<h2>SHAP Evidence</h2>
<p>Top features by mean |SHAP| value (computed over {html.escape(str(n_samples))} samples):</p>
<table>
<tr><th>Feature</th><th>Mean |SHAP|</th></tr>
{feat_rows}
</table>"""

    def _attention_evidence(self, attention_evidence: Any) -> str:
        if attention_evidence is None:
            return """
<h2>Attention Evidence</h2>
<p>No attention evidence available.</p>"""
        if isinstance(attention_evidence, list):
            rows = "".join(
                f"<tr><td>{html.escape(str(item.get('date', 'N/A')))}</td>"
                f"<td>{html.escape(str(item.get('weight', item.get('attention', 'N/A'))))}</td></tr>"
                for item in attention_evidence
                if isinstance(item, dict)
            )
            return f"""
<h2>Attention Evidence</h2>
<table>
<tr><th>Date</th><th>Attention Weight</th></tr>
{rows}
</table>"""
        return f"""
<h2>Attention Evidence</h2>
<p>{html.escape(str(attention_evidence))}</p>"""

    def _reason_codes(self, reason_codes: list[str]) -> str:
        if not reason_codes:
            return """
<h2>Reason Codes</h2>
<p>No reason codes generated.</p>"""
        items = "".join(f"<li>{html.escape(code)}</li>" for code in reason_codes)
        return f"""
<h2>Reason Codes</h2>
<ul>
{items}
</ul>"""

    def _explanation(self, narrative: str) -> str:
        return f"""
<h2>Explanation</h2>
<p>{html.escape(narrative)}</p>"""

    def _methodology_footer(self) -> str:
        return """
<div class="footer">
<h2>Methodology &amp; Reconstruction</h2>
<p>This report was generated by the Agentic Forecaster pipeline. The model uses an
LSTM with attention mechanism trained on walk-forward validated splits. Predictions
are calibrated via temperature scaling. Feature importance is computed using SHAP
values against a background sample of training data. Attention weights indicate
which time steps in the lookback window most influenced the output.</p>
<p>Risk sizing uses a half-Kelly criterion with volatility targeting. All metrics
are computed on the held-out test split. This report is for research purposes only
and does not constitute financial advice.</p>
</div>
</body>
</html>"""
