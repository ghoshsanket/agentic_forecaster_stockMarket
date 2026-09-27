"""Report Agent — fifth agent in the five-agent workflow.

Produces per-ticker HTML (and optionally PDF) reports combining:
  - run ID, ticker, origin date, target date, latest close
  - raw p(up), calibrated p(up), direction, confidence, confidence level
  - RSI, MACD, MACD signal, volatility, ATR
  - stop-loss PRICE, take-profit PRICE, RRR, RiskScore
  - top SHAP evidence, top attention dates, reason codes
  - narrative, model ID, config ID, reconstruction disclaimer
"""

from __future__ import annotations

import html
import logging
import uuid
from pathlib import Path

from agentic_forecaster.utils import ensure_dir

logger = logging.getLogger("agentic_forecaster.agents.report")

DISCLAIMER = (
    "This report is a software reconstruction of a research paper. "
    "It does not constitute investment advice. Past performance does not "
    "guarantee future results."
)


class ReportAgent:
    def __init__(self, config: dict):
        self.config = config
        self.rep_cfg = config.get("reporting", {})

    def run(
        self,
        ticker: str,
        origin_date: str,
        target_date: str,
        latest_close: float,
        p_up_raw: float,
        p_up_calibrated: float,
        direction: str,
        confidence: float,
        confidence_level: str,
        explanation: dict,
        risk_decision,
        metrics: dict,
        technical_indicators: dict,
        model_id: str = "",
        config_id: str = "",
        run_id: str | None = None,
        output_dir: str | Path = ".",
    ) -> dict:
        output_dir = ensure_dir(output_dir)
        safe_ticker = ticker.replace("/", "_")
        run_id = run_id or str(uuid.uuid4())[:8]

        narrative = explanation.get("llm_narrative", "")
        top_features = explanation.get("top_features", [])
        attention_evidence = explanation.get("attention_evidence", [])
        reason_codes = explanation.get("reason_codes", [])
        shap_method = (
            explanation.get("attribution_method")
            or explanation.get("shap_method")
            or "unknown"
        )

        feat_rows = "".join(
            f"<tr><td>{html.escape(f['feature'])}</td>"
            f"<td>{f['mean_abs_shap']:.4f}</td></tr>"
            for f in top_features
        )
        att_rows = "".join(
            f"<tr><td>day -{html.escape(str(a['day_offset']))}</td>"
            f"<td>{a['attention_weight']:.4f}</td></tr>"
            for a in (attention_evidence or [])
        )
        rc_items = "".join(f"<li>{html.escape(c)}</li>" for c in reason_codes)
        ti_rows = "".join(
            f"<tr><td>{html.escape(str(k))}</td><td>{v:.4f}</td></tr>"
            for k, v in technical_indicators.items()
        )
        metric_items = "".join(
            f'<div class="metric"><strong>{html.escape(k)}</strong>: {v:.4f}</div>'
            for k, v in metrics.items()
        )

        risk = risk_decision

        html_doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Forecast Report — {html.escape(ticker)}</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; color: #222; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: .5rem; }}
.metric {{ display: inline-block; margin: .5rem 1rem .5rem 0; }}
.badge {{ padding: .25rem .75rem; border-radius: 4px; color: #fff; }}
.HIGH {{ background: #c0392b; }} .MEDIUM {{ background: #e67e22; }}
.LOW {{ background: #27ae60; }}
table {{ border-collapse: collapse; margin-top: 1rem; }}
td, th {{ border: 1px solid #ccc; padding: .4rem .8rem; text-align: left; }}
.footer {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #ccc; font-size: .85rem; color: #666; }}
</style></head><body>
<h1>Forecast Report — {html.escape(ticker)}</h1>
<p><strong>Run ID:</strong> {html.escape(run_id)} &nbsp;
<strong>Model ID:</strong> {html.escape(model_id)} &nbsp;
<strong>Config ID:</strong> {html.escape(config_id)}</p>
<p><strong>Origin date:</strong> {html.escape(str(origin_date))} &nbsp;
<strong>Target date:</strong> {html.escape(str(target_date))}</p>
<p><strong>Latest close:</strong> {latest_close:.2f}</p>
<p><strong>Raw p(up):</strong> {p_up_raw:.4f} &nbsp;
<strong>Calibrated p(up):</strong> {p_up_calibrated:.4f}</p>
<p><strong>Direction:</strong> {html.escape(direction)} &nbsp;
<strong>Confidence:</strong> {confidence:.1%} &nbsp;
<strong>Confidence level:</strong>
<span class="badge {risk.confidence_level}">{risk.confidence_level}</span></p>

<h2>Technical Indicators</h2>
<table><tr><th>Indicator</th><th>Value</th></tr>{ti_rows}</table>

<h2>Risk Assessment (ATR-based)</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>ATR</td><td>{risk.atr:.4f}</td></tr>
<tr><td>Stop-loss price</td><td>{risk.stop_loss:.2f}</td></tr>
<tr><td>Take-profit price</td><td>{risk.take_profit:.2f}</td></tr>
<tr><td>RRR</td><td>{risk.rrr:.4f}</td></tr>
<tr><td>RiskScore</td><td>{risk.risk_score:.6f}</td></tr>
<tr><td>lambda_SL</td><td>{risk.lambda_sl}</td></tr>
<tr><td>lambda_TP</td><td>{risk.lambda_tp}</td></tr>
</table>

<h2>Explanation</h2>
<p>{html.escape(narrative)}</p>
<p><strong>SHAP method:</strong> {html.escape(shap_method)}</p>

<h2>Top features (SHAP)</h2>
<table><tr><th>Feature</th><th>Mean |SHAP|</th></tr>{feat_rows}</table>

<h2>Top attention dates</h2>
<table><tr><th>Day offset</th><th>Attention weight</th></tr>{att_rows}</table>

<h2>Reason codes</h2><ul>{rc_items}</ul>

<h2>Model metrics (test split)</h2>
{metric_items}

<div class="footer">
<p>{html.escape(DISCLAIMER)}</p>
</div>
</body></html>"""

        html_path = output_dir / f"{safe_ticker}_{str(origin_date)[:10]}.html"
        html_path.write_text(html_doc)

        result = {"html": str(html_path), "pdf": None, "run_id": run_id}
        if "pdf" in self.rep_cfg.get("formats", ["html"]):
            pdf_path = self._render_pdf(html_doc, output_dir / f"{safe_ticker}_{str(origin_date)[:10]}.pdf")
            result["pdf"] = str(pdf_path)
        return result

    def _render_pdf(self, html_doc: str, path: Path) -> Path:
        try:
            import re

            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas

            c = canvas.Canvas(str(path), pagesize=A4)
            c.setFont("Helvetica", 10)
            y = 800
            for line in html_doc.splitlines():
                text = re.sub(r"<[^>]+>", "", line.strip())
                if not text:
                    continue
                if y < 50:
                    c.showPage()
                    y = 800
                c.drawString(40, y, text[:120])
                y -= 14
            c.save()
            return path
        except Exception as exc:
            logger.warning("PDF rendering failed: %s", exc)
            return path
