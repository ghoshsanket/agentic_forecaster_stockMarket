"""Report Agent — fifth agent in the five-agent workflow.

Produces per-ticker HTML (and optionally PDF) reports combining:
  - prediction + calibrated conviction
  - SHAP / attention explanation
  - risk decision
  - metric summary
"""

from __future__ import annotations

import html
import logging
from pathlib import Path

from agentic_forecaster.utils import ensure_dir

logger = logging.getLogger("agentic_forecaster.agents.report")


class ReportAgent:
    def __init__(self, config: dict):
        self.config = config
        self.rep_cfg = config.get("reporting", {})
        self.risk_cfg = config.get("risk", {})

    def run(
        self,
        ticker: str,
        date: str,
        direction: str,
        conviction: float,
        explanation: dict,
        risk_decision,
        metrics: dict,
        output_dir: str | Path,
    ) -> dict:
        output_dir = ensure_dir(output_dir)
        safe_ticker = ticker.replace("/", "_")

        narrative = explanation.get("llm_narrative", "")
        top_features = explanation.get("top_features", [])
        feat_rows = "".join(
            f"<tr><td>{html.escape(f['feature'])}</td>"
            f"<td>{f['mean_abs_shap']:.4f}</td></tr>"
            for f in top_features
        )

        conviction_pct = f"{conviction * 100:.1f}%"
        risk = risk_decision

        html_doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Forecast Report — {html.escape(ticker)}</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; color: #222; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: .5rem; }}
.metric {{ display: inline-block; margin: .5rem 1rem .5rem 0; }}
.badge {{ padding: .25rem .75rem; border-radius: 4px; color: #fff; }}
.high {{ background: #c0392b; }} .medium {{ background: #e67e22; }}
.low {{ background: #27ae60; }} .no_trade {{ background: #7f8c8d; }}
table {{ border-collapse: collapse; margin-top: 1rem; }}
td, th {{ border: 1px solid #ccc; padding: .4rem .8rem; text-align: left; }}
</style></head><body>
<h1>Forecast Report — {html.escape(ticker)}</h1>
<p><strong>Date:</strong> {html.escape(str(date))}</p>
<p><strong>Direction:</strong> {html.escape(direction)}</p>
<p><strong>Conviction:</strong> {conviction_pct}</p>
<p><strong>Risk level:</strong>
<span class="badge {risk.risk_level}">{risk.risk_level}</span></p>
<p><strong>Position size:</strong> {risk.position_size:.2%}</p>
<p><strong>Stop loss:</strong> {risk.stop_loss:.2%} &nbsp;
<strong>Take profit:</strong> {risk.take_profit:.2%}</p>
<h2>Explanation</h2>
<p>{html.escape(narrative)}</p>
<h2>Top features (SHAP)</h2>
<table><tr><th>Feature</th><th>Mean |SHAP|</th></tr>{feat_rows}</table>
<h2>Model metrics (test split)</h2>
"""
        for k, v in metrics.items():
            html_doc += f'<div class="metric"><strong>{html.escape(k)}</strong>: {v:.4f}</div>'
        html_doc += "</body></html>"

        html_path = output_dir / f"{safe_ticker}_{str(date)[:10]}.html"
        html_path.write_text(html_doc)

        result = {"html": str(html_path), "pdf": None}
        if "pdf" in self.rep_cfg.get("formats", ["html"]):
            pdf_path = self._render_pdf(html_doc, output_dir / f"{safe_ticker}_{str(date)[:10]}.pdf")
            result["pdf"] = str(pdf_path)
        return result

    def _render_pdf(self, html_doc: str, path: Path) -> Path:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas

            c = canvas.Canvas(str(path), pagesize=A4)
            c.setFont("Helvetica", 10)
            y = 800
            for line in html_doc.splitlines():
                text = line.strip()
                if not text:
                    continue
                import re
                text = re.sub(r"<[^>]+>", "", text)
                if y < 50:
                    c.showPage()
                    y = 800
                c.drawString(40, y, text[:120])
                y -= 14
            c.save()
            return path
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("PDF rendering failed: %s", exc)
            return path
