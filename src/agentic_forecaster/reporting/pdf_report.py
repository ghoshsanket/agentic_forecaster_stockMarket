"""PDF report generation using ReportLab."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("agentic_forecaster.reporting.pdf")


class PDFReportGenerator:
    """Generates a PDF research report using ReportLab.

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
        """Generate a PDF research report.

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
            Where to write the PDF file. If None, a default name is used.

        Returns
        -------
        Path
            Path to the generated PDF file.
        """
        technical_indicators = technical_indicators or {}
        calibration = calibration or {}
        reason_codes = reason_codes or []

        narrative = explanation.get("llm_narrative", "")
        top_features = explanation.get("top_features", [])
        attention_evidence = explanation.get("attention_evidence")
        shap_summary = explanation.get("shap_summary", {})

        if output_path is None:
            output_path = Path(f"{ticker.replace('/', '_')}_{str(date)[:10]}_report.pdf")
        output_path = Path(output_path)

        from reportlab.lib import colors
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            ListFlowable,
            ListItem,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=LETTER,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=18,
            spaceAfter=12,
            textColor=colors.HexColor("#2c3e50"),
        )
        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=styles["Heading2"],
            fontSize=14,
            spaceAfter=8,
            textColor=colors.HexColor("#34495e"),
        )
        body_style = ParagraphStyle(
            "CustomBody",
            parent=styles["Normal"],
            fontSize=10,
            spaceAfter=6,
        )

        story = []

        story.append(Paragraph(f"Forecast Report — {ticker}", title_style))
        story.append(Paragraph(f"<b>Date:</b> {date}", body_style))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Forecast Summary", heading_style))
        story.append(Paragraph(f"<b>Direction:</b> {direction}", body_style))
        story.append(Paragraph(f"<b>Conviction:</b> {conviction * 100:.1f}%", body_style))
        story.append(Paragraph(f"<b>Risk Level:</b> {risk_decision.risk_level}", body_style))
        story.append(Paragraph(f"<b>Position Size:</b> {risk_decision.position_size:.2%}", body_style))
        story.append(Paragraph(
            f"<b>Stop Loss:</b> {risk_decision.stop_loss:.2%} &nbsp; "
            f"<b>Take Profit:</b> {risk_decision.take_profit:.2%}",
            body_style,
        ))
        story.append(Spacer(1, 0.2 * inch))

        if technical_indicators:
            story.append(Paragraph("Technical Indicators", heading_style))
            ti_data = [["Indicator", "Value"]]
            for k, v in technical_indicators.items():
                ti_data.append([str(k), str(v)])
            ti_table = Table(ti_data, colWidths=[3 * inch, 3 * inch])
            ti_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f2f6")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(ti_table)
            story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Calibration & Confidence", heading_style))
        if calibration:
            cal_data = [["Metric", "Value"]]
            for k, v in calibration.items():
                cal_data.append([str(k), str(v)])
            cal_table = Table(cal_data, colWidths=[3 * inch, 3 * inch])
            cal_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f2f6")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(cal_table)
            story.append(Spacer(1, 0.1 * inch))

        story.append(Paragraph("<b>Model Metrics (Test Split):</b>", body_style))
        for k, v in metrics.items():
            story.append(Paragraph(f"{k}: {v:.4f}", body_style))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Risk Assessment", heading_style))
        risk_data = [
            ["Parameter", "Value"],
            ["Risk Level", risk_decision.risk_level],
            ["Position Size", f"{risk_decision.position_size:.2%}"],
            ["Kelly Fraction", f"{risk_decision.kelly_fraction:.4f}"],
            ["Stop Loss", f"{risk_decision.stop_loss:.2%}"],
            ["Take Profit", f"{risk_decision.take_profit:.2%}"],
        ]
        risk_table = Table(risk_data, colWidths=[3 * inch, 3 * inch])
        risk_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f2f6")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(risk_table)
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("SHAP Evidence", heading_style))
        if top_features:
            n_samples = shap_summary.get("n_samples", "N/A")
            story.append(Paragraph(
                f"Top features by mean |SHAP| value (computed over {n_samples} samples):",
                body_style,
            ))
            shap_data = [["Feature", "Mean |SHAP|"]]
            for f in top_features:
                shap_data.append([f["feature"], f"{f['mean_abs_shap']:.4f}"])
            shap_table = Table(shap_data, colWidths=[3 * inch, 3 * inch])
            shap_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f2f6")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(shap_table)
        else:
            story.append(Paragraph("No SHAP feature attributions available.", body_style))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Attention Evidence", heading_style))
        if attention_evidence is not None:
            if isinstance(attention_evidence, list):
                att_data = [["Date", "Attention Weight"]]
                for item in attention_evidence:
                    if isinstance(item, dict):
                        att_data.append([
                            str(item.get("date", "N/A")),
                            str(item.get("weight", item.get("attention", "N/A"))),
                        ])
                att_table = Table(att_data, colWidths=[3 * inch, 3 * inch])
                att_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f2f6")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(att_table)
            else:
                story.append(Paragraph(str(attention_evidence), body_style))
        else:
            story.append(Paragraph("No attention evidence available.", body_style))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Reason Codes", heading_style))
        if reason_codes:
            items = [ListItem(Paragraph(code, body_style)) for code in reason_codes]
            story.append(ListFlowable(items, bulletType="bullet"))
        else:
            story.append(Paragraph("No reason codes generated.", body_style))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Explanation", heading_style))
        story.append(Paragraph(narrative, body_style))
        story.append(Spacer(1, 0.3 * inch))

        story.append(Paragraph("Methodology & Reconstruction", heading_style))
        story.append(Paragraph(
            "This report was generated by the Agentic Forecaster pipeline. The model uses an "
            "LSTM with attention mechanism trained on walk-forward validated splits. Predictions "
            "are calibrated via temperature scaling. Feature importance is computed using SHAP "
            "values against a background sample of training data. Attention weights indicate "
            "which time steps in the lookback window most influenced the output.",
            body_style,
        ))
        story.append(Paragraph(
            "Risk sizing uses a half-Kelly criterion with volatility targeting. All metrics "
            "are computed on the held-out test split. This report is for research purposes only "
            "and does not constitute financial advice.",
            body_style,
        ))

        doc.build(story)

        self._validate_pdf(output_path)
        logger.info("PDF report written to %s", output_path)
        return output_path

    def _validate_pdf(self, path: Path) -> None:
        """Validate that the PDF file exists, is nonempty, and has a valid header.

        Parameters
        ----------
        path : Path
            Path to the generated PDF file.

        Raises
        ------
        ValueError
            If the file does not exist, is empty, or lacks a valid PDF header.
        """
        if not path.exists():
            raise ValueError(f"PDF file was not created: {path}")
        size = path.stat().st_size
        if size == 0:
            raise ValueError(f"PDF file is empty: {path}")
        with open(path, "rb") as f:
            header = f.read(5)
        if header != b"%PDF-":
            raise ValueError(f"Invalid PDF header in {path}: {header!r}")
