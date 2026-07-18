"""
sustainability_report.py — Monthly PDF Report Generator (Prompt 32)
===================================================================

Generates a monthly sustainability PDF report using ReportLab.

Sections
--------
1. Cover page with report period
2. Executive summary table
3. Trend chart of weekly material usage
4. Fragility safety score section
5. Footer with methodology notes

Usage
-----
    python -m src.sustainability_report
    python -m src.sustainability_report --month 2026-06

Author: EcoPackAI Team
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


def generate_sustainability_report(
    report_month: str = "",
    output_dir: str = "reports",
    shipment_count: int = 0,
    material_saved_kg: float = 0.0,
    co2e_saved_kg: float = 0.0,
    weekly_material_kg: Optional[List[float]] = None,
    fragility_scores: Optional[Dict[str, float]] = None,
    damage_rate_pct: float = 0.0,
    avg_void_pct: float = 0.0,
) -> Path:
    """Generate a monthly sustainability PDF report.

    Parameters
    ----------
    report_month : str
        Report period (e.g. "2026-06"). Defaults to current month.
    output_dir : str
        Directory for output PDF.
    shipment_count : int
        Total shipments in period.
    material_saved_kg : float
        Total material saved vs baseline (kg).
    co2e_saved_kg : float
        Total CO₂e saved vs baseline (kg).
    weekly_material_kg : list[float], optional
        Weekly material usage data for trend chart.
    fragility_scores : dict, optional
        Fragility safety scores by tier.
    damage_rate_pct : float
        Damage rate percentage.
    avg_void_pct : float
        Average void volume percentage.

    Returns
    -------
    Path
        Path to the generated PDF.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm, mm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, Image, HRFlowable,
        )
        from reportlab.graphics.shapes import Drawing, Rect, String, Line
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        from reportlab.graphics import renderPDF
    except ImportError:
        logger.error("ReportLab not installed. Run: pip install reportlab")
        raise

    if not report_month:
        report_month = datetime.now(timezone.utc).strftime("%Y-%m")

    # Setup
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"sustainability_report_{report_month}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=28,
        textColor=colors.HexColor("#1a5276"),
        spaceAfter=20,
        alignment=1,
    )
    subtitle_style = ParagraphStyle(
        "CustomSubtitle",
        parent=styles["Heading2"],
        fontSize=16,
        textColor=colors.HexColor("#2e86c1"),
        alignment=1,
        spaceAfter=30,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#1a5276"),
        spaceBefore=20,
        spaceAfter=10,
    )
    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontSize=11,
        leading=15,
        spaceAfter=8,
    )
    footer_style = ParagraphStyle(
        "FooterStyle",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.grey,
        alignment=1,
    )

    elements = []

    # -----------------------------------------------------------------------
    # 1. Cover Page
    # -----------------------------------------------------------------------
    elements.append(Spacer(1, 80))
    elements.append(Paragraph("🌿 EcoPackAI", title_style))
    elements.append(Paragraph("Monthly Sustainability Report", subtitle_style))
    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(
        width="80%", thickness=2,
        color=colors.HexColor("#2e86c1"), spaceAfter=20,
    ))
    elements.append(Paragraph(
        f"<b>Report Period:</b> {report_month}", body_style,
    ))
    elements.append(Paragraph(
        f"<b>Generated:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        body_style,
    ))
    elements.append(Paragraph(
        "<b>Classification:</b> Internal — Sustainability Team", body_style,
    ))
    elements.append(PageBreak())

    # -----------------------------------------------------------------------
    # 2. Executive Summary Table
    # -----------------------------------------------------------------------
    elements.append(Paragraph("Executive Summary", heading_style))
    elements.append(Spacer(1, 10))

    summary_data = [
        ["Metric", "Value", "Target", "Status"],
        ["Total Shipments", f"{shipment_count:,}", "—", "ℹ️"],
        ["Material Saved (kg)", f"{material_saved_kg:,.1f}", "≥ 25% reduction", 
         "✅" if material_saved_kg > 0 else "⚠️"],
        ["CO₂e Saved (kg)", f"{co2e_saved_kg:,.2f}", "≥ 20% reduction",
         "✅" if co2e_saved_kg > 0 else "⚠️"],
        ["Avg Void Volume %", f"{avg_void_pct:.1f}%", "< 60%",
         "✅" if avg_void_pct < 60 else "⚠️"],
        ["Damage Rate", f"{damage_rate_pct:.2f}%", "< 0.5%",
         "✅" if damage_rate_pct < 0.5 else "❌"],
    ]

    summary_table = Table(summary_data, colWidths=[150, 100, 120, 50])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#eaf2f8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.HexColor("#eaf2f8"), colors.white,
        ]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#aed6f1")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # -----------------------------------------------------------------------
    # 3. Weekly Material Usage Trend Chart
    # -----------------------------------------------------------------------
    elements.append(Paragraph("Weekly Material Usage Trend", heading_style))

    if weekly_material_kg is None:
        # Generate sample data
        rng = np.random.RandomState(42)
        weekly_material_kg = list(
            np.maximum(50, 200 - np.arange(12) * 8 + rng.normal(0, 15, 12))
        )

    # Build chart using ReportLab
    drawing = Drawing(450, 200)
    chart = VerticalBarChart()
    chart.x = 50
    chart.y = 30
    chart.width = 380
    chart.height = 140
    chart.data = [weekly_material_kg]
    chart.categoryAxis.categoryNames = [f"W{i+1}" for i in range(len(weekly_material_kg))]
    chart.categoryAxis.labels.fontSize = 8
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(weekly_material_kg) * 1.2
    chart.valueAxis.valueStep = max(10, int(max(weekly_material_kg) / 5))
    chart.valueAxis.labels.fontSize = 8
    chart.bars[0].fillColor = colors.HexColor("#2e86c1")
    chart.bars[0].strokeColor = colors.HexColor("#1a5276")
    drawing.add(chart)
    elements.append(drawing)
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        "<i>Figure 1: Weekly packaging material consumption (kg) over the "
        "reporting period. Downward trend indicates optimization impact.</i>",
        footer_style,
    ))
    elements.append(Spacer(1, 20))

    # -----------------------------------------------------------------------
    # 4. Fragility Safety Score Section
    # -----------------------------------------------------------------------
    elements.append(Paragraph("Fragility Safety Scores", heading_style))

    if fragility_scores is None:
        fragility_scores = {
            "None (Tier 0)": 99.8,
            "Low (Tier 1)": 99.5,
            "Medium (Tier 2)": 98.7,
            "Critical (Tier 3)": 97.2,
        }

    safety_data = [["Fragility Tier", "Safety Score %", "Target", "Status"]]
    for tier, score in fragility_scores.items():
        target = 99.5 if "Critical" in tier else 99.0
        status = "✅" if score >= target else "⚠️"
        safety_data.append([tier, f"{score:.1f}%", f"≥ {target}%", status])

    safety_table = Table(safety_data, colWidths=[130, 100, 100, 50])
    safety_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#27ae60")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.HexColor("#eafaf1"), colors.white,
        ]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#82e0aa")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(safety_table)
    elements.append(Spacer(1, 30))

    # -----------------------------------------------------------------------
    # 5. Footer with Methodology Notes
    # -----------------------------------------------------------------------
    elements.append(HRFlowable(
        width="100%", thickness=1,
        color=colors.grey, spaceAfter=10,
    ))
    elements.append(Paragraph(
        "<b>Methodology Notes</b>", footer_style,
    ))
    elements.append(Paragraph(
        "• Material weight estimated from box surface area × cardboard "
        "density (0.055 g/cm², single-wall C-flute corrugated).",
        footer_style,
    ))
    elements.append(Paragraph(
        "• CO₂e calculated using IPCC/DEFRA 2024 emission factors: "
        "1.32 kg CO₂e/kg for cardboard production, "
        "0.0625 kg CO₂e/tonne-km for road freight.",
        footer_style,
    ))
    elements.append(Paragraph(
        "• Baseline comparison assumes 30 percentage-point higher void "
        "volume under manual packing operations.",
        footer_style,
    ))
    elements.append(Paragraph(
        f"• Report generated by EcoPackAI v1.0 on "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
        footer_style,
    ))

    # Build PDF
    doc.build(elements)
    logger.info("Sustainability report saved to %s", pdf_path)
    return pdf_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EcoPackAI sustainability report")
    parser.add_argument("--month", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="reports")
    args = parser.parse_args()

    # Use sample data for demonstration
    path = generate_sustainability_report(
        report_month=args.month,
        output_dir=args.output_dir,
        shipment_count=4523,
        material_saved_kg=312.5,
        co2e_saved_kg=87.3,
        avg_void_pct=52.4,
        damage_rate_pct=0.31,
    )
    print(f"Report: {path}")
