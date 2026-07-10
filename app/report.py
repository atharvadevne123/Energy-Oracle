"""PDF report generation for Energy-Oracle monitoring summaries."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["generate_monitoring_report"]


def generate_monitoring_report(
    summary: dict[str, Any],
    drift_results: dict[str, Any],
    output_path: str | Path = "/tmp/energy_oracle_report.pdf",
) -> Path:
    """
    Generate a PDF monitoring report with prediction stats and drift results.

    Args:
        summary: Output of monitoring.summarise_predictions().
        drift_results: Output of monitoring.run_drift_check().
        output_path: Where to write the PDF.

    Returns:
        Path to the generated PDF file.
    """
    from fpdf import FPDF

    output_path = Path(output_path)
    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_fill_color(20, 60, 30)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 14, "Energy-Oracle Monitoring Report", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {now}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Prediction summary
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Prediction Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for key, val in summary.items():
        pdf.cell(0, 7, f"  {key}: {val}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Drift results
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Drift Detection", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    status = drift_results.get("status", "unknown")
    pdf.cell(0, 7, f"  Status: {status}", new_x="LMARGIN", new_y="NEXT")
    for feat, stats in (drift_results.get("features") or {}).items():
        detected = stats.get("drift_detected", False)
        flag = "[DRIFT]" if detected else "[OK]"
        ks = stats.get("ks_statistic", 0)
        pdf.cell(
            0,
            6,
            f"  {flag} {feat}: KS={ks:.4f} p={stats.get('p_value', 1.0):.4f}",
            new_x="LMARGIN",
            new_y="NEXT",
        )

    pdf.output(str(output_path))
    logger.info("Report written to %s", output_path)
    return output_path
