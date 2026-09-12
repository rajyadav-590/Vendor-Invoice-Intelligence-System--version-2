"""
report_generator.py

Generates a professional PDF risk report for an invoice.

The report uses the results produced by:
    - train_anomaly.py
    - invoice_risk_scoring.py

Output:
    reports/invoice_risk_report_<PO_NUMBER>.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd


# ================================================================
# PATH SETUP
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"

RESULTS_DIR = PROJECT_ROOT / "results"

REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ================================================================
# REPORTLAB
# ================================================================

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


# ================================================================
# INPUT FILE
# ================================================================

RISK_RESULTS_PATH = (
    RESULTS_DIR
    / "invoice_risk_results.csv"
)


# ================================================================
# FORMATTING
# ================================================================

def safe_value(
    value,
    default="N/A",
):
    """
    Safely convert missing values to readable text.
    """

    if pd.isna(value):
        return default

    return value


def currency(
    value,
) -> str:
    """
    Format a value as currency.
    """

    try:

        return f"${float(value):,.2f}"

    except (
        TypeError,
        ValueError,
    ):

        return "$0.00"


def number(
    value,
) -> str:
    """
    Format numeric value.
    """

    try:

        return f"{float(value):,.2f}"

    except (
        TypeError,
        ValueError,
    ):

        return "0.00"


def percentage(
    value,
) -> str:
    """
    Format percentage.
    """

    try:

        return f"{float(value):.2f}%"

    except (
        TypeError,
        ValueError,
    ):

        return "0.00%"


# ================================================================
# LOAD RESULTS
# ================================================================

def load_risk_results() -> pd.DataFrame:
    """
    Load invoice risk results.
    """

    if not RISK_RESULTS_PATH.exists():

        raise FileNotFoundError(
            "Risk results file was not found.\n\n"
            "Run:\n"
            "python src\\invoice_risk_scoring.py"
        )

    df = pd.read_csv(
        RISK_RESULTS_PATH
    )

    if df.empty:

        raise ValueError(
            "Risk results file is empty."
        )

    return df


# ================================================================
# FIND INVOICE
# ================================================================

def find_invoice(
    df: pd.DataFrame,
    po_number: str,
) -> pd.Series:
    """
    Find an invoice using its PO number.
    """

    if "PONumber" not in df.columns:

        raise ValueError(
            "PONumber column is missing."
        )

    matches = df[
        df["PONumber"]
        .astype(str)
        == str(po_number)
    ]

    if matches.empty:

        raise ValueError(
            f"No invoice found for PO number: "
            f"{po_number}"
        )

    return matches.iloc[0]


# ================================================================
# PDF HEADER / FOOTER
# ================================================================

def add_page_number(
    canvas,
    document,
):
    """
    Add page number and report footer.
    """

    canvas.saveState()

    width, height = A4

    canvas.setFont(
        "Helvetica",
        8,
    )

    canvas.drawCentredString(
        width / 2,
        10 * mm,
        f"Page {document.page}",
    )

    canvas.drawString(
        20 * mm,
        10 * mm,
        "Vendor Invoice Intelligence",
    )

    canvas.restoreState()


# ================================================================
# CREATE STYLES
# ================================================================

def create_styles():
    """
    Create PDF styles.
    """

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=20,
            leading=24,
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            alignment=TA_CENTER,
            fontSize=10,
            leading=14,
            spaceAfter=18,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            spaceBefore=12,
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="RiskHeading",
            parent=styles["Heading1"],
            alignment=TA_CENTER,
            fontSize=18,
            leading=22,
            spaceBefore=8,
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SmallText",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
        )
    )

    return styles


# ================================================================
# INFORMATION TABLE
# ================================================================

def create_information_table(
    data: list[list[str]],
):
    """
    Create a two-column information table.
    """

    table = Table(
        data,
        colWidths=[
            55 * mm,
            120 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.lightgrey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, -1),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (1, 0),
                    (1, -1),
                    "Helvetica",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    return table


# ================================================================
# RISK COMPONENT TABLE
# ================================================================

def create_risk_component_table(
    invoice: pd.Series,
):
    """
    Create detailed risk component table.
    """

    components = [
        (
            "Anomaly Detection",
            "risk_anomaly_component",
            "40%",
            "risk_anomaly_contribution",
        ),
        (
            "Dollar Difference",
            "risk_dollar_component",
            "20%",
            "risk_dollar_contribution",
        ),
        (
            "Quantity Difference",
            "risk_quantity_component",
            "15%",
            "risk_quantity_contribution",
        ),
        (
            "Freight",
            "risk_freight_component",
            "10%",
            "risk_freight_contribution",
        ),
        (
            "Vendor Invoice Deviation",
            "risk_vendor_invoice_component",
            "10%",
            "risk_vendor_invoice_contribution",
        ),
        (
            "Vendor Freight Deviation",
            "risk_vendor_freight_component",
            "5%",
            "risk_vendor_freight_contribution",
        ),
    ]

    data = [
        [
            "Risk Component",
            "Component Score",
            "Weight",
            "Contribution",
        ]
    ]

    for (
        name,
        score_column,
        weight,
        contribution_column,
    ) in components:

        data.append(
            [
                name,
                number(
                    invoice.get(
                        score_column,
                        0,
                    )
                ),
                weight,
                number(
                    invoice.get(
                        contribution_column,
                        0,
                    )
                ),
            ]
        )

    table = Table(
        data,
        colWidths=[
            70 * mm,
            35 * mm,
            25 * mm,
            40 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "ALIGN",
                    (1, 1),
                    (-1, -1),
                    "RIGHT",
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    return table


# ================================================================
# GENERATE PDF
# ================================================================

def generate_report(
    invoice: pd.Series,
    output_path: Path,
) -> None:
    """
    Generate the complete invoice PDF report.
    """

    styles = create_styles()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Invoice Risk Analysis Report",
        author="Vendor Invoice Intelligence",
    )

    story = []

    # ============================================================
    # TITLE
    # ============================================================

    story.append(
        Paragraph(
            "VENDOR INVOICE INTELLIGENCE",
            styles["ReportTitle"],
        )
    )

    story.append(
        Paragraph(
            "Invoice Risk Analysis Report",
            styles["ReportSubtitle"],
        )
    )

    # ============================================================
    # REPORT METADATA
    # ============================================================

    report_date = datetime.now().strftime(
        "%d %B %Y, %H:%M"
    )

    story.append(
        create_information_table(
            [
                [
                    "Report Generated",
                    report_date,
                ],
                [
                    "Purchase Order",
                    str(
                        safe_value(
                            invoice.get(
                                "PONumber",
                                "N/A",
                            )
                        )
                    ),
                ],
                [
                    "Invoice Date",
                    str(
                        safe_value(
                            invoice.get(
                                "InvoiceDate",
                                "N/A",
                            )
                        )
                    ),
                ],
            ]
        )
    )

    story.append(
        Spacer(
            1,
            10,
        )
    )

    # ============================================================
    # INVOICE INFORMATION
    # ============================================================

    story.append(
        Paragraph(
            "1. Invoice Information",
            styles["SectionHeading"],
        )
    )

    invoice_information = [
        [
            "Vendor Number",
            str(
                safe_value(
                    invoice.get(
                        "VendorNumber",
                        "N/A",
                    )
                )
            ),
        ],
        [
            "Vendor Name",
            str(
                safe_value(
                    invoice.get(
                        "VendorName",
                        "N/A",
                    )
                )
            ),
        ],
        [
            "Invoice Quantity",
            number(
                invoice.get(
                    "invoice_quantity",
                    0,
                )
            ),
        ],
        [
            "Invoice Value",
            currency(
                invoice.get(
                    "invoice_dollars",
                    0,
                )
            ),
        ],
        [
            "Freight",
            currency(
                invoice.get(
                    "Freight",
                    0,
                )
            ),
        ],
        [
            "Purchase Quantity",
            number(
                invoice.get(
                    "total_item_quantity",
                    0,
                )
            ),
        ],
        [
            "Purchase Value",
            currency(
                invoice.get(
                    "total_item_dollars",
                    0,
                )
            ),
        ],
    ]

    story.append(
        create_information_table(
            invoice_information
        )
    )

    # ============================================================
    # RISK ASSESSMENT
    # ============================================================

    story.append(
        Paragraph(
            "2. Risk Assessment",
            styles["SectionHeading"],
        )
    )

    risk_score = float(
        invoice.get(
            "risk_score",
            0,
        )
    )

    risk_level = str(
        safe_value(
            invoice.get(
                "risk_level",
                "Unknown",
            )
        )
    )

    recommendation = str(
        safe_value(
            invoice.get(
                "recommendation",
                "N/A",
            )
        )
    )

    anomaly_score = float(
        invoice.get(
            "anomaly_score",
            0,
        )
    )

    risk_table = Table(
        [
            [
                "Risk Score",
                "Risk Level",
                "Anomaly Score",
            ],
            [
                f"{risk_score:.2f}/100",
                risk_level,
                f"{anomaly_score:.2f}/100",
            ],
        ],
        colWidths=[
            55 * mm,
            55 * mm,
            55 * mm,
        ],
    )

    risk_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (-1, 1),
                    "Helvetica-Bold",
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
            ]
        )
    )

    story.append(
        risk_table
    )

    story.append(
        Spacer(
            1,
            8,
        )
    )

    # ============================================================
    # RECOMMENDATION
    # ============================================================

    story.append(
        Paragraph(
            f"<b>Recommendation:</b> "
            f"{recommendation}",
            styles["Normal"],
        )
    )

    story.append(
        Spacer(
            1,
            8,
        )
    )

    # ============================================================
    # RISK COMPONENTS
    # ============================================================

    story.append(
        Paragraph(
            "3. Risk Component Analysis",
            styles["SectionHeading"],
        )
    )

    story.append(
        create_risk_component_table(
            invoice
        )
    )

    # ============================================================
    # RISK REASONS
    # ============================================================

    story.append(
        Paragraph(
            "4. Risk Indicators",
            styles["SectionHeading"],
        )
    )

    reasons = str(
        safe_value(
            invoice.get(
                "risk_reasons",
                "No major risk indicators detected",
            )
        )
    )

    reason_list = [
        reason.strip()
        for reason in reasons.split("|")
        if reason.strip()
    ]

    if reason_list:

        for reason in reason_list:

            story.append(
                Paragraph(
                    f"• {reason}",
                    styles["Normal"],
                )
            )

            story.append(
                Spacer(
                    1,
                    3,
                )
            )

    else:

        story.append(
            Paragraph(
                "No major risk indicators detected.",
                styles["Normal"],
            )
        )

    # ============================================================
    # FINANCIAL INDICATORS
    # ============================================================

    story.append(
        Paragraph(
            "5. Financial Indicators",
            styles["SectionHeading"],
        )
    )

    financial_data = [
        [
            "Indicator",
            "Value",
        ],
        [
            "Dollar Difference",
            currency(
                invoice.get(
                    "dollar_difference",
                    0,
                )
            ),
        ],
        [
            "Dollar Difference %",
            percentage(
                invoice.get(
                    "dollar_difference_percentage",
                    0,
                )
            ),
        ],
        [
            "Quantity Difference",
            number(
                invoice.get(
                    "quantity_difference",
                    0,
                )
            ),
        ],
        [
            "Quantity Difference %",
            percentage(
                invoice.get(
                    "quantity_difference_percentage",
                    0,
                )
            ),
        ],
        [
            "Freight %",
            percentage(
                invoice.get(
                    "freight_percentage",
                    0,
                )
            ),
        ],
        [
            "Invoice Cost / Unit",
            currency(
                invoice.get(
                    "invoice_cost_per_unit",
                    0,
                )
            ),
        ],
        [
            "Purchase Cost / Unit",
            currency(
                invoice.get(
                    "purchase_cost_per_unit",
                    0,
                )
            ),
        ],
    ]

    financial_table = Table(
        financial_data,
        colWidths=[
            90 * mm,
            75 * mm,
        ],
    )

    financial_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "ALIGN",
                    (1, 1),
                    (1, -1),
                    "RIGHT",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(
        financial_table
    )

    # ============================================================
    # METHODOLOGY
    # ============================================================

    story.append(
        Paragraph(
            "6. Methodology",
            styles["SectionHeading"],
        )
    )

    methodology = (
        "The invoice risk score is a transparent decision-support "
        "score ranging from 0 to 100. It combines the machine "
        "learning anomaly score with invoice discrepancies, freight "
        "behavior, and vendor-level deviations. The model does not "
        "represent the score as a probability of fraud. Final "
        "business decisions should include appropriate human review."
    )

    story.append(
        Paragraph(
            methodology,
            styles["Normal"],
        )
    )

    # ============================================================
    # DISCLAIMER
    # ============================================================

    story.append(
        Spacer(
            1,
            15,
        )
    )

    story.append(
        Paragraph(
            "<b>Disclaimer:</b> This report is intended for "
            "analytical and decision-support purposes. Anomaly "
            "detection does not by itself establish fraud, error, "
            "or misconduct.",
            styles["SmallText"],
        )
    )

    # ============================================================
    # BUILD PDF
    # ============================================================

    doc.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print(
        "INVOICE RISK REPORT GENERATOR"
    )
    print("=" * 70)

    df = load_risk_results()

    print(
        f"\nRisk records available: {len(df):,}"
    )

    # ------------------------------------------------------------
    # Ask for PO
    # ------------------------------------------------------------

    po_number = input(
        "\nEnter Purchase Order Number: "
    ).strip()

    if not po_number:

        print(
            "No PO number entered."
        )

        return

    # ------------------------------------------------------------
    # Find invoice
    # ------------------------------------------------------------

    invoice = find_invoice(
        df,
        po_number,
    )

    # ------------------------------------------------------------
    # Output path
    # ------------------------------------------------------------

    safe_po = (
        str(po_number)
        .replace(
            "/",
            "_",
        )
        .replace(
            "\\",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )

    output_path = (
        REPORTS_DIR
        /
        f"invoice_risk_report_{safe_po}.pdf"
    )

    # ------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------

    generate_report(
        invoice,
        output_path,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "REPORT GENERATED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nPDF Report:\n{output_path}"
    )


if __name__ == "__main__":
    main()