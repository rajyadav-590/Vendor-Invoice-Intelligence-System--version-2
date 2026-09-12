"""
invoice_risk_scoring.py

Transparent invoice risk scoring system.

The final 0-100 risk score combines:

    1. ML anomaly score
    2. Dollar discrepancy
    3. Quantity discrepancy
    4. Freight percentage
    5. Vendor invoice deviation
    6. Vendor freight deviation

This is a decision-support score, NOT a probability of fraud.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ================================================================
# IMPORTS
# ================================================================

from data_loader import (  # noqa: E402
    load_invoice_purchase_summary,
    validate_invoice_dataset,
)

from feature_engineering import (  # noqa: E402
    build_features,
)


# ================================================================
# FILE PATHS
# ================================================================

ANOMALY_RESULTS_PATH = (
    RESULTS_DIR / "invoice_anomaly_results.csv"
)

RISK_RESULTS_PATH = (
    RESULTS_DIR / "invoice_risk_results.csv"
)

RISK_SUMMARY_PATH = (
    RESULTS_DIR / "invoice_risk_summary.csv"
)


# ================================================================
# RISK WEIGHTS
# ================================================================

RISK_WEIGHTS = {
    "anomaly": 40,
    "dollar_difference": 20,
    "quantity_difference": 15,
    "freight": 10,
    "vendor_invoice": 10,
    "vendor_freight": 5,
}


# ================================================================
# UTILITY
# ================================================================

def safe_numeric(
    series: pd.Series,
) -> pd.Series:

    return pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0)


def percentile_score(
    series: pd.Series,
) -> pd.Series:
    """
    Convert values into a 0-100 relative score.

    Higher value = higher risk.
    """

    values = safe_numeric(series)

    if len(values) <= 1:
        return pd.Series(
            0.0,
            index=series.index,
        )

    values = values.abs()

    ranks = values.rank(
        method="average",
        pct=True,
    )

    return (
        ranks * 100
    ).clip(
        0,
        100,
    )


# ================================================================
# COMPONENT SCORES
# ================================================================

def anomaly_component(
    df: pd.DataFrame,
) -> pd.Series:

    if "anomaly_score" not in df.columns:
        raise ValueError(
            "anomaly_score column is missing."
        )

    return safe_numeric(
        df["anomaly_score"]
    ).clip(
        0,
        100,
    )


def dollar_component(
    df: pd.DataFrame,
) -> pd.Series:

    column = (
        "dollar_difference_percentage"
    )

    if column not in df.columns:
        raise ValueError(
            f"{column} column is missing."
        )

    return percentile_score(
        df[column]
    )


def quantity_component(
    df: pd.DataFrame,
) -> pd.Series:

    column = (
        "quantity_difference_percentage"
    )

    if column not in df.columns:
        raise ValueError(
            f"{column} column is missing."
        )

    return percentile_score(
        df[column]
    )


def freight_component(
    df: pd.DataFrame,
) -> pd.Series:

    column = "freight_percentage"

    if column not in df.columns:
        raise ValueError(
            f"{column} column is missing."
        )

    return percentile_score(
        df[column]
    )


def vendor_invoice_component(
    df: pd.DataFrame,
) -> pd.Series:

    column = (
        "invoice_vs_vendor_average_percentage"
    )

    if column not in df.columns:
        raise ValueError(
            f"{column} column is missing."
        )

    return percentile_score(
        df[column]
    )


def vendor_freight_component(
    df: pd.DataFrame,
) -> pd.Series:

    column = (
        "freight_vs_vendor_average_percentage"
    )

    if column not in df.columns:
        raise ValueError(
            f"{column} column is missing."
        )

    return percentile_score(
        df[column]
    )


# ================================================================
# CALCULATE RISK
# ================================================================

def calculate_risk_score(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    # ------------------------------------------------------------
    # Component scores
    # ------------------------------------------------------------

    result["risk_anomaly_component"] = (
        anomaly_component(result)
    )

    result["risk_dollar_component"] = (
        dollar_component(result)
    )

    result["risk_quantity_component"] = (
        quantity_component(result)
    )

    result["risk_freight_component"] = (
        freight_component(result)
    )

    result["risk_vendor_invoice_component"] = (
        vendor_invoice_component(result)
    )

    result["risk_vendor_freight_component"] = (
        vendor_freight_component(result)
    )

    # ------------------------------------------------------------
    # Weighted contributions
    # ------------------------------------------------------------

    result["risk_anomaly_contribution"] = (
        result["risk_anomaly_component"]
        * RISK_WEIGHTS["anomaly"]
        / 100
    )

    result["risk_dollar_contribution"] = (
        result["risk_dollar_component"]
        * RISK_WEIGHTS["dollar_difference"]
        / 100
    )

    result["risk_quantity_contribution"] = (
        result["risk_quantity_component"]
        * RISK_WEIGHTS["quantity_difference"]
        / 100
    )

    result["risk_freight_contribution"] = (
        result["risk_freight_component"]
        * RISK_WEIGHTS["freight"]
        / 100
    )

    result["risk_vendor_invoice_contribution"] = (
        result["risk_vendor_invoice_component"]
        * RISK_WEIGHTS["vendor_invoice"]
        / 100
    )

    result["risk_vendor_freight_contribution"] = (
        result["risk_vendor_freight_component"]
        * RISK_WEIGHTS["vendor_freight"]
        / 100
    )

    # ------------------------------------------------------------
    # Final risk score
    # ------------------------------------------------------------

    result["risk_score"] = (
        result["risk_anomaly_contribution"]
        + result["risk_dollar_contribution"]
        + result["risk_quantity_contribution"]
        + result["risk_freight_contribution"]
        + result["risk_vendor_invoice_contribution"]
        + result["risk_vendor_freight_contribution"]
    )

    result["risk_score"] = (
        result["risk_score"]
        .clip(0, 100)
        .round(2)
    )

    return result


# ================================================================
# RISK LEVEL
# ================================================================

def assign_risk_level(
    score: pd.Series,
) -> pd.Series:

    return pd.Series(
        np.select(
            [
                score <= 30,
                score <= 60,
            ],
            [
                "Low",
                "Medium",
            ],
            default="High",
        ),
        index=score.index,
    )


# ================================================================
# RECOMMENDATION
# ================================================================

def assign_recommendation(
    risk_level: pd.Series,
) -> pd.Series:

    return pd.Series(
        np.select(
            [
                risk_level == "Low",
                risk_level == "Medium",
            ],
            [
                "Standard Processing",
                "Review Recommended",
            ],
            default="Manual Review Required",
        ),
        index=risk_level.index,
    )


# ================================================================
# RISK REASONS
# ================================================================

def generate_risk_reasons(
    row: pd.Series,
) -> str:
    """
    Generate explanations based on the strongest weighted
    contributions.

    This prevents the dashboard from displaying a reason whose
    actual contribution is negligible.
    """

    signals = [
        (
            "Unusual invoice pattern detected by ML",
            float(
                row.get(
                    "risk_anomaly_contribution",
                    0,
                )
            ),
        ),
        (
            "Invoice and purchase dollar values differ",
            float(
                row.get(
                    "risk_dollar_contribution",
                    0,
                )
            ),
        ),
        (
            "Invoice and purchase quantities differ",
            float(
                row.get(
                    "risk_quantity_contribution",
                    0,
                )
            ),
        ),
        (
            "Freight behavior is unusual",
            float(
                row.get(
                    "risk_freight_contribution",
                    0,
                )
            ),
        ),
        (
            "Invoice value differs from vendor's typical value",
            float(
                row.get(
                    "risk_vendor_invoice_contribution",
                    0,
                )
            ),
        ),
        (
            "Freight differs from vendor's typical freight",
            float(
                row.get(
                    "risk_vendor_freight_contribution",
                    0,
                )
            ),
        ),
    ]

    # Keep only meaningful contributors.
    signals = [
        item
        for item in signals
        if item[1] >= 2
    ]

    signals.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    if not signals:
        return (
            "No major risk indicators detected"
        )

    return " | ".join(
        signal[0]
        for signal in signals[:4]
    )


# ================================================================
# FINALIZE
# ================================================================

def finalize_risk_output(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    result["risk_level"] = (
        assign_risk_level(
            result["risk_score"]
        )
    )

    result["recommendation"] = (
        assign_recommendation(
            result["risk_level"]
        )
    )

    result["risk_reasons"] = result.apply(
        generate_risk_reasons,
        axis=1,
    )

    return result


# ================================================================
# SUMMARY
# ================================================================

def create_risk_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    total = len(df)

    low = int(
        (
            df["risk_level"] == "Low"
        ).sum()
    )

    medium = int(
        (
            df["risk_level"] == "Medium"
        ).sum()
    )

    high = int(
        (
            df["risk_level"] == "High"
        ).sum()
    )

    return pd.DataFrame(
        {
            "Metric": [
                "Total Invoices",
                "Low Risk Invoices",
                "Medium Risk Invoices",
                "High Risk Invoices",
                "Low Risk Percentage",
                "Medium Risk Percentage",
                "High Risk Percentage",
                "Average Risk Score",
                "Maximum Risk Score",
            ],
            "Value": [
                total,
                low,
                medium,
                high,
                low / total * 100
                if total else 0,
                medium / total * 100
                if total else 0,
                high / total * 100
                if total else 0,
                df["risk_score"].mean(),
                df["risk_score"].max(),
            ],
        }
    )


# ================================================================
# LOAD ANOMALY RESULTS
# ================================================================

def load_anomaly_results() -> pd.DataFrame:

    if not ANOMALY_RESULTS_PATH.exists():

        raise FileNotFoundError(
            "Anomaly results do not exist.\n\n"
            "Run:\n"
            "python src\\train_anomaly.py"
        )

    df = pd.read_csv(
        ANOMALY_RESULTS_PATH
    )

    if df.empty:

        raise ValueError(
            "Anomaly results file is empty."
        )

    return df


# ================================================================
# MAIN
# ================================================================

def main() -> None:

    print("=" * 70)
    print(
        "INVOICE RISK SCORING"
    )
    print("=" * 70)

    # ------------------------------------------------------------
    # Load anomaly results
    # ------------------------------------------------------------

    print(
        "\nLoading anomaly results..."
    )

    df = load_anomaly_results()

    print(
        f"Records loaded: {len(df):,}"
    )

    # ------------------------------------------------------------
    # Calculate scores
    # ------------------------------------------------------------

    print(
        "\nCalculating risk components..."
    )

    risk_df = calculate_risk_score(
        df
    )

    risk_df = finalize_risk_output(
        risk_df
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    summary = create_risk_summary(
        risk_df
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "RISK SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Top risky invoices
    # ------------------------------------------------------------

    display_columns = [
        "VendorName",
        "PONumber",
        "invoice_dollars",
        "Freight",
        "anomaly_score",
        "risk_score",
        "risk_level",
        "recommendation",
        "risk_reasons",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in risk_df.columns
    ]

    print(
        "\nTop 20 risky invoices:"
    )

    print(
        risk_df
        .sort_values(
            "risk_score",
            ascending=False,
        )
        .head(20)[
            available_columns
        ]
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    risk_df.to_csv(
        RISK_RESULTS_PATH,
        index=False,
    )

    summary.to_csv(
        RISK_SUMMARY_PATH,
        index=False,
    )

    # ------------------------------------------------------------
    # Finished
    # ------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RISK SCORING COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"\nRisk results:\n{RISK_RESULTS_PATH}"
    )

    print(
        f"\nRisk summary:\n{RISK_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()