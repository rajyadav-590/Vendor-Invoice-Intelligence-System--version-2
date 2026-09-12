"""
Vendor Invoice Intelligence

AI-powered invoice analysis, anomaly detection,
freight prediction, vendor analytics and risk reporting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ================================================================
# PROJECT IMPORTS
# ================================================================

from data_loader import load_invoice_purchase_summary

from feature_engineering import (
    build_features,
    get_freight_features,
    get_anomaly_features,
)

from invoice_risk_scoring import (
    calculate_risk_score,
    finalize_risk_output,
)

from report_generator import generate_report

from risk_dashboard import (
    load_risk_results,
    calculate_risk_summary,
    get_risk_distribution,
    get_top_risky_invoices,
    get_top_risky_vendors,
)


# ================================================================
# STREAMLIT CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Vendor Invoice Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================================
# CUSTOM CSS
# ================================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# HEADER
# ================================================================

st.markdown(
    '<div class="main-title">'
    '📊 Vendor Invoice Intelligence'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered invoice analysis, anomaly detection, '
    'freight prediction, and vendor analytics'
    '</div>',
    unsafe_allow_html=True,
)


# ================================================================
# DATA LOADING
# ================================================================

@st.cache_data
def load_dashboard_data():

    raw_df = load_invoice_purchase_summary()

    feature_df = build_features(
        raw_df,
        create_target=False,
    )

    return feature_df


# ================================================================
# MODEL LOADING
# ================================================================

@st.cache_resource
def load_freight_model():

    model_path = (
        MODELS_DIR
        / "freight_prediction_pipeline.pkl"
    )

    if not model_path.exists():
        return None

    return joblib.load(model_path)


@st.cache_resource
def load_anomaly_model():

    model_path = (
        MODELS_DIR
        / "invoice_anomaly_pipeline.pkl"
    )

    if not model_path.exists():
        return None

    return joblib.load(model_path)


# ================================================================
# FORMATTING FUNCTIONS
# ================================================================

def format_currency(value):

    try:

        if pd.isna(value):
            return "$0.00"

        return f"${float(value):,.2f}"

    except Exception:
        return "$0.00"


def format_number(value):

    try:

        if pd.isna(value):
            return "0"

        return f"{float(value):,.0f}"

    except Exception:
        return "0"


def format_percentage(value):

    try:

        if pd.isna(value):
            return "0.00%"

        return f"{float(value):.2f}%"

    except Exception:
        return "0.00%"


# ================================================================
# RAW HISTORICAL COLUMNS
# ================================================================
#
# IMPORTANT:
#
# df is already feature-engineered.
# Therefore, we must NOT pass the complete df back into
# build_features().
#
# These are the original/raw columns used to rebuild historical
# features together with newly uploaded invoices.
#
# ================================================================

RAW_INVOICE_COLUMNS = [
    "VendorNumber",
    "VendorName",
    "InvoiceDate",
    "PONumber",
    "PODate",
    "PayDate",
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "Approval",
    "total_item_quantity",
    "total_item_dollars",
    "average_purchase_price",
    "number_of_purchase_lines",
    "number_of_brands",
]


def get_raw_historical_data(source_df):

    available_columns = [
        column
        for column in RAW_INVOICE_COLUMNS
        if column in source_df.columns
    ]

    return source_df[
        available_columns
    ].copy()


# ================================================================
# LOAD MAIN DATA
# ================================================================

try:

    df = load_dashboard_data()

except Exception as exc:

    st.error(
        "Unable to load invoice data."
    )

    st.exception(exc)

    st.stop()


# ================================================================
# LOAD RISK DATA
# ================================================================

risk_df = load_risk_results()

risk_summary = calculate_risk_summary(
    risk_df
)


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:

    st.header("Navigation")

    page = st.radio(
        "Select module",
        [
            "🏠 Dashboard",
            "🚚 Freight Prediction",
            "🚨 Invoice Risk",
            "🔍 Anomaly Detection",
            "🏢 Vendor Analytics",
            "📁 Batch Analysis",
            "📈 Model Performance",
            "ℹ️ About",
        ],
    )

    st.divider()

    st.caption(
        f"Invoices: {len(df):,}"
    )

    if "VendorNumber" in df.columns:

        st.caption(
            f"Vendors: "
            f"{df['VendorNumber'].nunique():,}"
        )


# ################################################################
# DASHBOARD
# ################################################################

if page == "🏠 Dashboard":

    st.header("Dashboard")

    st.write(
        "Overview of invoice activity, financial values, "
        "risk distribution and vendor behavior."
    )

    # ============================================================
    # BASIC KPIs
    # ============================================================

    total_invoices = len(df)

    total_vendors = (
        df["VendorNumber"].nunique()
        if "VendorNumber" in df.columns
        else 0
    )

    total_invoice_value = (
        df["invoice_dollars"].sum()
        if "invoice_dollars" in df.columns
        else 0
    )

    total_freight = (
        df["Freight"].sum()
        if "Freight" in df.columns
        else 0
    )

    average_invoice = (
        df["invoice_dollars"].mean()
        if "invoice_dollars" in df.columns
        else 0
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Total Invoices",
            f"{total_invoices:,}",
        )

    with col2:

        st.metric(
            "Vendors",
            f"{total_vendors:,}",
        )

    with col3:

        st.metric(
            "Total Invoice Value",
            format_currency(total_invoice_value),
        )

    with col4:

        st.metric(
            "Total Freight",
            format_currency(total_freight),
        )

    with col5:

        st.metric(
            "Average Invoice",
            format_currency(average_invoice),
        )

    st.divider()

    # ============================================================
    # RISK KPIs
    # ============================================================

    st.subheader(
        "⚠️ Invoice Risk Overview"
    )

    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

    with risk_col1:

        st.metric(
            "Low Risk",
            f"{risk_summary['low_risk']:,}",
        )

    with risk_col2:

        st.metric(
            "Medium Risk",
            f"{risk_summary['medium_risk']:,}",
        )

    with risk_col3:

        st.metric(
            "High Risk",
            f"{risk_summary['high_risk']:,}",
        )

    with risk_col4:

        st.metric(
            "Average Risk Score",
            f"{risk_summary['average_risk_score']:.2f}/100",
        )

    risk_col5, risk_col6, risk_col7 = st.columns(3)

    with risk_col5:

        st.metric(
            "High Risk %",
            format_percentage(
                risk_summary[
                    "high_risk_percentage"
                ]
            ),
        )

    with risk_col6:

        st.metric(
            "Medium + High %",
            format_percentage(
                risk_summary[
                    "medium_high_percentage"
                ]
            ),
        )

    with risk_col7:

        st.metric(
            "Risk Records",
            f"{risk_summary['total_invoices']:,}",
        )

    st.divider()

    # ============================================================
    # RISK DISTRIBUTION
    # ============================================================

    st.subheader(
        "📊 Risk Distribution"
    )

    risk_distribution = get_risk_distribution(
        risk_df
    )

    if not risk_distribution.empty:

        chart_data = (
            risk_distribution
            .set_index("Risk Level")
        )

        st.bar_chart(
            chart_data["Invoices"]
        )

    else:

        st.info(
            "Risk distribution data unavailable."
        )

    # ============================================================
    # TOP RISKY INVOICES
    # ============================================================

    st.subheader(
        "🚨 Top Risky Invoices"
    )

    top_invoices = get_top_risky_invoices(
        risk_df,
        limit=10,
    )

    if not top_invoices.empty:

        display_top = top_invoices.copy()

        if "dashboard_risk_score" in display_top.columns:

            display_top = display_top.rename(
                columns={
                    "dashboard_risk_score":
                    "Risk Score"
                }
            )

        if "dashboard_risk_level" in display_top.columns:

            display_top = display_top.rename(
                columns={
                    "dashboard_risk_level":
                    "Risk Level"
                }
            )

        st.dataframe(
            display_top,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Risk invoice data unavailable."
        )

    # ============================================================
    # TOP RISKY VENDORS
    # ============================================================

    st.subheader(
        "🏢 Highest-Risk Vendors"
    )

    top_vendors = get_top_risky_vendors(
        risk_df,
        limit=10,
    )

    if not top_vendors.empty:

        st.dataframe(
            top_vendors,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Vendor risk data unavailable."
        )

    st.divider()

    # ============================================================
    # MONTHLY TREND
    # ============================================================

    if "InvoiceDate" in df.columns:

        chart_df = df.copy()

        chart_df["InvoiceDate"] = pd.to_datetime(
            chart_df["InvoiceDate"],
            errors="coerce",
        )

        monthly = (
            chart_df
            .dropna(
                subset=["InvoiceDate"]
            )
            .set_index("InvoiceDate")
            .resample("ME")
            .agg(
                {
                    "invoice_dollars": "sum",
                    "Freight": "sum",
                }
            )
        )

        if not monthly.empty:

            st.subheader(
                "Monthly Invoice and Freight Value"
            )

            st.line_chart(
                monthly[
                    [
                        "invoice_dollars",
                        "Freight",
                    ]
                ]
            )


# ################################################################
# FREIGHT PREDICTION
# ################################################################

elif page == "🚚 Freight Prediction":

    st.header(
        "Freight Cost Prediction"
    )

    st.write(
        "Estimate expected freight cost using the trained "
        "machine-learning regression model."
    )

    freight_model = load_freight_model()

    if freight_model is None:

        st.error(
            "Freight prediction model not found."
        )

        st.info(
            "Run: `python src\\train_freight.py`"
        )

        st.stop()

    st.subheader(
        "Enter Invoice Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        invoice_quantity = st.number_input(
            "Invoice Quantity",
            min_value=0.0,
            value=100.0,
            step=1.0,
        )

    with col2:

        invoice_dollars = st.number_input(
            "Invoice Dollars",
            min_value=0.0,
            value=5000.0,
            step=100.0,
        )

    with col3:

        total_item_quantity = st.number_input(
            "Purchase Quantity",
            min_value=0.0,
            value=invoice_quantity,
            step=1.0,
        )

    col4, col5, col6 = st.columns(3)

    with col4:

        total_item_dollars = st.number_input(
            "Purchase Dollars",
            min_value=0.0,
            value=invoice_dollars,
            step=100.0,
        )

    with col5:

        number_of_purchase_lines = st.number_input(
            "Purchase Lines",
            min_value=0,
            value=1,
            step=1,
        )

    with col6:

        number_of_brands = st.number_input(
            "Number of Brands",
            min_value=0,
            value=1,
            step=1,
        )

    quantity_difference = (
        invoice_quantity
        -
        total_item_quantity
    )

    absolute_quantity_difference = abs(
        quantity_difference
    )

    dollar_difference = (
        invoice_dollars
        -
        total_item_dollars
    )

    absolute_dollar_difference = abs(
        dollar_difference
    )

    invoice_cost_per_unit = (
        invoice_dollars
        /
        invoice_quantity
        if invoice_quantity != 0
        else 0
    )

    purchase_cost_per_unit = (
        total_item_dollars
        /
        total_item_quantity
        if total_item_quantity != 0
        else 0
    )

    input_data = pd.DataFrame(
        {
            "invoice_quantity": [
                invoice_quantity
            ],
            "invoice_dollars": [
                invoice_dollars
            ],
            "total_item_quantity": [
                total_item_quantity
            ],
            "total_item_dollars": [
                total_item_dollars
            ],
            "quantity_difference": [
                quantity_difference
            ],
            "absolute_quantity_difference": [
                absolute_quantity_difference
            ],
            "dollar_difference": [
                dollar_difference
            ],
            "absolute_dollar_difference": [
                absolute_dollar_difference
            ],
            "invoice_cost_per_unit": [
                invoice_cost_per_unit
            ],
            "purchase_cost_per_unit": [
                purchase_cost_per_unit
            ],
            "number_of_purchase_lines": [
                number_of_purchase_lines
            ],
            "number_of_brands": [
                number_of_brands
            ],
        }
    )

    input_data = input_data[
        get_freight_features()
    ]

    if st.button(
        "Predict Freight Cost",
        type="primary",
    ):

        try:

            prediction = freight_model.predict(
                input_data
            )[0]

            st.success(
                f"Predicted Freight Cost: "
                f"{format_currency(prediction)}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Predicted Freight",
                    format_currency(prediction),
                )

            with col2:

                freight_percentage = (
                    prediction
                    /
                    invoice_dollars
                    *
                    100
                    if invoice_dollars != 0
                    else 0
                )

                st.metric(
                    "Freight %",
                    format_percentage(
                        freight_percentage
                    ),
                )

            with col3:

                total_cost = (
                    invoice_dollars
                    +
                    prediction
                )

                st.metric(
                    "Total Estimated Cost",
                    format_currency(total_cost),
                )

        except Exception as exc:

            st.error(
                "Freight prediction failed."
            )

            st.exception(exc)


# ################################################################
# INVOICE RISK
# ################################################################

elif page == "🚨 Invoice Risk":

    st.header(
        "Invoice Risk Analysis"
    )

    st.write(
        "Assess an invoice using anomaly detection, financial "
        "mismatch, freight behavior, and vendor-level indicators."
    )

    anomaly_model = load_anomaly_model()

    if anomaly_model is None:

        st.error(
            "Anomaly detection model not found."
        )

        st.info(
            "Run: `python src\\train_anomaly.py`"
        )

        st.stop()

    analysis_type = st.radio(
        "Choose Analysis Type",
        [
            "Existing Invoice",
            "New Invoice Assessment",
        ],
        horizontal=True,
    )

    st.divider()

    # ============================================================
    # EXISTING INVOICE
    # ============================================================

    if analysis_type == "Existing Invoice":

        st.subheader(
            "Search and Select an Existing Invoice"
        )

        if "PONumber" not in df.columns:

            st.error(
                "PONumber column is unavailable."
            )

            st.stop()

        search_col1, search_col2 = st.columns(
            [3, 1]
        )

        with search_col1:

            invoice_search = st.text_input(
                "🔎 Search by PO Number or Vendor",
                placeholder="Example: 8125 or CENTEUR",
            )

        with search_col2:

            max_results = st.number_input(
                "Maximum Results",
                min_value=10,
                max_value=100,
                value=25,
                step=5,
            )

        search_df = df.copy()

        search_df["_po_search"] = (
            search_df["PONumber"]
            .fillna("")
            .astype(str)
        )

        if "VendorName" in search_df.columns:

            search_df["_vendor_search"] = (
                search_df["VendorName"]
                .fillna("")
                .astype(str)
            )

        else:

            search_df["_vendor_search"] = ""

        if invoice_search.strip():

            search_text = (
                invoice_search
                .strip()
                .lower()
            )

            po_match = (
                search_df["_po_search"]
                .str.lower()
                .str.contains(
                    search_text,
                    na=False,
                    regex=False,
                )
            )

            vendor_match = (
                search_df["_vendor_search"]
                .str.lower()
                .str.contains(
                    search_text,
                    na=False,
                    regex=False,
                )
            )

            search_df = search_df[
                po_match | vendor_match
            ]

        search_df = search_df.head(
            int(max_results)
        )

        if search_df.empty:

            st.warning(
                "No invoices found."
            )

            st.stop()

        invoice_options = []
        option_to_po = {}

        for _, row in search_df.iterrows():

            po = str(
                row.get(
                    "PONumber",
                    "N/A",
                )
            )

            vendor = str(
                row.get(
                    "VendorName",
                    "Unknown Vendor",
                )
            )

            value = row.get(
                "invoice_dollars",
                0,
            )

            try:

                value_text = (
                    f"${float(value):,.2f}"
                )

            except Exception:

                value_text = "$0.00"

            option = (
                f"PO {po} | "
                f"{vendor} | "
                f"{value_text}"
            )

            invoice_options.append(
                option
            )

            option_to_po[
                option
            ] = po

        selected_option = st.selectbox(
            "Select Invoice",
            invoice_options,
        )

        selected_po = option_to_po[
            selected_option
        ]

        selected_rows = df[
            df["PONumber"]
            .astype(str)
            ==
            selected_po
        ]

        if selected_rows.empty:

            st.error(
                "Selected invoice could not be found."
            )

            st.stop()

        selected = (
            selected_rows
            .iloc[0]
            .copy()
        )

        st.subheader(
            "Invoice Information"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Vendor",
                str(
                    selected.get(
                        "VendorName",
                        "Unknown",
                    )
                ),
            )

        with col2:

            st.metric(
                "Invoice Value",
                format_currency(
                    selected.get(
                        "invoice_dollars",
                        0,
                    )
                ),
            )

        with col3:

            st.metric(
                "Freight",
                format_currency(
                    selected.get(
                        "Freight",
                        0,
                    )
                ),
            )

        with col4:

            st.metric(
                "Invoice Quantity",
                format_number(
                    selected.get(
                        "invoice_quantity",
                        0,
                    )
                ),
            )

        # ========================================================
        # ANOMALY FEATURES
        # ========================================================

        anomaly_features = get_anomaly_features()

        missing_features = [
            feature
            for feature in anomaly_features
            if feature not in selected.index
        ]

        if missing_features:

            st.error(
                "Required anomaly features are missing:"
            )

            st.write(missing_features)

            st.stop()

        anomaly_input = pd.DataFrame(
            [selected]
        )[anomaly_features]

        try:

            model = anomaly_model.named_steps[
                "model"
            ]

            preprocessor = (
                anomaly_model.named_steps[
                    "preprocessor"
                ]
            )

            transformed = (
                preprocessor.transform(
                    anomaly_input
                )
            )

            prediction = int(
                model.predict(
                    transformed
                )[0]
            )

            normality = float(
                model.decision_function(
                    transformed
                )[0]
            )

            anomaly_score = float(
                np.clip(
                    (
                        0.5
                        -
                        normality
                    )
                    * 100,
                    0,
                    100,
                )
            )

            anomaly_status = (
                "Anomaly"
                if prediction == -1
                else "Normal"
            )

        except Exception as exc:

            st.error(
                "Unable to calculate anomaly prediction."
            )

            st.exception(exc)

            st.stop()

        # ========================================================
        # RISK SCORE
        # ========================================================

        risk_input = pd.DataFrame(
            [selected]
        ).copy()

        risk_input[
            "anomaly_score"
        ] = anomaly_score

        try:

            scored = calculate_risk_score(
                risk_input
            )

            scored = finalize_risk_output(
                scored
            )

            final_row = scored.iloc[0]

        except Exception as exc:

            st.error(
                "Risk scoring failed."
            )

            st.exception(exc)

            st.stop()

        risk_score = float(
            final_row["risk_score"]
        )

        risk_level = str(
            final_row["risk_level"]
        )

        recommendation = str(
            final_row["recommendation"]
        )

        reasons = str(
            final_row["risk_reasons"]
        )

        st.divider()

        st.subheader(
            "Anomaly Detection Result"
        )

        anomaly_col1, anomaly_col2 = st.columns(2)

        with anomaly_col1:

            st.metric(
                "Anomaly Status",
                anomaly_status,
            )

        with anomaly_col2:

            st.metric(
                "Anomaly Score",
                f"{anomaly_score:.1f}/100",
            )

        if anomaly_status == "Anomaly":

            st.warning(
                "⚠️ This invoice shows unusual "
                "characteristics compared with "
                "the learned invoice patterns."
            )

        else:

            st.success(
                "✅ This invoice follows patterns "
                "that are generally considered normal."
            )

        st.caption(
            "The anomaly score is a relative indicator "
            "and is not a probability of fraud."
        )

        # ========================================================
        # RISK ASSESSMENT
        # ========================================================

        st.subheader(
            "Overall Risk Assessment"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Risk Score",
                f"{risk_score:.1f}/100",
            )

        with col2:

            st.metric(
                "Risk Level",
                risk_level,
            )

        with col3:

            st.metric(
                "Anomaly Score",
                f"{anomaly_score:.1f}/100",
            )

        st.progress(
            min(
                max(
                    int(risk_score),
                    0,
                ),
                100,
            )
        )

        st.info(
            f"Recommendation: **{recommendation}**"
        )

        st.subheader(
            "Risk Indicators"
        )

        st.write(reasons)

        # ========================================================
        # COMPONENT ANALYSIS
        # ========================================================

        st.subheader(
            "Risk Component Analysis"
        )

        component_data = pd.DataFrame(
            {
                "Risk Component": [
                    "Anomaly",
                    "Dollar Difference",
                    "Quantity Difference",
                    "Freight",
                    "Vendor Invoice Deviation",
                    "Vendor Freight Deviation",
                ],
                "Component Score": [
                    final_row[
                        "risk_anomaly_component"
                    ],
                    final_row[
                        "risk_dollar_component"
                    ],
                    final_row[
                        "risk_quantity_component"
                    ],
                    final_row[
                        "risk_freight_component"
                    ],
                    final_row[
                        "risk_vendor_invoice_component"
                    ],
                    final_row[
                        "risk_vendor_freight_component"
                    ],
                ],
                "Weight": [
                    "40%",
                    "20%",
                    "15%",
                    "10%",
                    "10%",
                    "5%",
                ],
                "Weighted Contribution": [
                    final_row[
                        "risk_anomaly_contribution"
                    ],
                    final_row[
                        "risk_dollar_contribution"
                    ],
                    final_row[
                        "risk_quantity_contribution"
                    ],
                    final_row[
                        "risk_freight_contribution"
                    ],
                    final_row[
                        "risk_vendor_invoice_contribution"
                    ],
                    final_row[
                        "risk_vendor_freight_contribution"
                    ],
                ],
            }
        )

        component_data[
            "Component Score"
        ] = component_data[
            "Component Score"
        ].round(2)

        component_data[
            "Weighted Contribution"
        ] = component_data[
            "Weighted Contribution"
        ].round(2)

        st.dataframe(
            component_data,
            use_container_width=True,
            hide_index=True,
        )

        # ========================================================
        # PDF REPORT
        # ========================================================

        st.divider()

        st.subheader(
            "Invoice Risk Report"
        )

        if st.button(
            "📄 Generate PDF Report",
            type="primary",
        ):

            try:

                safe_po = (
                    str(selected_po)
                    .replace("/", "_")
                    .replace("\\", "_")
                    .replace(" ", "_")
                )

                report_path = (
                    REPORTS_DIR
                    /
                    f"invoice_risk_report_{safe_po}.pdf"
                )

                generate_report(
                    final_row,
                    report_path,
                )

                st.success(
                    "PDF report generated successfully."
                )

                with open(
                    report_path,
                    "rb",
                ) as pdf_file:

                    pdf_bytes = pdf_file.read()

                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=(
                        f"invoice_risk_report_"
                        f"{safe_po}.pdf"
                    ),
                    mime="application/pdf",
                )

            except Exception as exc:

                st.error(
                    "Unable to generate PDF report."
                )

                st.exception(exc)

    # ============================================================
    # NEW INVOICE ASSESSMENT
    # ============================================================

    else:

        st.subheader(
            "New Invoice Assessment"
        )

        st.write(
            "Enter the details of a new invoice. "
            "The trained anomaly detection model will "
            "evaluate whether the invoice follows normal "
            "patterns in the dataset."
        )

        st.info(
            "This assessment does not require the invoice "
            "to already exist in the database."
        )

        # ========================================================
        # BASIC INFORMATION
        # ========================================================

        st.markdown(
            "### 1. Invoice Information"
        )

        col1, col2 = st.columns(2)

        with col1:

            new_vendor_number = st.text_input(
                "Vendor Number",
                value="1000",
            )

        with col2:

            new_vendor_name = st.text_input(
                "Vendor Name",
                value="New Vendor",
            )

        col1, col2 = st.columns(2)

        with col1:

            new_invoice_date = st.date_input(
                "Invoice Date"
            )

        with col2:

            new_po_date = st.date_input(
                "PO Date"
            )

        # ========================================================
        # FINANCIAL INFORMATION
        # ========================================================

        st.markdown(
            "### 2. Financial Information"
        )

        col1, col2 = st.columns(2)

        with col1:

            new_invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=0.0,
                value=100.0,
                step=1.0,
            )

        with col2:

            new_invoice_dollars = st.number_input(
                "Invoice Dollar Value",
                min_value=0.0,
                value=1000.0,
                step=100.0,
            )

        col1, col2 = st.columns(2)

        with col1:

            new_freight = st.number_input(
                "Freight",
                min_value=0.0,
                value=50.0,
                step=10.0,
            )

        with col2:

            new_total_item_quantity = st.number_input(
                "Purchase Item Quantity",
                min_value=0.0,
                value=100.0,
                step=1.0,
            )

        col1, col2 = st.columns(2)

        with col1:

            new_total_item_dollars = st.number_input(
                "Purchase Item Dollar Value",
                min_value=0.0,
                value=1000.0,
                step=100.0,
            )

        with col2:

            new_average_purchase_price = st.number_input(
                "Average Purchase Price",
                min_value=0.0,
                value=10.0,
                step=1.0,
            )

        # ========================================================
        # PURCHASE INFORMATION
        # ========================================================

        st.markdown(
            "### 3. Purchase Information"
        )

        col1, col2 = st.columns(2)

        with col1:

            new_purchase_lines = st.number_input(
                "Number of Purchase Lines",
                min_value=0,
                value=1,
                step=1,
            )

        with col2:

            new_number_brands = st.number_input(
                "Number of Brands",
                min_value=0,
                value=1,
                step=1,
            )

        st.divider()

        assess_button = st.button(
            "🔍 Analyze New Invoice",
            type="primary",
            use_container_width=True,
        )

        if assess_button:

            try:

                # ==================================================
                # BUILD NEW RAW INVOICE
                # ==================================================

                new_invoice = pd.DataFrame(
                    [
                        {
                            "VendorNumber":
                                new_vendor_number,

                            "VendorName":
                                new_vendor_name,

                            "InvoiceDate":
                                pd.Timestamp(
                                    new_invoice_date
                                ),

                            "PONumber":
                                "NEW-INVOICE",

                            "PODate":
                                pd.Timestamp(
                                    new_po_date
                                ),

                            "PayDate":
                                pd.NaT,

                            "invoice_quantity":
                                float(
                                    new_invoice_quantity
                                ),

                            "invoice_dollars":
                                float(
                                    new_invoice_dollars
                                ),

                            "Freight":
                                float(
                                    new_freight
                                ),

                            "Approval":
                                "Unknown",

                            "total_item_quantity":
                                float(
                                    new_total_item_quantity
                                ),

                            "total_item_dollars":
                                float(
                                    new_total_item_dollars
                                ),

                            "average_purchase_price":
                                float(
                                    new_average_purchase_price
                                ),

                            "number_of_purchase_lines":
                                int(
                                    new_purchase_lines
                                ),

                            "number_of_brands":
                                int(
                                    new_number_brands
                                ),
                        }
                    ]
                )

                # ==================================================
                # HISTORICAL RAW DATA
                # ==================================================

                historical_for_features = (
                    get_raw_historical_data(df)
                )

                # ==================================================
                # COMBINE RAW HISTORICAL + NEW INVOICE
                # ==================================================

                combined = pd.concat(
                    [
                        historical_for_features,
                        new_invoice,
                    ],
                    ignore_index=True,
                    sort=False,
                )

                # ==================================================
                # FEATURE ENGINEERING
                # ==================================================

                combined_features = build_features(
                    combined,
                    create_target=False,
                )

                new_features = (
                    combined_features
                    .iloc[[-1]]
                    .copy()
                )

                # ==================================================
                # ANOMALY FEATURES
                # ==================================================

                anomaly_features = (
                    get_anomaly_features()
                )

                missing_features = [
                    feature
                    for feature in anomaly_features
                    if feature
                    not in new_features.columns
                ]

                if missing_features:

                    st.error(
                        "Some anomaly features could not be created:"
                    )

                    st.write(
                        missing_features
                    )

                    st.stop()

                anomaly_input = (
                    new_features[
                        anomaly_features
                    ].copy()
                )

                # ==================================================
                # ANOMALY MODEL
                # ==================================================

                model = (
                    anomaly_model
                    .named_steps["model"]
                )

                preprocessor = (
                    anomaly_model
                    .named_steps["preprocessor"]
                )

                transformed = (
                    preprocessor.transform(
                        anomaly_input
                    )
                )

                prediction = int(
                    model.predict(
                        transformed
                    )[0]
                )

                normality = float(
                    model.decision_function(
                        transformed
                    )[0]
                )

                anomaly_score = float(
                    np.clip(
                        (
                            0.5
                            -
                            normality
                        )
                        * 100,
                        0,
                        100,
                    )
                )

                anomaly_status = (
                    "Anomaly"
                    if prediction == -1
                    else "Normal"
                )

                # ==================================================
                # RISK SCORE
                # ==================================================

                risk_input = (
                    new_features.copy()
                )

                risk_input[
                    "anomaly_score"
                ] = anomaly_score

                scored = calculate_risk_score(
                    risk_input
                )

                scored = finalize_risk_output(
                    scored
                )

                final_row = scored.iloc[0]

                risk_score = float(
                    final_row["risk_score"]
                )

                risk_level = str(
                    final_row["risk_level"]
                )

                recommendation = str(
                    final_row["recommendation"]
                )

                reasons = str(
                    final_row["risk_reasons"]
                )

                # ==================================================
                # DISPLAY
                # ==================================================

                st.success(
                    "New invoice analysis completed."
                )

                st.divider()

                st.subheader(
                    "New Invoice Assessment Result"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Anomaly Status",
                        anomaly_status,
                    )

                with col2:

                    st.metric(
                        "Anomaly Score",
                        f"{anomaly_score:.1f}/100",
                    )

                with col3:

                    st.metric(
                        "Risk Score",
                        f"{risk_score:.1f}/100",
                    )

                st.progress(
                    min(
                        max(
                            int(risk_score),
                            0,
                        ),
                        100,
                    )
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Risk Level",
                        risk_level,
                    )

                with col2:

                    st.info(
                        f"Recommendation: "
                        f"**{recommendation}**"
                    )

                if anomaly_status == "Anomaly":

                    st.warning(
                        "⚠️ The model identified this "
                        "invoice as unusual compared "
                        "with learned invoice patterns."
                    )

                else:

                    st.success(
                        "✅ The model did not identify "
                        "this invoice as an anomaly."
                    )

                st.caption(
                    "Anomaly detection identifies unusual "
                    "patterns. An anomaly does not automatically "
                    "mean fraud."
                )

                # ==================================================
                # RISK INDICATORS
                # ==================================================

                st.subheader(
                    "Risk Indicators"
                )

                st.write(reasons)

                # ==================================================
                # COMPONENT ANALYSIS
                # ==================================================

                st.subheader(
                    "Risk Component Analysis"
                )

                component_data = pd.DataFrame(
                    {
                        "Risk Component": [
                            "Anomaly",
                            "Dollar Difference",
                            "Quantity Difference",
                            "Freight",
                            "Vendor Invoice Deviation",
                            "Vendor Freight Deviation",
                        ],
                        "Component Score": [
                            final_row[
                                "risk_anomaly_component"
                            ],
                            final_row[
                                "risk_dollar_component"
                            ],
                            final_row[
                                "risk_quantity_component"
                            ],
                            final_row[
                                "risk_freight_component"
                            ],
                            final_row[
                                "risk_vendor_invoice_component"
                            ],
                            final_row[
                                "risk_vendor_freight_component"
                            ],
                        ],
                        "Weight": [
                            "40%",
                            "20%",
                            "15%",
                            "10%",
                            "10%",
                            "5%",
                        ],
                        "Weighted Contribution": [
                            final_row[
                                "risk_anomaly_contribution"
                            ],
                            final_row[
                                "risk_dollar_contribution"
                            ],
                            final_row[
                                "risk_quantity_contribution"
                            ],
                            final_row[
                                "risk_freight_contribution"
                            ],
                            final_row[
                                "risk_vendor_invoice_contribution"
                            ],
                            final_row[
                                "risk_vendor_freight_contribution"
                            ],
                        ],
                    }
                )

                component_data[
                    "Component Score"
                ] = component_data[
                    "Component Score"
                ].round(2)

                component_data[
                    "Weighted Contribution"
                ] = component_data[
                    "Weighted Contribution"
                ].round(2)

                st.dataframe(
                    component_data,
                    use_container_width=True,
                    hide_index=True,
                )

                # ==================================================
                # SUBMITTED DATA
                # ==================================================

                st.subheader(
                    "Submitted Invoice Details"
                )

                summary_data = pd.DataFrame(
                    {
                        "Field": [
                            "Vendor Number",
                            "Vendor Name",
                            "Invoice Quantity",
                            "Invoice Value",
                            "Freight",
                            "Purchase Quantity",
                            "Purchase Value",
                            "Purchase Lines",
                            "Number of Brands",
                        ],
                        "Value": [
                            new_vendor_number,
                            new_vendor_name,
                            new_invoice_quantity,
                            format_currency(
                                new_invoice_dollars
                            ),
                            format_currency(
                                new_freight
                            ),
                            new_total_item_quantity,
                            format_currency(
                                new_total_item_dollars
                            ),
                            new_purchase_lines,
                            new_number_brands,
                        ],
                    }
                )

                st.dataframe(
                    summary_data,
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as exc:

                st.error(
                    "New invoice analysis failed."
                )

                st.exception(exc)


# ################################################################
# ANOMALY DETECTION
# ################################################################

elif page == "🔍 Anomaly Detection":

    st.header(
        "Invoice Anomaly Detection"
    )

    st.write(
        "Identify invoices whose characteristics are unusual "
        "relative to the invoice population."
    )

    anomaly_results_path = (
        RESULTS_DIR
        /
        "invoice_anomaly_results.csv"
    )

    if not anomaly_results_path.exists():

        st.error(
            "Anomaly results have not been generated."
        )

        st.info(
            "Run: `python src\\train_anomaly.py`"
        )

        st.stop()

    anomaly_df = pd.read_csv(
        anomaly_results_path
    )

    total = len(anomaly_df)

    if "anomaly_status" in anomaly_df.columns:

        anomaly_count = int(
            (
                anomaly_df[
                    "anomaly_status"
                ]
                ==
                "Anomaly"
            ).sum()
        )

    else:

        anomaly_count = 0

    anomaly_percentage = (
        anomaly_count
        /
        total
        *
        100
        if total
        else 0
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Invoices Analyzed",
            f"{total:,}",
        )

    with col2:

        st.metric(
            "Anomalies",
            f"{anomaly_count:,}",
        )

    with col3:

        st.metric(
            "Anomaly Rate",
            format_percentage(
                anomaly_percentage
            ),
        )

    st.divider()

    if "anomaly_status" in anomaly_df.columns:

        status_filter = st.selectbox(
            "Invoice Status",
            [
                "All",
                "Anomaly",
                "Normal",
            ],
        )

    else:

        status_filter = "All"

    if "risk_category" in anomaly_df.columns:

        category_filter = st.selectbox(
            "Risk Category",
            [
                "All",
                "Low",
                "Medium",
                "High",
            ],
        )

    else:

        category_filter = "All"

    filtered = anomaly_df.copy()

    if (
        status_filter != "All"
        and
        "anomaly_status"
        in filtered.columns
    ):

        filtered = filtered[
            filtered[
                "anomaly_status"
            ]
            ==
            status_filter
        ]

    if (
        category_filter != "All"
        and
        "risk_category"
        in filtered.columns
    ):

        filtered = filtered[
            filtered[
                "risk_category"
            ]
            ==
            category_filter
        ]

    display_columns = [
        column
        for column in [
            "VendorNumber",
            "VendorName",
            "InvoiceDate",
            "PONumber",
            "invoice_quantity",
            "invoice_dollars",
            "Freight",
            "dollar_difference",
            "quantity_difference",
            "freight_percentage",
            "anomaly_score",
            "anomaly_status",
            "risk_category",
        ]
        if column in filtered.columns
    ]

    st.dataframe(
        filtered[
            display_columns
        ],
        use_container_width=True,
        height=500,
    )

    csv_data = (
        filtered
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "⬇️ Download Anomaly Results",
        data=csv_data,
        file_name="invoice_anomaly_results.csv",
        mime="text/csv",
    )


# ################################################################
# VENDOR ANALYTICS
# ################################################################

elif page == "🏢 Vendor Analytics":

    st.header(
        "Vendor Analytics"
    )

    st.write(
        "Analyze invoice volume, spending, freight behavior, "
        "and vendor performance."
    )

    if "VendorName" not in df.columns:

        st.error(
            "Vendor information is unavailable."
        )

        st.stop()

    vendor_summary = (
        df.groupby(
            [
                "VendorNumber",
                "VendorName",
            ],
            dropna=False,
        )
        .agg(
            invoice_count=(
                "invoice_dollars",
                "count",
            ),
            total_invoice_value=(
                "invoice_dollars",
                "sum",
            ),
            average_invoice_value=(
                "invoice_dollars",
                "mean",
            ),
            total_freight=(
                "Freight",
                "sum",
            ),
            average_freight=(
                "Freight",
                "mean",
            ),
        )
        .reset_index()
    )

    vendor_summary[
        "freight_percentage"
    ] = np.where(
        vendor_summary[
            "total_invoice_value"
        ] != 0,
        (
            vendor_summary[
                "total_freight"
            ]
            /
            vendor_summary[
                "total_invoice_value"
            ]
            *
            100
        ),
        0,
    )

    vendor_names = (
        vendor_summary[
            "VendorName"
        ]
        .dropna()
        .astype(str)
        .sort_values()
        .tolist()
    )

    selected_vendor = st.selectbox(
        "Select Vendor",
        vendor_names,
    )

    vendor_data = vendor_summary[
        vendor_summary[
            "VendorName"
        ].astype(str)
        ==
        selected_vendor
    ]

    if vendor_data.empty:

        st.warning(
            "No vendor data found."
        )

        st.stop()

    vendor = vendor_data.iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Invoices",
            f"{int(vendor['invoice_count']):,}",
        )

    with col2:

        st.metric(
            "Total Value",
            format_currency(
                vendor[
                    "total_invoice_value"
                ]
            ),
        )

    with col3:

        st.metric(
            "Average Invoice",
            format_currency(
                vendor[
                    "average_invoice_value"
                ]
            ),
        )

    with col4:

        st.metric(
            "Average Freight",
            format_currency(
                vendor[
                    "average_freight"
                ]
            ),
        )

    st.divider()

    st.subheader(
        "Vendor Performance"
    )

    st.dataframe(
        vendor_summary.sort_values(
            "total_invoice_value",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    chart = (
        vendor_summary
        .sort_values(
            "total_invoice_value",
            ascending=False,
        )
        .head(15)
        .set_index("VendorName")
    )

    st.subheader(
        "Top Vendors by Invoice Value"
    )

    st.bar_chart(
        chart[
            "total_invoice_value"
        ]
    )


# ################################################################
# BATCH ANALYSIS
# ################################################################

elif page == "📁 Batch Analysis":

    st.header(
        "Batch Invoice Analysis"
    )

    st.write(
        "Upload a CSV file containing invoice records "
        "for bulk freight prediction, anomaly detection "
        "and risk assessment."
    )

    # ============================================================
    # FILE UPLOAD
    # ============================================================

    uploaded_file = st.file_uploader(
        "Upload Invoice CSV",
        type=["csv"],
    )

    if uploaded_file is None:

        st.info(
            "Upload a CSV file to begin."
        )

        st.stop()

    # ============================================================
    # READ CSV
    # ============================================================

    try:

        uploaded_df = pd.read_csv(
            uploaded_file
        )

    except Exception as exc:

        st.error(
            "Unable to read uploaded CSV."
        )

        st.exception(exc)

        st.stop()

    # ============================================================
    # PREVIEW
    # ============================================================

    st.subheader(
        "Uploaded Data"
    )

    st.write(
        f"Rows: {len(uploaded_df):,}"
    )

    st.dataframe(
        uploaded_df.head(20),
        use_container_width=True,
    )

    # ============================================================
    # LOAD MODELS
    # ============================================================

    freight_model = load_freight_model()
    anomaly_model = load_anomaly_model()

    if freight_model is None:

        st.error(
            "Freight prediction model is unavailable."
        )

        st.info(
            "Run the freight model training script first."
        )

        st.stop()

    if anomaly_model is None:

        st.error(
            "Anomaly detection model is unavailable."
        )

        st.info(
            "Run: `python src\\train_anomaly.py`"
        )

        st.stop()

    # ============================================================
    # REQUIRED FREIGHT COLUMNS
    # ============================================================

    freight_required_columns = [
        "invoice_quantity",
        "invoice_dollars",
        "total_item_quantity",
        "total_item_dollars",
        "number_of_purchase_lines",
        "number_of_brands",
    ]

    missing_freight_columns = [
        column
        for column in freight_required_columns
        if column not in uploaded_df.columns
    ]

    if missing_freight_columns:

        st.error(
            "Missing required columns for freight prediction:"
        )

        st.code(
            "\n".join(
                missing_freight_columns
            )
        )

        st.stop()

    # ============================================================
    # REQUIRED ANOMALY COLUMNS
    # ============================================================

    anomaly_base_columns = [
        "invoice_quantity",
        "invoice_dollars",
        "Freight",
        "total_item_quantity",
        "total_item_dollars",
        "number_of_purchase_lines",
        "number_of_brands",
    ]

    missing_anomaly_columns = [
        column
        for column in anomaly_base_columns
        if column not in uploaded_df.columns
    ]

    if missing_anomaly_columns:

        st.error(
            "Missing required columns for anomaly detection:"
        )

        st.code(
            "\n".join(
                missing_anomaly_columns
            )
        )

        st.info(
            "The uploaded CSV must contain actual Freight "
            "values for anomaly detection."
        )

        st.stop()

    # ============================================================
    # PREPARE BATCH
    # ============================================================

    batch = uploaded_df.copy()

    numeric_columns = [
        "invoice_quantity",
        "invoice_dollars",
        "Freight",
        "total_item_quantity",
        "total_item_dollars",
        "number_of_purchase_lines",
        "number_of_brands",
    ]

    for column in numeric_columns:

        batch[column] = pd.to_numeric(
            batch[column],
            errors="coerce",
        )

        median_value = batch[column].median()

        if pd.isna(median_value):
            median_value = 0

        batch[column] = batch[column].fillna(
            median_value
        )

    # ============================================================
    # OPTIONAL RAW COLUMNS
    # ============================================================

    if "VendorNumber" not in batch.columns:

        st.error(
            "VendorNumber is required for vendor-level "
            "anomaly and risk analysis."
        )

        st.info(
            "Please include VendorNumber in the CSV."
        )

        st.stop()

    if "VendorName" not in batch.columns:

        batch["VendorName"] = "Unknown Vendor"

    if "InvoiceDate" not in batch.columns:

        batch["InvoiceDate"] = pd.NaT

    if "PONumber" not in batch.columns:

        batch["PONumber"] = [
            f"BATCH-{i + 1}"
            for i in range(len(batch))
        ]

    if "PODate" not in batch.columns:

        batch["PODate"] = pd.NaT

    if "PayDate" not in batch.columns:

        batch["PayDate"] = pd.NaT

    if "Approval" not in batch.columns:

        batch["Approval"] = "Unknown"

    if "average_purchase_price" not in batch.columns:

        batch[
            "average_purchase_price"
        ] = np.where(
            batch[
                "total_item_quantity"
            ] != 0,
            batch[
                "total_item_dollars"
            ]
            /
            batch[
                "total_item_quantity"
            ],
            0,
        )

    # ============================================================
    # BASIC BATCH FEATURES
    # ============================================================

    batch[
        "quantity_difference"
    ] = (
        batch["invoice_quantity"]
        -
        batch["total_item_quantity"]
    )

    batch[
        "absolute_quantity_difference"
    ] = (
        batch[
            "quantity_difference"
        ].abs()
    )

    batch[
        "dollar_difference"
    ] = (
        batch["invoice_dollars"]
        -
        batch["total_item_dollars"]
    )

    batch[
        "absolute_dollar_difference"
    ] = (
        batch[
            "dollar_difference"
        ].abs()
    )

    batch[
        "invoice_cost_per_unit"
    ] = np.where(
        batch[
            "invoice_quantity"
        ] != 0,
        batch[
            "invoice_dollars"
        ]
        /
        batch[
            "invoice_quantity"
        ],
        0,
    )

    batch[
        "purchase_cost_per_unit"
    ] = np.where(
        batch[
            "total_item_quantity"
        ] != 0,
        batch[
            "total_item_dollars"
        ]
        /
        batch[
            "total_item_quantity"
        ],
        0,
    )

    # ============================================================
    # FREIGHT PREDICTION
    # ============================================================

    try:

        freight_input = batch[
            get_freight_features()
        ].copy()

        predicted_freight = (
            freight_model.predict(
                freight_input
            )
        )

        batch[
            "predicted_freight"
        ] = predicted_freight

        batch[
            "freight_difference"
        ] = (
            batch["Freight"]
            -
            batch["predicted_freight"]
        )

        batch[
            "absolute_freight_difference"
        ] = (
            batch[
                "freight_difference"
            ].abs()
        )

        batch[
            "freight_percentage"
        ] = np.where(
            batch[
                "invoice_dollars"
            ] != 0,
            (
                batch[
                    "predicted_freight"
                ]
                /
                batch[
                    "invoice_dollars"
                ]
                *
                100
            ),
            0,
        )

        batch[
            "actual_freight_percentage"
        ] = np.where(
            batch[
                "invoice_dollars"
            ] != 0,
            (
                batch[
                    "Freight"
                ]
                /
                batch[
                    "invoice_dollars"
                ]
                *
                100
            ),
            0,
        )

        batch[
            "estimated_total_cost"
        ] = (
            batch[
                "invoice_dollars"
            ]
            +
            batch[
                "predicted_freight"
            ]
        )

    except Exception as exc:

        st.error(
            "Batch freight prediction failed."
        )

        st.exception(exc)

        st.stop()

    # ============================================================
    # ANOMALY FEATURE ENGINEERING
    # ============================================================
    #
    # CRITICAL FIX
    #
    # df is already feature-engineered.
    #
    # The previous code did:
    #
    #     pd.concat([df, batch])
    #
    # and then called build_features().
    #
    # This caused duplicate vendor-generated columns.
    #
    # Instead, only raw historical columns are selected here.
    #
    # ============================================================

    try:

        historical_raw = (
            get_raw_historical_data(df)
        )

        batch_for_features = batch[
            [
                column
                for column in RAW_INVOICE_COLUMNS
                if column in batch.columns
            ]
        ].copy()

        combined_raw = pd.concat(
            [
                historical_raw,
                batch_for_features,
            ],
            ignore_index=True,
            sort=False,
        )

        # --------------------------------------------------------
        # Build features exactly ONCE from raw data.
        # --------------------------------------------------------

        combined_engineered = build_features(
            combined_raw,
            create_target=False,
        )

        # --------------------------------------------------------
        # Last rows belong to uploaded batch.
        # --------------------------------------------------------

        batch_engineered = (
            combined_engineered
            .tail(len(batch))
            .copy()
            .reset_index(drop=True)
        )

        # --------------------------------------------------------
        # Preserve original uploaded columns that were not part
        # of raw feature engineering.
        # --------------------------------------------------------

        original_extra_columns = [
            column
            for column in batch.columns
            if column not in batch_engineered.columns
        ]

        if original_extra_columns:

            extras = (
                batch[
                    original_extra_columns
                ]
                .reset_index(drop=True)
            )

            batch_engineered = pd.concat(
                [
                    batch_engineered,
                    extras,
                ],
                axis=1,
            )

    except Exception as exc:

        st.error(
            "Anomaly feature generation failed."
        )

        st.exception(exc)

        st.stop()

    # ============================================================
    # CHECK ANOMALY FEATURES
    # ============================================================

    anomaly_features = (
        get_anomaly_features()
    )

    missing_anomaly_features = [
        column
        for column in anomaly_features
        if column
        not in batch_engineered.columns
    ]

    if missing_anomaly_features:

        st.error(
            "Some anomaly model features could not be generated:"
        )

        st.code(
            "\n".join(
                missing_anomaly_features
            )
        )

        st.stop()

    # ============================================================
    # ANOMALY PREDICTION
    # ============================================================

    try:

        anomaly_input = (
            batch_engineered[
                anomaly_features
            ].copy()
        )

        preprocessor = (
            anomaly_model
            .named_steps[
                "preprocessor"
            ]
        )

        anomaly_estimator = (
            anomaly_model
            .named_steps[
                "model"
            ]
        )

        transformed = (
            preprocessor.transform(
                anomaly_input
            )
        )

        predictions = (
            anomaly_estimator.predict(
                transformed
            )
        )

        normality_scores = (
            anomaly_estimator
            .decision_function(
                transformed
            )
        )

        anomaly_scores = np.clip(
            (
                0.5
                -
                normality_scores
            )
            *
            100,
            0,
            100,
        )

        batch_engineered[
            "anomaly_prediction"
        ] = predictions

        batch_engineered[
            "anomaly_score"
        ] = anomaly_scores

        batch_engineered[
            "anomaly_status"
        ] = np.where(
            predictions == -1,
            "Anomaly",
            "Normal",
        )

        batch_engineered[
            "risk_category"
        ] = np.select(
            [
                batch_engineered[
                    "anomaly_score"
                ] <= 30,

                batch_engineered[
                    "anomaly_score"
                ] <= 60,
            ],
            [
                "Low",
                "Medium",
            ],
            default="High",
        )

    except Exception as exc:

        st.error(
            "Batch anomaly detection failed."
        )

        st.exception(exc)

        st.stop()

    # ============================================================
    # RISK SCORING
    # ============================================================

    try:

        risk_input = (
            batch_engineered.copy()
        )

        risk_scored = calculate_risk_score(
            risk_input
        )

        risk_scored = finalize_risk_output(
            risk_scored
        )

    except Exception as exc:

        st.error(
            "Batch risk scoring failed."
        )

        st.exception(exc)

        st.stop()

    # ============================================================
    # SUMMARY METRICS
    # ============================================================

    total_batch = len(
        risk_scored
    )

    anomaly_count = int(
        (
            risk_scored[
                "anomaly_status"
            ]
            ==
            "Anomaly"
        ).sum()
    )

    high_risk_count = int(
        (
            risk_scored[
                "risk_level"
            ]
            ==
            "High"
        ).sum()
    )

    medium_risk_count = int(
        (
            risk_scored[
                "risk_level"
            ]
            ==
            "Medium"
        ).sum()
    )

    average_risk = float(
        risk_scored[
            "risk_score"
        ].mean()
    )

    # ============================================================
    # BATCH SUMMARY
    # ============================================================

    st.divider()

    st.subheader(
        "Batch Analysis Summary"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Invoices",
            f"{total_batch:,}",
        )

    with col2:

        st.metric(
            "Anomalies",
            f"{anomaly_count:,}",
        )

    with col3:

        st.metric(
            "High Risk",
            f"{high_risk_count:,}",
        )

    with col4:

        st.metric(
            "Medium Risk",
            f"{medium_risk_count:,}",
        )

    with col5:

        st.metric(
            "Average Risk",
            f"{average_risk:.2f}/100",
        )

    # ============================================================
    # FREIGHT RESULTS
    # ============================================================

    st.divider()

    st.subheader(
        "Freight Prediction Results"
    )

    freight_result_columns = [
        column
        for column in [
            "VendorName",
            "VendorNumber",
            "PONumber",
            "invoice_quantity",
            "invoice_dollars",
            "Freight",
            "predicted_freight",
            "freight_difference",
            "actual_freight_percentage",
            "estimated_total_cost",
        ]
        if column in risk_scored.columns
    ]

    st.dataframe(
        risk_scored[
            freight_result_columns
        ],
        use_container_width=True,
        height=400,
    )

    # ============================================================
    # ANOMALY RESULTS
    # ============================================================

    st.divider()

    st.subheader(
        "Anomaly Detection Results"
    )

    anomaly_result_columns = [
        column
        for column in [
            "VendorName",
            "VendorNumber",
            "PONumber",
            "invoice_dollars",
            "Freight",
            "anomaly_score",
            "anomaly_status",
            "risk_category",
        ]
        if column in risk_scored.columns
    ]

    anomaly_display = (
        risk_scored[
            anomaly_result_columns
        ]
        .sort_values(
            "anomaly_score",
            ascending=False,
        )
    )

    st.dataframe(
        anomaly_display,
        use_container_width=True,
        height=400,
    )

    # ============================================================
    # HIGH-RISK INVOICES
    # ============================================================

    st.divider()

    st.subheader(
        "High-Risk Invoices"
    )

    high_risk = (
        risk_scored[
            risk_scored[
                "risk_level"
            ]
            ==
            "High"
        ]
        .sort_values(
            "risk_score",
            ascending=False,
        )
    )

    if high_risk.empty:

        st.success(
            "No high-risk invoices were detected "
            "in the uploaded batch."
        )

    else:

        high_risk_columns = [
            column
            for column in [
                "VendorName",
                "VendorNumber",
                "PONumber",
                "invoice_dollars",
                "Freight",
                "anomaly_score",
                "risk_score",
                "risk_level",
                "recommendation",
                "risk_reasons",
            ]
            if column in high_risk.columns
        ]

        st.dataframe(
            high_risk[
                high_risk_columns
            ],
            use_container_width=True,
            height=400,
        )

    # ============================================================
    # COMPLETE RESULTS
    # ============================================================

    st.divider()

    st.subheader(
        "Complete Batch Results"
    )

    complete_columns = [
        column
        for column in [
            "VendorName",
            "VendorNumber",
            "PONumber",
            "InvoiceDate",
            "invoice_quantity",
            "invoice_dollars",
            "Freight",
            "total_item_quantity",
            "total_item_dollars",
            "quantity_difference",
            "dollar_difference",
            "predicted_freight",
            "freight_difference",
            "anomaly_score",
            "anomaly_status",
            "risk_category",
            "risk_score",
            "risk_level",
            "recommendation",
            "risk_reasons",
        ]
        if column in risk_scored.columns
    ]

    st.dataframe(
        risk_scored[
            complete_columns
        ],
        use_container_width=True,
        height=500,
    )

    # ============================================================
    # DOWNLOAD RESULTS
    # ============================================================

    output_csv = (
        risk_scored
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "⬇️ Download Batch Results",
        data=output_csv,
        file_name="batch_invoice_analysis.csv",
        mime="text/csv",
    )


# ################################################################
# MODEL PERFORMANCE
# ################################################################

elif page == "📈 Model Performance":

    st.header(
        "Model Performance"
    )

    st.write(
        "Evaluation results for the machine-learning "
        "components of the system."
    )

    # ============================================================
    # FREIGHT MODEL
    # ============================================================

    st.subheader(
        "Freight Prediction Models"
    )

    freight_results_path = (
        RESULTS_DIR
        /
        "freight_model_results.csv"
    )

    if freight_results_path.exists():

        freight_results = pd.read_csv(
            freight_results_path
        )

        st.dataframe(
            freight_results,
            use_container_width=True,
            hide_index=True,
        )

        if (
            "RMSE" in freight_results.columns
            and
            "Model" in freight_results.columns
        ):

            st.subheader(
                "RMSE Comparison"
            )

            st.bar_chart(
                freight_results.set_index(
                    "Model"
                )["RMSE"]
            )

        if (
            "R2" in freight_results.columns
            and
            "Model" in freight_results.columns
        ):

            st.subheader(
                "R² Comparison"
            )

            st.bar_chart(
                freight_results.set_index(
                    "Model"
                )["R2"]
            )

    else:

        st.warning(
            "Freight model results not found."
        )

    st.divider()

    # ============================================================
    # ANOMALY MODEL
    # ============================================================

    st.subheader(
        "Anomaly Detection"
    )

    anomaly_summary_path = (
        RESULTS_DIR
        /
        "anomaly_summary.csv"
    )

    if anomaly_summary_path.exists():

        anomaly_summary = pd.read_csv(
            anomaly_summary_path
        )

        st.dataframe(
            anomaly_summary,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.warning(
            "Anomaly summary not found."
        )

    st.divider()

    # ============================================================
    # RISK FRAMEWORK
    # ============================================================

    st.subheader(
        "Risk Scoring Framework"
    )

    weights_df = pd.DataFrame(
        {
            "Risk Component": [
                "Anomaly Detection",
                "Dollar Difference",
                "Quantity Difference",
                "Freight",
                "Vendor Invoice Deviation",
                "Vendor Freight Deviation",
            ],
            "Weight (%)": [
                40,
                20,
                15,
                10,
                10,
                5,
            ],
        }
    )

    st.dataframe(
        weights_df,
        use_container_width=True,
        hide_index=True,
    )


# ################################################################
# ABOUT
# ################################################################

elif page == "ℹ️ About":

    st.header(
        "About the Project"
    )

    st.markdown(
        """
        ## AI-Based Vendor Invoice Intelligence
        and Anomaly Detection System

        This system combines machine learning, data analytics,
        anomaly detection, and explainable risk scoring to assist
        with vendor invoice analysis.

        ### Core Modules

        **🚚 Freight Cost Prediction**

        Predicts expected freight cost using machine-learning
        regression.

        **🔍 Anomaly Detection**

        Identifies unusual invoice patterns using machine learning.

        **🚨 Invoice Risk Scoring**

        Produces a transparent 0-100 decision-support score.

        **🏢 Vendor Analytics**

        Provides vendor-level spending and freight analysis.

        **📁 Batch Analysis**

        Processes multiple invoices through CSV upload.

        **📄 PDF Reporting**

        Generates a professional invoice risk report.

        ### Technology Stack

        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - Streamlit
        - SQLite
        - Joblib
        - ReportLab

        ### Important Note

        The risk score is a decision-support indicator.
        It is not a probability of fraud.

        An anomaly does not automatically indicate fraud,
        misconduct, or an incorrect invoice.

        Human review should be used for final decisions.
        """
    )

    st.divider()

    st.subheader(
        "System Architecture"
    )

    st.code(
        """
                     Invoice Database
                            |
                            v
                   Data Preprocessing
                            |
                            v
                   Feature Engineering
                            |
              +-------------+-------------+
              |                           |
              v                           v
        Freight Model              Anomaly Detection
              |                           |
              v                           v
       Freight Prediction            Anomaly Score
              |                           |
              +-------------+-------------+
                            |
                            v
                       Risk Scoring
                            |
              +-------------+-------------+
              |                           |
              v                           v
       Streamlit Dashboard          PDF Report
                                  Generator
        """,
        language="text",
    )