"""
train_anomaly.py

Unsupervised anomaly detection for vendor invoices.

Model:
    Isolation Forest

Purpose:
    Detect invoices whose characteristics are unusual compared
    with the overall invoice population.

Unlike the approval-risk classifier, anomaly detection does NOT
require labelled examples.

All available invoices can therefore be analyzed.

Outputs:
    models/invoice_anomaly_pipeline.pkl
    results/invoice_anomaly_results.csv
    results/anomaly_summary.csv

The saved pipeline includes preprocessing + Isolation Forest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline


# -------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# -------------------------------------------------------------------
# PROJECT IMPORTS
# -------------------------------------------------------------------

from data_loader import (  # noqa: E402
    load_invoice_purchase_summary,
    validate_invoice_dataset,
)

from feature_engineering import (  # noqa: E402
    build_features,
    get_anomaly_features,
)

from preprocessing import (  # noqa: E402
    create_anomaly_preprocessor,
)


# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

RANDOM_STATE = 42

# Expected approximate proportion of unusual observations.
# 0.05 means the model expects roughly 5% of records to be
# anomalous. This is a modelling assumption, not a ground-truth
# fraud rate.
CONTAMINATION = 0.05

N_ESTIMATORS = 300

MODEL_PATH = (
    MODELS_DIR
    / "invoice_anomaly_pipeline.pkl"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "invoice_anomaly_results.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "anomaly_summary.csv"
)


# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------

def load_anomaly_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load the complete invoice dataset and create engineered
    features.

    Returns
    -------
    X : pandas.DataFrame
        Features used by Isolation Forest.

    full_df : pandas.DataFrame
        Complete feature-engineered dataframe containing invoice
        identifiers and analytical features.
    """

    print("\nLoading invoice data...")

    raw_df = load_invoice_purchase_summary()

    validate_invoice_dataset(
        raw_df
    )

    print(
        f"Total invoices loaded: {len(raw_df):,}"
    )

    print(
        "\nCreating feature-engineered dataset..."
    )

    full_df = build_features(
        raw_df,
        create_target=False,
    )

    feature_columns = get_anomaly_features()

    X = full_df[
        feature_columns
    ].copy()

    # Replace impossible numerical values.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    print(
        f"Anomaly records: {len(X):,}"
    )

    print(
        f"Anomaly features: {X.shape[1]}"
    )

    return X, full_df


# -------------------------------------------------------------------
# CREATE MODEL
# -------------------------------------------------------------------

def create_anomaly_model() -> IsolationForest:
    """
    Create the Isolation Forest detector.
    """

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return model


# -------------------------------------------------------------------
# CREATE PIPELINE
# -------------------------------------------------------------------

def create_pipeline(
    X: pd.DataFrame,
) -> Pipeline:
    """
    Create complete preprocessing + Isolation Forest pipeline.
    """

    preprocessor = create_anomaly_preprocessor(
        X
    )

    model = create_anomaly_model()

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# -------------------------------------------------------------------
# ANOMALY SCORE
# -------------------------------------------------------------------

def calculate_anomaly_score(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Calculate normalized anomaly scores.

    Isolation Forest's decision_function gives:
        higher value  -> more normal
        lower value   -> more anomalous

    We transform it so that:
        higher score -> more anomalous

    Then normalize approximately to a 0-100 scale within the
    current dataset.

    Important:
        This is a relative anomaly score, not a probability
        of fraud.
    """

    model = pipeline.named_steps[
        "model"
    ]

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    transformed_X = preprocessor.transform(
        X
    )

    normality_score = model.decision_function(
        transformed_X
    )

    # Invert because lower Isolation Forest values indicate
    # stronger anomaly.
    anomaly_strength = -normality_score

    minimum = anomaly_strength.min()
    maximum = anomaly_strength.max()

    if maximum == minimum:

        normalized = np.zeros(
            len(anomaly_strength)
        )

    else:

        normalized = (
            (
                anomaly_strength
                - minimum
            )
            /
            (
                maximum
                - minimum
            )
        ) * 100

    return normalized


# -------------------------------------------------------------------
# CREATE RESULT LABEL
# -------------------------------------------------------------------

def create_anomaly_labels(
    predictions: np.ndarray,
) -> np.ndarray:
    """
    Convert Isolation Forest predictions into readable labels.

    Isolation Forest:
        1  = normal
        -1 = anomaly
    """

    return np.where(
        predictions == -1,
        "Anomaly",
        "Normal",
    )


# -------------------------------------------------------------------
# CREATE RISK CATEGORIES
# -------------------------------------------------------------------

def create_anomaly_risk_category(
    anomaly_scores: np.ndarray,
) -> np.ndarray:
    """
    Convert anomaly score into a simple risk category.

    Categories:

        0-30   -> Low
        31-60  -> Medium
        61-100 -> High

    These thresholds are a transparent business interpretation
    of the anomaly score and are not probabilities.
    """

    return np.select(
        [
            anomaly_scores <= 30,
            anomaly_scores <= 60,
        ],
        [
            "Low",
            "Medium",
        ],
        default="High",
    )


# -------------------------------------------------------------------
# CREATE RESULT DATAFRAME
# -------------------------------------------------------------------

def create_results_dataframe(
    full_df: pd.DataFrame,
    predictions: np.ndarray,
    anomaly_scores: np.ndarray,
) -> pd.DataFrame:
    """
    Combine invoice information and anomaly outputs.
    """

    results = full_df.copy()

    results["anomaly_prediction"] = predictions

    results["anomaly_status"] = (
        create_anomaly_labels(
            predictions
        )
    )

    results["anomaly_score"] = (
        np.round(
            anomaly_scores,
            2,
        )
    )

    results["risk_category"] = (
        create_anomaly_risk_category(
            anomaly_scores
        )
    )

    # Sort most suspicious invoices first.
    results = (
        results
        .sort_values(
            "anomaly_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return results


# -------------------------------------------------------------------
# SUMMARY
# -------------------------------------------------------------------

def create_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a summary of anomaly results.
    """

    total = len(results)

    anomaly_count = int(
        (
            results["anomaly_status"]
            == "Anomaly"
        ).sum()
    )

    normal_count = int(
        (
            results["anomaly_status"]
            == "Normal"
        ).sum()
    )

    summary = pd.DataFrame(
        {
            "Metric": [
                "Total Invoices",
                "Normal Invoices",
                "Anomalous Invoices",
                "Anomaly Percentage",
                "Low Risk",
                "Medium Risk",
                "High Risk",
                "Average Anomaly Score",
                "Maximum Anomaly Score",
            ],
            "Value": [
                total,
                normal_count,
                anomaly_count,
                (
                    anomaly_count / total * 100
                    if total > 0
                    else 0
                ),
                int(
                    (
                        results["risk_category"]
                        == "Low"
                    ).sum()
                ),
                int(
                    (
                        results["risk_category"]
                        == "Medium"
                    ).sum()
                ),
                int(
                    (
                        results["risk_category"]
                        == "High"
                    ).sum()
                ),
                results[
                    "anomaly_score"
                ].mean(),
                results[
                    "anomaly_score"
                ].max(),
            ],
        }
    )

    return summary


# -------------------------------------------------------------------
# FEATURE IMPORTANCE / CONTRIBUTION NOTE
# -------------------------------------------------------------------

def create_feature_documentation() -> pd.DataFrame:
    """
    Create documentation describing the anomaly features.

    Isolation Forest does not provide conventional supervised
    feature_importances_ in the same way as Random Forest.

    Therefore this file documents the feature set rather than
    pretending to calculate feature importance.
    """

    features = get_anomaly_features()

    descriptions = {
        "invoice_quantity":
            "Quantity recorded on the invoice.",

        "invoice_dollars":
            "Dollar value recorded on the invoice.",

        "Freight":
            "Freight amount associated with the invoice.",

        "total_item_quantity":
            "Aggregated purchase quantity linked to the PO.",

        "total_item_dollars":
            "Aggregated purchase dollar value linked to the PO.",

        "average_purchase_price":
            "Average purchase price for linked records.",

        "number_of_purchase_lines":
            "Number of purchase transaction lines.",

        "number_of_brands":
            "Number of distinct brands in linked purchases.",

        "quantity_difference":
            "Difference between invoice and purchase quantity.",

        "absolute_quantity_difference":
            "Absolute quantity difference.",

        "quantity_difference_percentage":
            "Percentage quantity difference.",

        "dollar_difference":
            "Difference between invoice and purchase dollars.",

        "absolute_dollar_difference":
            "Absolute dollar difference.",

        "dollar_difference_percentage":
            "Percentage dollar difference.",

        "invoice_cost_per_unit":
            "Invoice dollars divided by invoice quantity.",

        "purchase_cost_per_unit":
            "Purchase dollars divided by purchase quantity.",

        "freight_percentage":
            "Freight as a percentage of invoice dollars.",

        "freight_per_unit":
            "Freight cost per invoice unit.",

        "invoice_plus_freight":
            "Invoice dollars plus freight.",

        "po_to_invoice_days":
            "Days between purchase order and invoice.",

        "invoice_year":
            "Calendar year of invoice.",

        "invoice_month":
            "Calendar month of invoice.",

        "invoice_quarter":
            "Calendar quarter of invoice.",

        "invoice_day_of_week":
            "Day of week of invoice.",

        "vendor_invoice_count":
            "Number of invoices associated with the vendor.",

        "vendor_average_invoice_value":
            "Average invoice value for the vendor.",

        "vendor_average_freight":
            "Average freight amount for the vendor.",

        "vendor_average_freight_percentage":
            "Average freight percentage for the vendor.",

        "invoice_vs_vendor_average_percentage":
            "Percentage deviation from vendor's average invoice value.",

        "freight_vs_vendor_average_percentage":
            "Percentage deviation from vendor's average freight.",
    }

    return pd.DataFrame(
        {
            "Feature": features,
            "Description": [
                descriptions.get(
                    feature,
                    "Engineered analytical feature."
                )
                for feature in features
            ],
        }
    )


# -------------------------------------------------------------------
# SAVE MODEL
# -------------------------------------------------------------------

def save_pipeline(
    pipeline: Pipeline,
) -> None:
    """
    Save complete anomaly-detection pipeline.
    """

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    print(
        f"\nAnomaly pipeline saved to:\n{MODEL_PATH}"
    )


# -------------------------------------------------------------------
# DISPLAY TOP ANOMALIES
# -------------------------------------------------------------------

def display_top_anomalies(
    results: pd.DataFrame,
    number: int = 15,
) -> None:
    """
    Display the most suspicious invoices.
    """

    useful_columns = [
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
        "risk_category",
    ]

    available_columns = [
        column
        for column in useful_columns
        if column in results.columns
    ]

    print(
        "\nTop suspicious invoices:"
    )

    print(
        results[
            available_columns
        ]
        .head(number)
        .to_string(index=False)
    )


# -------------------------------------------------------------------
# MAIN TRAINING FUNCTION
# -------------------------------------------------------------------

def train() -> None:
    """
    Complete Isolation Forest training workflow.
    """

    print("=" * 70)
    print(
        "INVOICE ANOMALY DETECTION - ISOLATION FOREST"
    )
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    X, full_df = load_anomaly_data()

    # ---------------------------------------------------------------
    # Create pipeline
    # ---------------------------------------------------------------

    print(
        "\nCreating anomaly detection pipeline..."
    )

    pipeline = create_pipeline(
        X
    )

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------

    print(
        "\nTraining Isolation Forest..."
    )

    pipeline.fit(
        X
    )

    print(
        "Training completed."
    )

    # ---------------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------------

    print(
        "\nGenerating anomaly predictions..."
    )

    predictions = pipeline.predict(
        X
    )

    # ---------------------------------------------------------------
    # Anomaly scores
    # ---------------------------------------------------------------

    anomaly_scores = calculate_anomaly_score(
        pipeline,
        X,
    )

    # ---------------------------------------------------------------
    # Result DataFrame
    # ---------------------------------------------------------------

    results = create_results_dataframe(
        full_df=full_df,
        predictions=predictions,
        anomaly_scores=anomaly_scores,
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    summary = create_summary(
        results
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ANOMALY DETECTION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Display top anomalies
    # ---------------------------------------------------------------

    display_top_anomalies(
        results,
        number=15,
    )

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------

    results.to_csv(
        RESULTS_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # ---------------------------------------------------------------
    # Save feature documentation
    # ---------------------------------------------------------------

    feature_documentation = (
        create_feature_documentation()
    )

    feature_documentation_path = (
        RESULTS_DIR
        / "anomaly_feature_documentation.csv"
    )

    feature_documentation.to_csv(
        feature_documentation_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Save complete pipeline
    # ---------------------------------------------------------------

    save_pipeline(
        pipeline
    )

    # ---------------------------------------------------------------
    # Final output
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ANOMALY DETECTION COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"Total invoices analyzed: "
        f"{len(results):,}"
    )

    anomaly_count = int(
        (
            results["anomaly_status"]
            == "Anomaly"
        ).sum()
    )

    print(
        f"Anomalies detected: "
        f"{anomaly_count:,}"
    )

    print(
        f"Anomaly rate: "
        f"{(anomaly_count / len(results)) * 100:.2f}%"
    )

    print(
        f"\nSaved model:"
    )

    print(
        MODEL_PATH
    )

    print(
        f"\nSaved invoice results:"
    )

    print(
        RESULTS_PATH
    )

    print(
        f"\nSaved summary:"
    )

    print(
        SUMMARY_PATH
    )

    print(
        f"\nSaved feature documentation:"
    )

    print(
        feature_documentation_path
    )


# -------------------------------------------------------------------
# SCRIPT ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    train()