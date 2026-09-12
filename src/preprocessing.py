"""
preprocessing.py

Reusable preprocessing utilities for the
AI-Based Vendor Invoice Intelligence and Anomaly Detection System.

This module prepares data for:

1. Freight Cost Prediction
2. Invoice Approval Risk Classification
3. Invoice Anomaly Detection

The preprocessing logic is kept separate from model training so that
the exact same transformations can be used during prediction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -------------------------------------------------------------------
# IMPORT FEATURE ENGINEERING
# -------------------------------------------------------------------

from feature_engineering import (
    build_features,
    get_anomaly_features,
    get_freight_features,
    get_risk_features,
)


# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------

def ensure_features_exist(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """
    Check whether all required model features are present.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    feature_columns : list[str]
        Required model features.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """

    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "The following required features are missing:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )


def clean_model_input(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Select model features and perform basic numeric cleanup.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature-engineered dataframe.

    feature_columns : list[str]
        Columns required by the model.

    Returns
    -------
    pandas.DataFrame
        Clean model input.
    """

    ensure_features_exist(df, feature_columns)

    X = df[feature_columns].copy()

    # Replace infinite values produced by divisions.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return X


# -------------------------------------------------------------------
# NUMERIC PIPELINE
# -------------------------------------------------------------------

def create_numeric_pipeline(
    scale: bool = True,
) -> Pipeline:
    """
    Create preprocessing pipeline for numerical variables.

    Steps:
        1. Median imputation
        2. Optional standardization

    Parameters
    ----------
    scale : bool
        Whether to standardize numerical values.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """

    steps: list[tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(strategy="median"),
        )
    ]

    if scale:
        steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    return Pipeline(steps=steps)


# -------------------------------------------------------------------
# CATEGORICAL PIPELINE
# -------------------------------------------------------------------

def create_categorical_pipeline() -> Pipeline:
    """
    Create preprocessing pipeline for categorical variables.

    Steps:
        1. Most-frequent imputation
        2. One-hot encoding

    Returns
    -------
    sklearn.pipeline.Pipeline
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )


# -------------------------------------------------------------------
# GENERIC PREPROCESSOR
# -------------------------------------------------------------------

def create_preprocessor(
    df: pd.DataFrame,
    feature_columns: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Create a ColumnTransformer based on the supplied features.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature-engineered dataframe.

    feature_columns : list[str]
        Model input features.

    scale_numeric : bool
        Whether numerical features should be standardized.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Configured preprocessing transformer.
    """

    X = clean_model_input(
        df,
        feature_columns,
    )

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                create_numeric_pipeline(
                    scale=scale_numeric
                ),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                create_categorical_pipeline(),
                categorical_columns,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# -------------------------------------------------------------------
# FREIGHT PREPROCESSOR
# -------------------------------------------------------------------

def create_freight_preprocessor(
    df: pd.DataFrame,
) -> ColumnTransformer:
    """
    Create preprocessing transformer for freight prediction.

    Tree-based models do not require scaling, but the transformer
    is designed so that the complete preprocessing logic can be
    attached to a model pipeline consistently.
    """

    features = get_freight_features()

    return create_preprocessor(
        df=df,
        feature_columns=features,
        scale_numeric=False,
    )


# -------------------------------------------------------------------
# RISK PREPROCESSOR
# -------------------------------------------------------------------

def create_risk_preprocessor(
    df: pd.DataFrame,
) -> ColumnTransformer:
    """
    Create preprocessing transformer for approval-risk
    classification.

    Numeric features are standardized so that the same
    preprocessing pipeline can also support linear models
    if required later.
    """

    features = get_risk_features()

    return create_preprocessor(
        df=df,
        feature_columns=features,
        scale_numeric=True,
    )


# -------------------------------------------------------------------
# ANOMALY PREPROCESSOR
# -------------------------------------------------------------------

def create_anomaly_preprocessor(
    df: pd.DataFrame,
) -> ColumnTransformer:
    """
    Create preprocessing transformer for anomaly detection.

    Isolation Forest is tree-based, so scaling is not mandatory.
    We nevertheless keep preprocessing centralized and consistent.
    """

    features = get_anomaly_features()

    return create_preprocessor(
        df=df,
        feature_columns=features,
        scale_numeric=False,
    )


# -------------------------------------------------------------------
# PREPARE FREIGHT DATA
# -------------------------------------------------------------------

def prepare_freight_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for freight prediction.

    Target:
        Freight

    Returns
    -------
    X : pandas.DataFrame
        Freight model features.

    y : pandas.Series
        Freight target.
    """

    required_target = "Freight"

    if required_target not in df.columns:
        raise ValueError(
            "Freight target column is missing."
        )

    feature_columns = get_freight_features()

    X = clean_model_input(
        df,
        feature_columns,
    )

    y = pd.to_numeric(
        df[required_target],
        errors="coerce",
    )

    # Keep only rows with valid target values.
    valid_rows = y.notna()

    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    return X, y


# -------------------------------------------------------------------
# PREPARE RISK DATA
# -------------------------------------------------------------------

def prepare_risk_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for invoice approval-risk classification.

    Target:
        approval_target

    Returns
    -------
    X : pandas.DataFrame
        Risk model features.

    y : pandas.Series
        Binary approval target.
    """

    target = "approval_target"

    if target not in df.columns:
        raise ValueError(
            "approval_target is missing. "
            "Run build_features(..., create_target=True, "
            "remove_missing_target=True) first."
        )

    feature_columns = get_risk_features()

    X = clean_model_input(
        df,
        feature_columns,
    )

    y = pd.to_numeric(
        df[target],
        errors="coerce",
    )

    valid_rows = y.notna()

    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int)

    return X, y


# -------------------------------------------------------------------
# PREPARE ANOMALY DATA
# -------------------------------------------------------------------

def prepare_anomaly_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare invoice data for unsupervised anomaly detection.

    No target is required.

    Returns
    -------
    pandas.DataFrame
        Features for Isolation Forest.
    """

    feature_columns = get_anomaly_features()

    return clean_model_input(
        df,
        feature_columns,
    )


# -------------------------------------------------------------------
# CREATE FULL DATASETS DIRECTLY FROM RAW INVOICE DATA
# -------------------------------------------------------------------

def prepare_freight_dataset(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Complete freight preparation pipeline starting from the
    invoice-level raw dataframe.
    """

    feature_df = build_features(
        raw_df,
        create_target=False,
    )

    return prepare_freight_data(feature_df)


def prepare_risk_dataset(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Complete risk classification preparation pipeline.

    Only invoices with observed Approval information are retained.
    """

    feature_df = build_features(
        raw_df,
        create_target=True,
        remove_missing_target=True,
    )

    return prepare_risk_data(feature_df)


def prepare_anomaly_dataset(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Complete anomaly-detection preparation pipeline.

    All invoices are eligible because anomaly detection is
    unsupervised.
    """

    feature_df = build_features(
        raw_df,
        create_target=False,
    )

    return prepare_anomaly_data(feature_df)


# -------------------------------------------------------------------
# MODEL FEATURE INFORMATION
# -------------------------------------------------------------------

def get_feature_summary(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Return a compact feature summary.

    Useful for debugging and documentation.
    """

    ensure_features_exist(
        df,
        feature_columns,
    )

    rows: list[dict[str, Any]] = []

    for column in feature_columns:

        series = df[column]

        rows.append(
            {
                "feature": column,
                "dtype": str(series.dtype),
                "missing_values": int(series.isna().sum()),
                "unique_values": int(series.nunique()),
            }
        )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# COMPLETE PREPROCESSING TEST
# -------------------------------------------------------------------

def main() -> None:
    """
    Test all three preprocessing pipelines.

    Run from project root:

        python src/preprocessing.py
    """

    # Local import allows direct execution from src/.
    from data_loader import (
        load_invoice_purchase_summary,
        validate_invoice_dataset,
    )

    print("=" * 70)
    print("PREPROCESSING PIPELINE TEST")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    print("\n1. Loading database data...")

    raw_df = load_invoice_purchase_summary()

    validate_invoice_dataset(
        raw_df
    )

    print(
        f"Loaded {len(raw_df):,} invoice records."
    )

    # ---------------------------------------------------------------
    # Create common features
    # ---------------------------------------------------------------

    print("\n2. Creating common features...")

    feature_df = build_features(
        raw_df,
        create_target=False,
    )

    print(
        f"Feature-engineered dataset: "
        f"{feature_df.shape[0]:,} rows × "
        f"{feature_df.shape[1]} columns"
    )

    # ---------------------------------------------------------------
    # Freight
    # ---------------------------------------------------------------

    print("\n3. Preparing freight dataset...")

    X_freight, y_freight = prepare_freight_data(
        feature_df
    )

    print(
        f"Freight X shape: {X_freight.shape}"
    )

    print(
        f"Freight y shape: {y_freight.shape}"
    )

    print(
        f"Freight target mean: "
        f"{y_freight.mean():.2f}"
    )

    # ---------------------------------------------------------------
    # Risk
    # ---------------------------------------------------------------

    print("\n4. Preparing approval-risk dataset...")

    risk_df = build_features(
        raw_df,
        create_target=True,
        remove_missing_target=True,
    )

    X_risk, y_risk = prepare_risk_data(
        risk_df
    )

    print(
        f"Risk X shape: {X_risk.shape}"
    )

    print(
        f"Risk y shape: {y_risk.shape}"
    )

    print("\nRisk target distribution:")

    print(
        y_risk.value_counts()
        .sort_index()
        .to_string()
    )

    # ---------------------------------------------------------------
    # Anomaly Detection
    # ---------------------------------------------------------------

    print(
        "\n5. Preparing anomaly-detection dataset..."
    )

    X_anomaly = prepare_anomaly_data(
        feature_df
    )

    print(
        f"Anomaly X shape: {X_anomaly.shape}"
    )

    # ---------------------------------------------------------------
    # Preprocessors
    # ---------------------------------------------------------------

    print(
        "\n6. Creating sklearn preprocessing objects..."
    )

    freight_preprocessor = create_freight_preprocessor(
        feature_df
    )

    risk_preprocessor = create_risk_preprocessor(
        risk_df
    )

    anomaly_preprocessor = create_anomaly_preprocessor(
        feature_df
    )

    print(
        "Freight preprocessor:",
        type(freight_preprocessor).__name__,
    )

    print(
        "Risk preprocessor:",
        type(risk_preprocessor).__name__,
    )

    print(
        "Anomaly preprocessor:",
        type(anomaly_preprocessor).__name__,
    )

    # ---------------------------------------------------------------
    # Feature summary
    # ---------------------------------------------------------------

    print(
        "\n7. Risk feature summary:"
    )

    summary = get_feature_summary(
        risk_df,
        get_risk_features(),
    )

    print(
        summary.to_string(index=False)
    )

    print(
        "\nPreprocessing pipeline test completed successfully."
    )


if __name__ == "__main__":
    main()