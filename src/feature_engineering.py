"""
feature_engineering.py

Creates the analytical features used by:
    1. Freight Cost Prediction
    2. Invoice Approval Risk Classification
    3. Invoice Anomaly Detection

Input:
    Invoice-level DataFrame created by data_loader.py

Output:
    Feature-engineered invoice-level DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# BASIC NUMERIC CLEANING
# -------------------------------------------------------------------

def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert important numeric columns to numeric dtype.

    Invalid values are converted to NaN and later handled by
    the missing-value processing step.
    """

    df = df.copy()

    numeric_columns = [
        "invoice_quantity",
        "invoice_dollars",
        "Freight",
        "total_item_quantity",
        "total_item_dollars",
        "average_purchase_price",
        "number_of_purchase_lines",
        "number_of_brands",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


# -------------------------------------------------------------------
# DATE PROCESSING
# -------------------------------------------------------------------

def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date columns and create useful time-based features.

    Created features:
        - invoice_year
        - invoice_month
        - invoice_quarter
        - invoice_day_of_week
        - po_to_invoice_days
        - invoice_to_pay_days
    """

    df = df.copy()

    date_columns = [
        "InvoiceDate",
        "PODate",
        "PayDate",
    ]

    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )

    # ---------------------------------------------------------------
    # Invoice calendar features
    # ---------------------------------------------------------------

    if "InvoiceDate" in df.columns:

        df["invoice_year"] = df["InvoiceDate"].dt.year

        df["invoice_month"] = df["InvoiceDate"].dt.month

        df["invoice_quarter"] = df["InvoiceDate"].dt.quarter

        df["invoice_day_of_week"] = (
            df["InvoiceDate"].dt.dayofweek
        )

    # ---------------------------------------------------------------
    # Purchase order → invoice delay
    # ---------------------------------------------------------------

    if {"PODate", "InvoiceDate"}.issubset(df.columns):

        df["po_to_invoice_days"] = (
            df["InvoiceDate"] - df["PODate"]
        ).dt.days

    # ---------------------------------------------------------------
    # Invoice → payment delay
    # ---------------------------------------------------------------

    if {"InvoiceDate", "PayDate"}.issubset(df.columns):

        df["invoice_to_pay_days"] = (
            df["PayDate"] - df["InvoiceDate"]
        ).dt.days

    return df


# -------------------------------------------------------------------
# INVOICE VS PURCHASE FEATURES
# -------------------------------------------------------------------

def create_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create differences between invoice values and
    aggregated purchase values.

    These are particularly important for invoice-risk analysis.
    """

    df = df.copy()

    # ---------------------------------------------------------------
    # Quantity difference
    # ---------------------------------------------------------------

    df["quantity_difference"] = (
        df["invoice_quantity"]
        - df["total_item_quantity"]
    )

    df["absolute_quantity_difference"] = (
        df["quantity_difference"].abs()
    )

    # ---------------------------------------------------------------
    # Quantity difference percentage
    # ---------------------------------------------------------------

    df["quantity_difference_percentage"] = np.where(
        df["total_item_quantity"] != 0,
        (
            df["quantity_difference"].abs()
            / df["total_item_quantity"].abs()
        ) * 100,
        0,
    )

    # ---------------------------------------------------------------
    # Dollar difference
    # ---------------------------------------------------------------

    df["dollar_difference"] = (
        df["invoice_dollars"]
        - df["total_item_dollars"]
    )

    df["absolute_dollar_difference"] = (
        df["dollar_difference"].abs()
    )

    # ---------------------------------------------------------------
    # Dollar difference percentage
    # ---------------------------------------------------------------

    df["dollar_difference_percentage"] = np.where(
        df["total_item_dollars"] != 0,
        (
            df["dollar_difference"].abs()
            / df["total_item_dollars"].abs()
        ) * 100,
        0,
    )

    return df


# -------------------------------------------------------------------
# COST FEATURES
# -------------------------------------------------------------------

def create_cost_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cost-related features.

    Features:
        - invoice_cost_per_unit
        - purchase_cost_per_unit
        - freight_percentage
        - freight_per_unit
        - invoice_plus_freight
    """

    df = df.copy()

    # ---------------------------------------------------------------
    # Invoice cost per unit
    # ---------------------------------------------------------------

    df["invoice_cost_per_unit"] = np.where(
        df["invoice_quantity"] != 0,
        df["invoice_dollars"]
        / df["invoice_quantity"],
        0,
    )

    # ---------------------------------------------------------------
    # Purchase cost per unit
    # ---------------------------------------------------------------

    df["purchase_cost_per_unit"] = np.where(
        df["total_item_quantity"] != 0,
        df["total_item_dollars"]
        / df["total_item_quantity"],
        0,
    )

    # ---------------------------------------------------------------
    # Freight percentage
    # ---------------------------------------------------------------

    df["freight_percentage"] = np.where(
        df["invoice_dollars"] != 0,
        (
            df["Freight"]
            / df["invoice_dollars"]
        ) * 100,
        0,
    )

    # ---------------------------------------------------------------
    # Freight per unit
    # ---------------------------------------------------------------

    df["freight_per_unit"] = np.where(
        df["invoice_quantity"] != 0,
        df["Freight"]
        / df["invoice_quantity"],
        0,
    )

    # ---------------------------------------------------------------
    # Total invoice cost including freight
    # ---------------------------------------------------------------

    df["invoice_plus_freight"] = (
        df["invoice_dollars"]
        + df["Freight"]
    )

    return df


# -------------------------------------------------------------------
# VENDOR FEATURES
# -------------------------------------------------------------------

def create_vendor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create vendor-level historical features.

    Features:
        - vendor_invoice_count
        - vendor_average_invoice_value
        - vendor_average_freight
        - vendor_average_freight_percentage

    Important:
        Statistics are calculated using the available dataset.
        For production-grade time-series deployment these should
        ideally be calculated using only historical rows.
    """

    df = df.copy()

    if "VendorNumber" not in df.columns:
        return df

    # ---------------------------------------------------------------
    # Number of invoices for each vendor
    # ---------------------------------------------------------------

    vendor_invoice_count = (
        df.groupby("VendorNumber")
        .size()
        .rename("vendor_invoice_count")
    )

    df = df.merge(
        vendor_invoice_count,
        on="VendorNumber",
        how="left",
    )

    # ---------------------------------------------------------------
    # Average invoice value by vendor
    # ---------------------------------------------------------------

    vendor_average_invoice = (
        df.groupby("VendorNumber")["invoice_dollars"]
        .mean()
        .rename("vendor_average_invoice_value")
    )

    df = df.merge(
        vendor_average_invoice,
        on="VendorNumber",
        how="left",
    )

    # ---------------------------------------------------------------
    # Average freight by vendor
    # ---------------------------------------------------------------

    vendor_average_freight = (
        df.groupby("VendorNumber")["Freight"]
        .mean()
        .rename("vendor_average_freight")
    )

    df = df.merge(
        vendor_average_freight,
        on="VendorNumber",
        how="left",
    )

    # ---------------------------------------------------------------
    # Average freight percentage by vendor
    # ---------------------------------------------------------------

    vendor_average_freight_percentage = (
        df.groupby("VendorNumber")["freight_percentage"]
        .mean()
        .rename("vendor_average_freight_percentage")
    )

    df = df.merge(
        vendor_average_freight_percentage,
        on="VendorNumber",
        how="left",
    )

    return df


# -------------------------------------------------------------------
# VENDOR DEVIATION FEATURES
# -------------------------------------------------------------------

def create_vendor_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate how unusual an invoice is relative to its vendor's
    typical invoice behavior.
    """

    df = df.copy()

    if "vendor_average_invoice_value" in df.columns:

        df["invoice_vs_vendor_average"] = (
            df["invoice_dollars"]
            - df["vendor_average_invoice_value"]
        )

        df["invoice_vs_vendor_average_percentage"] = np.where(
            df["vendor_average_invoice_value"] != 0,
            (
                (
                    df["invoice_dollars"]
                    - df["vendor_average_invoice_value"]
                ).abs()
                / df["vendor_average_invoice_value"].abs()
            ) * 100,
            0,
        )

    if "vendor_average_freight" in df.columns:

        df["freight_vs_vendor_average"] = (
            df["Freight"]
            - df["vendor_average_freight"]
        )

        df["freight_vs_vendor_average_percentage"] = np.where(
            df["vendor_average_freight"] != 0,
            (
                (
                    df["Freight"]
                    - df["vendor_average_freight"]
                ).abs()
                / df["vendor_average_freight"].abs()
            ) * 100,
            0,
        )

    return df


# -------------------------------------------------------------------
# APPROVAL TARGET
# -------------------------------------------------------------------

def create_approval_target(
    df: pd.DataFrame,
    remove_missing_target: bool = True,
) -> pd.DataFrame:
    """
    Create the classification target for historical approval.

    Target:
        approval_target = 1  -> Approval exists
        approval_target = 0  -> Approval does not exist

    Note:
        Rows without historical approval information should not
        automatically be treated as confirmed negative examples
        when training a supervised model.

    For this reason, when remove_missing_target=True, rows without
    an observed Approval value are removed from the classification
    training dataset.
    """

    df = df.copy()

    if "Approval" not in df.columns:
        raise ValueError(
            "Approval column is required to create the "
            "classification target."
        )

    # Determine whether approval information exists.
    approval_present = (
        df["Approval"]
        .notna()
        &
        (
            df["Approval"]
            .astype(str)
            .str.strip()
            .ne("")
        )
    )

    df["approval_target"] = approval_present.astype(int)

    if remove_missing_target:
        # For a true supervised model we need observed historical
        # outcomes. The feature-engineering stage therefore keeps
        # only rows where Approval is present.
        df = df.loc[approval_present].copy()

    return df


# -------------------------------------------------------------------
# MISSING VALUE HANDLING
# -------------------------------------------------------------------

def basic_missing_value_handling(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Perform safe basic missing-value handling.

    Numeric features:
        Filled with median.

    Categorical columns:
        Filled with 'Unknown'.
    """

    df = df.copy()

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    for column in numeric_columns:

        median_value = df[column].median()

        if pd.isna(median_value):
            median_value = 0

        df[column] = df[column].fillna(median_value)

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:

        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
        )

    return df


# -------------------------------------------------------------------
# REMOVE INVALID NUMERIC VALUES
# -------------------------------------------------------------------

def clean_infinite_values(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Replace positive/negative infinity with NaN.

    This is important because division-based features can otherwise
    create infinity values.
    """

    df = df.copy()

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    return df


# -------------------------------------------------------------------
# MAIN FEATURE ENGINEERING PIPELINE
# -------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    create_target: bool = False,
    remove_missing_target: bool = False,
) -> pd.DataFrame:
    """
    Execute the complete feature-engineering pipeline.

    Parameters
    ----------
    df : pandas.DataFrame
        Invoice-level dataset from data_loader.py.

    create_target : bool
        Whether to create approval_target.

    remove_missing_target : bool
        Whether to remove invoices that have no historical approval
        information.

    Returns
    -------
    pandas.DataFrame
        Feature-engineered dataset.
    """

    if df.empty:
        raise ValueError(
            "Input DataFrame is empty."
        )

    required_columns = [
        "invoice_quantity",
        "invoice_dollars",
        "Freight",
        "total_item_quantity",
        "total_item_dollars",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    # ---------------------------------------------------------------
    # Step 1: Numeric cleaning
    # ---------------------------------------------------------------

    df = clean_numeric_columns(df)

    # ---------------------------------------------------------------
    # Step 2: Date features
    # ---------------------------------------------------------------

    df = create_date_features(df)

    # ---------------------------------------------------------------
    # Step 3: Difference features
    # ---------------------------------------------------------------

    df = create_difference_features(df)

    # ---------------------------------------------------------------
    # Step 4: Cost features
    # ---------------------------------------------------------------

    df = create_cost_features(df)

    # ---------------------------------------------------------------
    # Step 5: Vendor statistics
    # ---------------------------------------------------------------

    df = create_vendor_features(df)

    # ---------------------------------------------------------------
    # Step 6: Vendor deviation features
    # ---------------------------------------------------------------

    df = create_vendor_deviation_features(df)

    # ---------------------------------------------------------------
    # Step 7: Approval target
    # ---------------------------------------------------------------

    if create_target:
        df = create_approval_target(
            df,
            remove_missing_target=remove_missing_target,
        )

    # ---------------------------------------------------------------
    # Step 8: Infinite-value cleanup
    # ---------------------------------------------------------------

    df = clean_infinite_values(df)

    # ---------------------------------------------------------------
    # Step 9: Missing-value handling
    # ---------------------------------------------------------------

    df = basic_missing_value_handling(df)

    return df


# -------------------------------------------------------------------
# FEATURE LISTS
# -------------------------------------------------------------------

def get_freight_features() -> list[str]:
    """
    Return features used for freight-cost prediction.

    The target is Freight.
    """

    return [
        "invoice_quantity",
        "invoice_dollars",
        "total_item_quantity",
        "total_item_dollars",
        "quantity_difference",
        "absolute_quantity_difference",
        "dollar_difference",
        "absolute_dollar_difference",
        "invoice_cost_per_unit",
        "purchase_cost_per_unit",
        "number_of_purchase_lines",
        "number_of_brands",
    ]


def get_risk_features() -> list[str]:
    """
    Return features used for invoice approval-risk classification.
    """

    return [
        "invoice_quantity",
        "invoice_dollars",
        "Freight",
        "total_item_quantity",
        "total_item_dollars",
        "average_purchase_price",
        "number_of_purchase_lines",
        "number_of_brands",
        "quantity_difference",
        "absolute_quantity_difference",
        "quantity_difference_percentage",
        "dollar_difference",
        "absolute_dollar_difference",
        "dollar_difference_percentage",
        "invoice_cost_per_unit",
        "purchase_cost_per_unit",
        "freight_percentage",
        "freight_per_unit",
        "invoice_plus_freight",
        "po_to_invoice_days",
        "invoice_to_pay_days",
        "invoice_year",
        "invoice_month",
        "invoice_quarter",
        "invoice_day_of_week",
        "vendor_invoice_count",
        "vendor_average_invoice_value",
        "vendor_average_freight",
        "vendor_average_freight_percentage",
        "invoice_vs_vendor_average",
        "invoice_vs_vendor_average_percentage",
        "freight_vs_vendor_average",
        "freight_vs_vendor_average_percentage",
    ]


def get_anomaly_features() -> list[str]:
    """
    Return features used by the Isolation Forest anomaly detector.
    """

    return [
        "invoice_quantity",
        "invoice_dollars",
        "Freight",
        "total_item_quantity",
        "total_item_dollars",
        "average_purchase_price",
        "number_of_purchase_lines",
        "number_of_brands",
        "quantity_difference",
        "absolute_quantity_difference",
        "quantity_difference_percentage",
        "dollar_difference",
        "absolute_dollar_difference",
        "dollar_difference_percentage",
        "invoice_cost_per_unit",
        "purchase_cost_per_unit",
        "freight_percentage",
        "freight_per_unit",
        "invoice_plus_freight",
        "po_to_invoice_days",
        "invoice_year",
        "invoice_month",
        "invoice_quarter",
        "invoice_day_of_week",
        "vendor_invoice_count",
        "vendor_average_invoice_value",
        "vendor_average_freight",
        "vendor_average_freight_percentage",
        "invoice_vs_vendor_average_percentage",
        "freight_vs_vendor_average_percentage",
    ]


# -------------------------------------------------------------------
# SIMPLE TEST
# -------------------------------------------------------------------

def main() -> None:
    """
    Test the complete feature-engineering pipeline.

    Run from project root:

        python src/feature_engineering.py
    """

    # Import here so the script can be run directly from the
    # project root.
    from data_loader import (
        load_invoice_purchase_summary,
        validate_invoice_dataset,
    )

    print("=" * 70)
    print("FEATURE ENGINEERING TEST")
    print("=" * 70)

    print("\nLoading invoice data...")

    df = load_invoice_purchase_summary()

    validate_invoice_dataset(df)

    print(f"Original rows: {len(df):,}")
    print(f"Original columns: {len(df.columns)}")

    print("\nCreating features...")

    features = build_features(
        df,
        create_target=False,
    )

    print(f"Feature rows: {len(features):,}")
    print(f"Feature columns: {len(features.columns)}")

    print("\nGenerated features:")

    for column in features.columns:
        print(f"  - {column}")

    print("\nFreight model features:")

    for column in get_freight_features():
        print(f"  - {column}")

    print("\nRisk model features:")

    for column in get_risk_features():
        print(f"  - {column}")

    print("\nAnomaly model features:")

    for column in get_anomaly_features():
        print(f"  - {column}")

    print("\nSample engineered data:")

    preview_columns = [
        "invoice_dollars",
        "Freight",
        "total_item_dollars",
        "dollar_difference",
        "dollar_difference_percentage",
        "freight_percentage",
        "invoice_cost_per_unit",
        "purchase_cost_per_unit",
    ]

    available_columns = [
        column
        for column in preview_columns
        if column in features.columns
    ]

    print(
        features[available_columns]
        .head(10)
        .to_string(index=False)
    )

    print("\nFeature engineering completed successfully.")


if __name__ == "__main__":
    main()