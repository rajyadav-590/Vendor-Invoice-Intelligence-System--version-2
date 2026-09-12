"""
data_loader.py

Centralized database loading utilities for the
AI-Based Vendor Invoice Intelligence and Anomaly Detection System.

Database:
    Data/inventory.db

Tables:
    - vendor_invoice
    - purchases
    - purchase_prices
    - begin_inventory
    - end_inventory
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd


# -------------------------------------------------------------------
# PROJECT PATHS
# -------------------------------------------------------------------

# src/data_loader.py
# parent        -> src
# parent.parent -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "Data"
DATABASE_PATH = DATA_DIR / "inventory.db"


# -------------------------------------------------------------------
# DATABASE CONNECTION
# -------------------------------------------------------------------

def get_connection(db_path: Optional[Path | str] = None) -> sqlite3.Connection:
    """
    Create and return a SQLite database connection.

    Parameters
    ----------
    db_path : Path | str | None
        Optional custom database path.
        When None, the project's default database is used.

    Returns
    -------
    sqlite3.Connection
        SQLite connection object.

    Raises
    ------
    FileNotFoundError
        If the database file does not exist.
    """

    path = Path(db_path) if db_path else DATABASE_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Database not found:\n{path}\n\n"
            f"Please make sure inventory.db is inside the Data folder."
        )

    return sqlite3.connect(path)


# -------------------------------------------------------------------
# GENERIC TABLE LOADER
# -------------------------------------------------------------------

def load_table(
    table_name: str,
    db_path: Optional[Path | str] = None
) -> pd.DataFrame:
    """
    Load an entire SQLite table into a pandas DataFrame.

    Parameters
    ----------
    table_name : str
        Name of the SQLite table.
    db_path : Path | str | None
        Optional custom database path.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the table contents.
    """

    allowed_tables = {
        "vendor_invoice",
        "purchases",
        "purchase_prices",
        "begin_inventory",
        "end_inventory",
    }

    if table_name not in allowed_tables:
        raise ValueError(
            f"Unsupported table '{table_name}'. "
            f"Allowed tables: {sorted(allowed_tables)}"
        )

    connection = get_connection(db_path)

    try:
        query = f'SELECT * FROM "{table_name}"'
        return pd.read_sql_query(query, connection)
    finally:
        connection.close()


# -------------------------------------------------------------------
# INDIVIDUAL TABLE LOADERS
# -------------------------------------------------------------------

def load_vendor_invoices(
    db_path: Optional[Path | str] = None
) -> pd.DataFrame:
    """
    Load the vendor_invoice table.

    Returns
    -------
    pandas.DataFrame
    """

    return load_table("vendor_invoice", db_path)


def load_purchases(
    db_path: Optional[Path | str] = None
) -> pd.DataFrame:
    """
    Load the purchases table.

    Returns
    -------
    pandas.DataFrame
    """

    return load_table("purchases", db_path)


def load_purchase_prices(
    db_path: Optional[Path | str] = None
) -> pd.DataFrame:
    """
    Load the purchase_prices table.

    Returns
    -------
    pandas.DataFrame
    """

    return load_table("purchase_prices", db_path)


def load_begin_inventory(
    db_path: Optional[Path | str] = None
) -> pd.DataFrame:
    """
    Load the begin_inventory table.

    Returns
    -------
    pandas.DataFrame
    """

    return load_table("begin_inventory", db_path)


def load_end_inventory(
    db_path: Optional[Path | str] = None
) -> pd.DataFrame:
    """
    Load the end_inventory table.

    Returns
    -------
    pandas.DataFrame
    """

    return load_table("end_inventory", db_path)


# -------------------------------------------------------------------
# DATABASE INFORMATION
# -------------------------------------------------------------------

def get_table_counts(
    db_path: Optional[Path | str] = None
) -> dict[str, int]:
    """
    Return the number of rows in every project table.

    Returns
    -------
    dict[str, int]
        Mapping of table name -> row count.
    """

    table_names = [
        "vendor_invoice",
        "purchases",
        "purchase_prices",
        "begin_inventory",
        "end_inventory",
    ]

    connection = get_connection(db_path)

    try:
        counts: dict[str, int] = {}

        for table in table_names:
            query = f'SELECT COUNT(*) AS count FROM "{table}"'
            result = pd.read_sql_query(query, connection)

            counts[table] = int(result.loc[0, "count"])

        return counts

    finally:
        connection.close()


# -------------------------------------------------------------------
# JOINED INVOICE DATASET
# -------------------------------------------------------------------

def load_invoice_purchase_summary(
    db_path: Optional[Path | str] = None
) -> pd.DataFrame:
    """
    Build an invoice-level dataset by combining vendor_invoice
    with aggregated purchase information.

    One row in the resulting DataFrame represents one vendor invoice.

    Aggregated purchase features include:
        - total_item_quantity
        - total_item_dollars
        - average_purchase_price
        - number_of_purchase_lines
        - number_of_brands

    Returns
    -------
    pandas.DataFrame
        Invoice-level analytical dataset.
    """

    connection = get_connection(db_path)

    query = """
    SELECT
        vi.VendorNumber,
        vi.VendorName,
        vi.InvoiceDate,
        vi.PONumber,
        vi.PODate,
        vi.PayDate,
        vi.Quantity AS invoice_quantity,
        vi.Dollars AS invoice_dollars,
        vi.Freight,
        vi.Approval,

        COALESCE(p.total_item_quantity, 0) AS total_item_quantity,
        COALESCE(p.total_item_dollars, 0) AS total_item_dollars,
        COALESCE(p.average_purchase_price, 0) AS average_purchase_price,
        COALESCE(p.number_of_purchase_lines, 0) AS number_of_purchase_lines,
        COALESCE(p.number_of_brands, 0) AS number_of_brands

    FROM vendor_invoice AS vi

    LEFT JOIN (
        SELECT
            PONumber,

            SUM(COALESCE(Quantity, 0)) AS total_item_quantity,

            SUM(COALESCE(Dollars, 0)) AS total_item_dollars,

            AVG(COALESCE(PurchasePrice, 0)) AS average_purchase_price,

            COUNT(*) AS number_of_purchase_lines,

            COUNT(DISTINCT Brand) AS number_of_brands

        FROM purchases

        GROUP BY PONumber
    ) AS p

        ON vi.PONumber = p.PONumber
    """

    try:
        df = pd.read_sql_query(query, connection)
    finally:
        connection.close()

    return df


# -------------------------------------------------------------------
# DATA VALIDATION
# -------------------------------------------------------------------

def validate_invoice_dataset(df: pd.DataFrame) -> None:
    """
    Validate the invoice-level dataset.

    Raises
    ------
    ValueError
        When important columns are missing.
    """

    required_columns = {
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
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "The invoice dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )


# -------------------------------------------------------------------
# SIMPLE TEST
# -------------------------------------------------------------------

def main() -> None:
    """
    Basic database test.

    Run:
        python src/data_loader.py
    """

    print("=" * 70)
    print("VENDOR INVOICE INTELLIGENCE - DATABASE TEST")
    print("=" * 70)

    print(f"\nProject root:\n{PROJECT_ROOT}")
    print(f"\nDatabase:\n{DATABASE_PATH}")

    print("\nTable counts:")

    counts = get_table_counts()

    for table_name, count in counts.items():
        print(f"  {table_name:<20} {count:,}")

    print("\nLoading invoice-level dataset...")

    invoice_df = load_invoice_purchase_summary()

    validate_invoice_dataset(invoice_df)

    print(f"\nRows: {len(invoice_df):,}")
    print(f"Columns: {len(invoice_df.columns)}")

    print("\nColumns:")
    for column in invoice_df.columns:
        print(f"  - {column}")

    print("\nFirst 5 rows:")
    print(invoice_df.head().to_string(index=False))

    print("\nApproval distribution:")

    approval_counts = invoice_df["Approval"].value_counts(dropna=False)

    print(approval_counts.to_string())

    print("\nDatabase loading test completed successfully.")


if __name__ == "__main__":
    main()