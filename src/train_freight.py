"""
train_freight.py

Train and compare machine-learning regression models for
freight-cost prediction.

Models:
    1. Linear Regression
    2. Decision Tree Regressor
    3. Random Forest Regressor
    4. Gradient Boosting Regressor

Evaluation metrics:
    - MAE
    - RMSE
    - R2

The best model is selected using test-set RMSE and saved as
a complete sklearn Pipeline containing both preprocessing and
the trained model.

Output:
    models/freight_prediction_pipeline.pkl
    models/freight_model_results.csv
    models/freight_feature_importance.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor


# -------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "Data"
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

# Make src importable when running:
# python src/train_freight.py
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
    get_freight_features,
)

from preprocessing import (  # noqa: E402
    create_freight_preprocessor,
)


# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

RANDOM_STATE = 42

TEST_SIZE = 0.20

TARGET_COLUMN = "Freight"

MODEL_PATH = (
    MODELS_DIR
    / "freight_prediction_pipeline.pkl"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "freight_model_results.csv"
)

FEATURE_IMPORTANCE_PATH = (
    RESULTS_DIR
    / "freight_feature_importance.csv"
)


# -------------------------------------------------------------------
# LOAD AND PREPARE DATA
# -------------------------------------------------------------------

def load_training_data() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
]:
    """
    Load the database and construct the freight-prediction dataset.

    Returns
    -------
    X : pandas.DataFrame
        Model features.

    y : pandas.Series
        Freight target.

    full_df : pandas.DataFrame
        Complete feature-engineered dataframe. This is retained for
        possible future reporting and analysis.
    """

    print("\nLoading invoice data...")

    raw_df = load_invoice_purchase_summary()

    validate_invoice_dataset(
        raw_df
    )

    print(
        f"Loaded {len(raw_df):,} invoice records."
    )

    print("\nCreating feature-engineered dataset...")

    full_df = build_features(
        raw_df,
        create_target=False,
    )

    feature_columns = get_freight_features()

    X = full_df[
        feature_columns
    ].copy()

    y = pd.to_numeric(
        full_df[TARGET_COLUMN],
        errors="coerce",
    )

    # Remove rows where the target is unavailable.
    valid_rows = y.notna()

    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    # Replace infinite values.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    print(
        f"Usable records: {len(X):,}"
    )

    print(
        f"Number of features: {X.shape[1]}"
    )

    return X, y, full_df


# -------------------------------------------------------------------
# MODEL DEFINITIONS
# -------------------------------------------------------------------

def create_models() -> dict[str, Any]:
    """
    Create the regression models.

    Returns
    -------
    dict
        Model name -> sklearn estimator.
    """

    models: dict[str, Any] = {

        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=RANDOM_STATE,
            max_depth=12,
            min_samples_leaf=2,
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=1,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    return models


# -------------------------------------------------------------------
# PREPROCESSOR
# -------------------------------------------------------------------

def create_pipeline(
    X_train: pd.DataFrame,
    model: Any,
) -> Pipeline:
    """
    Create a complete preprocessing + model pipeline.

    The complete pipeline is saved so that inference uses exactly
    the same preprocessing as training.
    """

    preprocessor = create_freight_preprocessor(
        X_train
    )

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
# EVALUATION
# -------------------------------------------------------------------

def evaluate_regression_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Train and evaluate a regression pipeline.

    Returns
    -------
    dict
        Model metrics.
    """

    print(
        f"\nTraining {model_name}..."
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    result = {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

    print(
        f"  MAE  : {mae:.4f}"
    )

    print(
        f"  RMSE : {rmse:.4f}"
    )

    print(
        f"  R²   : {r2:.4f}"
    )

    return result


# -------------------------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------------------------

def extract_feature_importance(
    pipeline: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extract feature importance from a tree-based regression model.

    For Linear Regression, coefficients can be inspected separately.

    Returns
    -------
    pandas.DataFrame
    """

    model = pipeline.named_steps[
        "model"
    ]

    # ---------------------------------------------------------------
    # Tree-based models
    # ---------------------------------------------------------------

    if hasattr(
        model,
        "feature_importances_",
    ):

        importance = model.feature_importances_

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importance,
            }
        )

        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        return importance_df

    # ---------------------------------------------------------------
    # Linear Regression
    # ---------------------------------------------------------------

    if hasattr(
        model,
        "coef_",
    ):

        coefficients = np.asarray(
            model.coef_
        )

        # For one-dimensional regression.
        coefficients = coefficients.reshape(-1)

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Coefficient": coefficients,
                "AbsoluteCoefficient": np.abs(
                    coefficients
                ),
            }
        )

        importance_df = (
            importance_df
            .sort_values(
                "AbsoluteCoefficient",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        return importance_df

    return pd.DataFrame(
        {
            "Feature": feature_names,
        }
    )


# -------------------------------------------------------------------
# RESIDUAL ANALYSIS
# -------------------------------------------------------------------

def calculate_residual_statistics(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Calculate basic residual statistics for the best model.
    """

    predictions = pipeline.predict(
        X_test
    )

    residuals = (
        y_test.to_numpy()
        - predictions
    )

    return {
        "Mean Residual": float(
            np.mean(residuals)
        ),
        "Residual Std": float(
            np.std(residuals)
        ),
        "Maximum Absolute Residual": float(
            np.max(np.abs(residuals))
        ),
    }


# -------------------------------------------------------------------
# SAVE MODEL
# -------------------------------------------------------------------

def save_model(
    pipeline: Pipeline,
    path: Path,
) -> None:
    """
    Save the complete sklearn pipeline.
    """

    joblib.dump(
        pipeline,
        path,
    )

    print(
        f"\nBest freight model saved to:\n{path}"
    )


# -------------------------------------------------------------------
# MAIN TRAINING FUNCTION
# -------------------------------------------------------------------

def train() -> None:
    """
    Complete freight-model training workflow.
    """

    print("=" * 70)
    print(
        "FREIGHT COST PREDICTION - MODEL TRAINING"
    )
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    X, y, full_df = load_training_data()

    feature_names = get_freight_features()

    # ---------------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------------

    print(
        "\nSplitting data into training and test sets..."
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(
        f"Training records: {len(X_train):,}"
    )

    print(
        f"Testing records : {len(X_test):,}"
    )

    # ---------------------------------------------------------------
    # Create models
    # ---------------------------------------------------------------

    models = create_models()

    results: list[dict[str, Any]] = []

    trained_pipelines: dict[
        str,
        Pipeline
    ] = {}

    # ---------------------------------------------------------------
    # Train all models
    # ---------------------------------------------------------------

    for model_name, model in models.items():

        pipeline = create_pipeline(
            X_train,
            model,
        )

        result = evaluate_regression_model(
            model_name=model_name,
            pipeline=pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        results.append(
            result
        )

        trained_pipelines[
            model_name
        ] = pipeline

    # ---------------------------------------------------------------
    # Results table
    # ---------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # Lower RMSE is better.
    results_df = (
        results_df
        .sort_values(
            "RMSE",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    display_df = results_df.copy()

    display_df["MAE"] = display_df[
        "MAE"
    ].round(4)

    display_df["RMSE"] = display_df[
        "RMSE"
    ].round(4)

    display_df["R2"] = display_df[
        "R2"
    ].round(4)

    print(
        display_df.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Select best model
    # ---------------------------------------------------------------

    best_model_name = results_df.iloc[
        0
    ]["Model"]

    best_pipeline = trained_pipelines[
        best_model_name
    ]

    print(
        "\nBest model:"
    )

    print(
        f"  {best_model_name}"
    )

    # ---------------------------------------------------------------
    # Residual statistics
    # ---------------------------------------------------------------

    residual_stats = calculate_residual_statistics(
        best_pipeline,
        X_test,
        y_test,
    )

    print(
        "\nResidual statistics:"
    )

    for name, value in residual_stats.items():
        print(
            f"  {name}: {value:.4f}"
        )

    # ---------------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------------

    importance_df = extract_feature_importance(
        best_pipeline,
        feature_names,
    )

    print(
        "\nTop features:"
    )

    print(
        importance_df.head(10).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    # ---------------------------------------------------------------
    # Save best model
    # ---------------------------------------------------------------

    save_model(
        best_pipeline,
        MODEL_PATH,
    )

    # ---------------------------------------------------------------
    # Training summary
    # ---------------------------------------------------------------

    best_row = results_df.iloc[0]

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"Best Model : {best_model_name}"
    )

    print(
        f"MAE        : {best_row['MAE']:.4f}"
    )

    print(
        f"RMSE       : {best_row['RMSE']:.4f}"
    )

    print(
        f"R²         : {best_row['R2']:.4f}"
    )

    print(
        f"\nModel file:\n{MODEL_PATH}"
    )

    print(
        f"\nResults file:\n{RESULTS_PATH}"
    )

    print(
        f"\nFeature importance:\n{FEATURE_IMPORTANCE_PATH}"
    )


# -------------------------------------------------------------------
# SCRIPT ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    train()