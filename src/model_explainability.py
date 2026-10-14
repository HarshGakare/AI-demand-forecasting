from pathlib import Path
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lightgbm_demand_model.joblib"
)

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features_train.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

IMPORTANCE_PATH = (
    REPORTS_DIR / "shap_feature_importance.csv"
)

SUMMARY_PLOT_PATH = (
    FIGURES_DIR / "shap_summary.png"
)

BAR_PLOT_PATH = (
    FIGURES_DIR / "shap_feature_importance.png"
)

SAMPLE_SIZE = 2000
RANDOM_STATE = 42
TOP_FEATURES = 20


# Rossmann columns commonly stored as categorical features.
# The list is filtered to the features actually used by the model.
KNOWN_CATEGORICAL_COLUMNS = [
    "StateHoliday",
    "StoreType",
    "Assortment",
    "PromoInterval",
]


# --------------------------------------------------
# Load model and align features
# --------------------------------------------------

def load_explainability_data():
    """Load the trained model and align input columns and categories."""

    print("Loading model and feature data...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run the model training step first."
        )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURES_PATH}\n"
            "Run feature engineering first."
        )

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(FEATURES_PATH)

    if "Sales" not in df.columns:
        raise ValueError(
            "The feature dataset must contain a 'Sales' column."
        )

    # Obtain the LightGBM Booster regardless of the saved wrapper type.
    if hasattr(model, "booster_"):
        booster = model.booster_
    elif hasattr(model, "feature_name"):
        booster = model
    else:
        raise TypeError(
            "The saved model does not expose LightGBM feature metadata."
        )

    feature_names = booster.feature_name()

    missing_features = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "The feature CSV is missing columns required by the model: "
            f"{missing_features}"
        )

    # Preserve the exact feature order used during training.
    X = df.loc[:, feature_names].copy()
    y = pd.to_numeric(df["Sales"], errors="coerce")

    # Use the model's saved category levels.
    saved_categories = getattr(
        booster,
        "pandas_categorical",
        None,
    )

    if saved_categories:
        categorical_columns = [
            column
            for column in KNOWN_CATEGORICAL_COLUMNS
            if column in feature_names
        ]

        if len(categorical_columns) != len(saved_categories):
            raise ValueError(
                "\nCould not safely align LightGBM categorical metadata.\n"
                f"Model expects {len(saved_categories)} categorical "
                f"features, but the configured list matched "
                f"{len(categorical_columns)}.\n"
                f"Matched columns: {categorical_columns}\n"
                f"Model features: {feature_names}\n"
                "Do not guess the mapping. Check which columns were "
                "converted to category during model training."
            )

        for column, categories in zip(
            categorical_columns,
            saved_categories,
        ):
            # CSV-loaded values and saved categories may have different
            # Python types. Normalize both to strings for alignment.
            category_labels = [str(value) for value in categories]

            values = X[column].astype("string")
            X[column] = pd.Categorical(
                values,
                categories=category_labels,
            )

            # A value not present in the saved category list becomes
            # missing. Report this so unexpected levels are visible.
            unknown_count = int(
                (
                    values.notna()
                    & X[column].isna()
                ).sum()
            )

            if unknown_count:
                warnings.warn(
                    f"{column}: {unknown_count} values were not present "
                    "in the model's saved categories and were treated "
                    "as missing.",
                    stacklevel=2,
                )

        print(
            "Categorical columns aligned:",
            ", ".join(categorical_columns),
        )

    # Drop rows with missing target values. Do not drop feature rows
    # just because an input feature is missing: LightGBM can handle it.
    valid_target = y.notna()
    X = X.loc[valid_target].copy()
    y = y.loc[valid_target].copy()

    if X.empty:
        raise ValueError(
            "No usable rows remain after removing missing target values."
        )

    # Use a reproducible sample to keep SHAP runtime manageable.
    sample_size = min(SAMPLE_SIZE, len(X))

    X = X.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    ).copy()

    y = y.loc[X.index].copy()

    print(f"Rows explained: {len(X):,}")
    print(f"Features: {len(X.columns)}")

    return model, X, y


# --------------------------------------------------
# Calculate SHAP values
# --------------------------------------------------

def calculate_shap_values(model, X):
    """Calculate SHAP values for the trained LightGBM model."""

    print("Calculating SHAP values...")

    explainer = shap.TreeExplainer(model)

    # Disable the additivity check because certain SHAP/LightGBM
    # version combinations can produce tiny numerical discrepancies.
    try:
        shap_result = explainer.shap_values(
            X,
            check_additivity=False,
        )
    except TypeError:
        # Compatibility fallback for older SHAP versions.
        shap_result = explainer.shap_values(X)

    # Older SHAP versions may return a list for multiple outputs.
    if isinstance(shap_result, list):
        if len(shap_result) != 1:
            raise ValueError(
                "Expected one regression output, but SHAP returned "
                f"{len(shap_result)} outputs."
            )
        shap_result = shap_result[0]

    # Newer SHAP versions may return an Explanation object.
    if hasattr(shap_result, "values"):
        shap_result = shap_result.values

    shap_values = np.asarray(shap_result)

    # Regression SHAP values should have shape:
    # rows x features.
    if shap_values.ndim == 3:
        if shap_values.shape[-1] == 1:
            shap_values = shap_values[:, :, 0]
        elif shap_values.shape[0] == 1:
            shap_values = shap_values[0]
        else:
            raise ValueError(
                "Unexpected multi-output SHAP shape: "
                f"{shap_values.shape}"
            )

    expected_shape = (len(X), len(X.columns))

    if shap_values.shape != expected_shape:
        raise ValueError(
            f"SHAP returned shape {shap_values.shape}; "
            f"expected {expected_shape}."
        )

    if not np.isfinite(shap_values).all():
        raise ValueError(
            "SHAP values contain NaN or infinite values."
        )

    print("SHAP calculation completed.")

    return explainer, shap_values


# --------------------------------------------------
# Feature importance report
# --------------------------------------------------

def create_feature_importance_report(X, shap_values):
    """Save mean absolute SHAP importance for every feature."""

    print("Creating feature importance report...")

    importance = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "Feature": X.columns,
            "MeanAbsoluteSHAP": importance,
        }
    )

    importance_df = importance_df.sort_values(
        "MeanAbsoluteSHAP",
        ascending=False,
    ).reset_index(drop=True)

    importance_df.insert(
        0,
        "Rank",
        np.arange(1, len(importance_df) + 1),
    )

    importance_df.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    print(f"Saved: {IMPORTANCE_PATH}")

    return importance_df


# --------------------------------------------------
# SHAP summary plot
# --------------------------------------------------

def create_summary_plot(X, shap_values):
    """Save the SHAP beeswarm summary plot."""

    print("Creating SHAP summary plot...")

    plt.figure(figsize=(12, 9))

    shap.summary_plot(
        shap_values,
        X,
        max_display=TOP_FEATURES,
        show=False,
    )

    plt.title(
        "SHAP Summary: Demand Forecasting Model",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(
        SUMMARY_PLOT_PATH,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {SUMMARY_PLOT_PATH}")


# --------------------------------------------------
# SHAP bar plot
# --------------------------------------------------

def create_bar_plot(X, shap_values):
    """Save a global feature importance bar plot."""

    print("Creating SHAP feature importance bar plot...")

    plt.figure(figsize=(12, 9))

    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        max_display=TOP_FEATURES,
        show=False,
    )

    plt.title(
        "Mean Absolute SHAP Feature Importance",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(
        BAR_PLOT_PATH,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {BAR_PLOT_PATH}")


# --------------------------------------------------
# Main execution
# --------------------------------------------------

def main():
    """Run the complete model explainability workflow."""

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model, X, y = load_explainability_data()

    explainer, shap_values = calculate_shap_values(
        model,
        X,
    )

    importance_df = create_feature_importance_report(
        X,
        shap_values,
    )

    create_summary_plot(
        X,
        shap_values,
    )

    create_bar_plot(
        X,
        shap_values,
    )

    print("\n" + "=" * 60)
    print("MODEL EXPLAINABILITY COMPLETE")
    print("=" * 60)

    print("\nTop 15 important features:")
    print(
        importance_df.head(15).to_string(
            index=False,
            float_format=lambda value: f"{value:,.6f}",
        )
    )

    print("\nGenerated files:")
    print(f"- {IMPORTANCE_PATH}")
    print(f"- {SUMMARY_PLOT_PATH}")
    print(f"- {BAR_PLOT_PATH}")

    print(
        "\nInterpretation: Mean absolute SHAP measures how strongly "
        "a feature contributes to predictions on the explained sample. "
        "It does not establish causation."
    )


if __name__ == "__main__":
    main()