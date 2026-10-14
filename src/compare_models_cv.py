
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.evaluation import evaluate_forecast


ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = ROOT / "data" / "processed" / "features_train.csv"
PARAMS_PATH = ROOT / "reports" / "lightgbm_best_params.csv"

OUTPUT_PATH = ROOT / "reports" / "model_comparison_cv.csv"
SUMMARY_PATH = ROOT / "reports" / "model_comparison_cv_summary.csv"

DATE_COLUMN = "Date"
TARGET_COLUMN = "Sales"
STORE_COLUMN = "Store"

N_FOLDS = 4
VALIDATION_DAYS = 30
RANDOM_STATE = 42
MAX_RF_ROWS = 250_000


def load_data():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature data not found: {FEATURES_PATH}"
        )

    df = pd.read_csv(FEATURES_PATH, low_memory=False)

    required = {DATE_COLUMN, TARGET_COLUMN, STORE_COLUMN}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in feature data: {sorted(missing)}"
        )

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce",
    )

    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    df = df.dropna(
        subset=[DATE_COLUMN, TARGET_COLUMN, STORE_COLUMN]
    )

    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    # Sales = 0 is valid, for example when a store is closed.
    df = df[df[TARGET_COLUMN] >= 0].copy()

    if df.empty:
        raise ValueError("No usable training rows remain after cleaning.")

    return df


def load_best_params():
    if not PARAMS_PATH.exists():
        print(
            "WARNING: Best LightGBM parameters not found. "
            "Using conservative defaults."
        )
        return {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "min_split_gain": 0.0,
        }

    params_df = pd.read_csv(PARAMS_PATH)

    if len(params_df.columns) < 2:
        raise ValueError(
            "Unexpected format in lightgbm_best_params.csv"
        )

    params = {}

    for _, row in params_df.iterrows():
        name = str(row.iloc[0]).strip()

        if not name or pd.isna(row.iloc[1]):
            continue

        params[name] = row.iloc[1]

    integer_params = [
        "n_estimators",
        "num_leaves",
        "max_depth",
        "min_child_samples",
    ]

    float_params = [
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "min_split_gain",
    ]

    for name in integer_params:
        if name in params:
            params[name] = int(float(params[name]))

    for name in float_params:
        if name in params:
            params[name] = float(params[name])

    defaults = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "min_split_gain": 0.0,
    }

    defaults.update(params)

    return defaults


def get_feature_columns(df):
    excluded = {
        TARGET_COLUMN,
        DATE_COLUMN,
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    if not feature_columns:
        raise ValueError("No feature columns available.")

    return feature_columns



def prepare_features(df, feature_columns):
    """Prepare numeric and categorical columns for model training."""
    X = df[feature_columns].copy()

    categorical_columns = [
        "StateHoliday",
        "StoreType",
        "Assortment",
        "PromoInterval",
    ]

    for column in X.columns:
        if column in categorical_columns:
            # Treat missing values as their own category.
            X[column] = (
                X[column]
                .astype("string")
                .fillna("__MISSING__")
                .astype(str)
                .astype("category")
            )

        elif (
            pd.api.types.is_object_dtype(X[column])
            or pd.api.types.is_string_dtype(X[column])
        ):
            # Attempt numeric conversion for other string columns.
            converted = pd.to_numeric(X[column], errors="coerce")

            original_nonmissing = X[column].notna()
            conversion_failed = (
                original_nonmissing & converted.isna()
            )

            if conversion_failed.any():
                raise ValueError(
                    f"Unexpected non-numeric feature column: {column}. "
                    "Add it to the categorical column list if appropriate."
                )

            X[column] = converted

    return X



def make_folds(df):
    unique_dates = np.sort(df[DATE_COLUMN].dropna().unique())

    if len(unique_dates) <= N_FOLDS * VALIDATION_DAYS:
        raise ValueError(
            "Insufficient unique dates for the requested CV configuration."
        )

    first_validation_end = len(unique_dates) - N_FOLDS * VALIDATION_DAYS
    folds = []

    for fold_index in range(N_FOLDS):
        validation_start_index = (
            first_validation_end + fold_index * VALIDATION_DAYS
        )
        validation_end_index = validation_start_index + VALIDATION_DAYS

        if validation_end_index > len(unique_dates):
            break

        validation_dates = unique_dates[
            validation_start_index:validation_end_index
        ]

        validation_start = pd.Timestamp(validation_dates[0])
        validation_end = pd.Timestamp(validation_dates[-1])

        train_mask = df[DATE_COLUMN] < validation_start
        valid_mask = df[DATE_COLUMN].isin(validation_dates)

        train_df = df.loc[train_mask]
        valid_df = df.loc[valid_mask]

        if train_df.empty or valid_df.empty:
            continue

        folds.append(
            {
                "fold": fold_index + 1,
                "train_end": train_df[DATE_COLUMN].max(),
                "valid_start": validation_start,
                "valid_end": validation_end,
                "train_df": train_df,
                "valid_df": valid_df,
            }
        )

    if not folds:
        raise ValueError("No valid chronological folds could be created.")

    return folds


def train_lightgbm(X_train, y_train, X_valid, y_valid, params):
    model_params = {
        **params,
        "objective": "regression",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }

    model = lgb.LGBMRegressor(**model_params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            lgb.early_stopping(75, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    predictions = model.predict(X_valid)
    return predictions



def train_random_forest(X_train, y_train, X_valid, max_rows=250_000):
    """Train Random Forest using numeric, consistently encoded features."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import OrdinalEncoder

    # Convert categorical columns to strings and encode them consistently.
    categorical_columns = [
        column
        for column in X_train.columns
        if isinstance(X_train[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(X_train[column])
        or pd.api.types.is_string_dtype(X_train[column])
    ]

    X_fit = X_train.copy()
    X_eval = X_valid.copy()

    if categorical_columns:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        X_fit[categorical_columns] = encoder.fit_transform(
            X_fit[categorical_columns].astype(str)
        )
        X_eval[categorical_columns] = encoder.transform(
            X_eval[categorical_columns].astype(str)
        )

    # Random Forest training is capped to control memory and runtime.
    y_array = np.asarray(y_train)
    if len(X_fit) > max_rows:
        rng = np.random.default_rng(42)
        sampled_indices = rng.choice(
            len(X_fit), size=max_rows, replace=False
        )
        X_fit = X_fit.iloc[sampled_indices]
        y_fit = y_array[sampled_indices]
    else:
        y_fit = y_array

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_fit, y_fit)
    return model.predict(X_eval)


def main():
    df = load_data()
    params = load_best_params()
    feature_columns = get_feature_columns(df)
    folds = make_folds(df)

    print(f"Rows: {len(df):,}")
    print(f"Features: {len(feature_columns)}")
    print(f"Chronological folds: {len(folds)}")
    print("Models: Tuned LightGBM and Random Forest")

    records = []

    for fold in folds:
        train_df = fold["train_df"]
        valid_df = fold["valid_df"]

        X_train = prepare_features(train_df, feature_columns)
        X_valid = prepare_features(valid_df, feature_columns)

        
        # Ensure validation data uses exactly the training categories.
        for column in X_train.columns:
            if isinstance(X_train[column].dtype, pd.CategoricalDtype):
                categories = X_train[column].cat.categories

                X_valid[column] = pd.Categorical(
                    X_valid[column].astype(str),
                    categories=categories,
                )


        y_train = train_df[TARGET_COLUMN].to_numpy()
        y_valid = valid_df[TARGET_COLUMN].to_numpy()

        print(
            f"\nFold {fold['fold']}: "
            f"train through {fold['train_end'].date()}, "
            f"validate {fold['valid_start'].date()} "
            f"to {fold['valid_end'].date()}"
        )

        model_predictions = {
            "Tuned LightGBM": train_lightgbm(
                X_train,
                y_train,
                X_valid,
                y_valid,
                params,
            ),
            "Random Forest": train_random_forest(
                X_train,
                y_train,
                X_valid,
            ),
        }

        for model_name, predictions in model_predictions.items():
            predictions = np.maximum(predictions, 0)
            metrics = evaluate_forecast(y_valid, predictions)

            record = {
                "Model": model_name,
                "Fold": fold["fold"],
                "TrainEnd": fold["train_end"],
                "ValidationStart": fold["valid_start"],
                "ValidationEnd": fold["valid_end"],
                "TrainRows": len(train_df),
                "ValidationRows": len(valid_df),
                **metrics,
            }

            records.append(record)

            print(
                f"  {model_name}: "
                f"MAE={metrics['MAE']:.3f}, "
                f"RMSE={metrics['RMSE']:.3f}, "
                f"WAPE={metrics['WAPE']:.3f}%"
            )

    results = pd.DataFrame(records)
    summary = (
        results.groupby("Model")[["MAE", "RMSE", "WAPE"]]
        .agg(["mean", "std", "median"])
        .round(4)
    )

    REPORTS_DIR = ROOT / "reports"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH)

    print("\nCross-validation summary:")
    print(summary.to_string())

    print(f"\nDetailed results: {OUTPUT_PATH}")
    print(f"Summary results:  {SUMMARY_PATH}")
    print(
        "\nNote: this compares the two models on the same folds. "
        "It does not establish recursive multi-day forecast accuracy."
    )


if __name__ == "__main__":
    main()
