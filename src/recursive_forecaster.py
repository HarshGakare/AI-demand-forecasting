from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation import evaluate_forecast




PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = (PROJECT_ROOT/ "data"/ "processed"/ "features_train.csv")
SALES_PATH = (PROJECT_ROOT / "data" / "processed" / "train_merged.csv" )
MODEL_PATH = ( PROJECT_ROOT / "models" / "lightgbm_demand_model.joblib")
OUTPUT_DIR = PROJECT_ROOT / "reports"

PREDICTIONS_PATH = (OUTPUT_DIR / "recursive_forecast_predictions.csv")
METRICS_PATH = ( OUTPUT_DIR / "recursive_forecast_metrics.csv")

HORIZONS = [7, 14, 30]


def load_inputs():
    print("Loading training features...")
    features_df = pd.read_csv(
        FEATURES_PATH,
        parse_dates=["Date"],
    )

    print("Loading historical sales...")
    sales_df = pd.read_csv(
        SALES_PATH,
        parse_dates=["Date"],
        low_memory=False,
    )

    print("Loading LightGBM model...")
    model = joblib.load(MODEL_PATH)

    return features_df, sales_df, model



def get_model_feature_names(model):
    """
    Get the exact feature order expected by LightGBM.
    """

    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)

    if hasattr(model, "booster_"):
        return list(model.booster_.feature_name())

    raise AttributeError(
        "Unable to determine LightGBM model feature names."
    )


def build_categorical_metadata(
    features_df,
    model,
    model_feature_names,
):
    """
    Reconstruct the categorical columns exactly as they
    were identified during LightGBM training.

    The training pipeline converted every object/category
    feature to pandas category dtype before fitting.
    The saved LightGBM Booster also stores the category
    values in pandas_categorical.
    """

    # These are the columns that were categorical during
    # training because lightgbm_tuning.py identifies
    # object/category columns this way.
    categorical_columns = (
        features_df[
            model_feature_names
        ]
        .select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    booster = model.booster_

    pandas_categorical = getattr(
        booster,
        "pandas_categorical",
        None,
    )

    if pandas_categorical is None:
        raise ValueError(
            "The saved LightGBM model does not contain "
            "pandas_categorical metadata."
        )

    if len(categorical_columns) != len(
        pandas_categorical
    ):
        raise ValueError(
            "Categorical feature mismatch.\n"
            f"Training pipeline found "
            f"{len(categorical_columns)} categorical columns: "
            f"{categorical_columns}\n"
            f"LightGBM model contains "
            f"{len(pandas_categorical)} categorical "
            "definitions."
        )

    metadata = {}

    for column, categories in zip(
        categorical_columns,
        pandas_categorical,
    ):
        metadata[column] = {
            "categories": pd.Index(
                categories
            ),
            "ordered": False,
        }

    return metadata




def build_store_static_features(features_df):
    """
    Keep the latest known static/store-level values for each store.
    Dynamic time-dependent features are rebuilt during the recursive forecast.
    """

    features_sorted = features_df.sort_values(["Store", "Date"] ).copy()

    latest_rows = (
        features_sorted
        .groupby("Store", as_index=False)
        .tail(1)
        .copy()
    )

    latest_rows = latest_rows.set_index("Store")

    return latest_rows


def add_calendar_features(row, forecast_date):
    """
    Add the same calendar features used by the training feature engineering pipeline.
    """

    row["Year"] = forecast_date.year
    row["Month"] = forecast_date.month
    row["Day"] = forecast_date.day
    row["DayOfWeek"] = forecast_date.dayofweek + 1
    row["WeekOfYear"] = (forecast_date.isocalendar().week)
    row["DayOfYear"] = forecast_date.dayofyear
    row["Quarter"] = forecast_date.quarter
    row["IsWeekend"] = int(forecast_date.dayofweek >= 5 )
    row["IsMonthStart"] = int(forecast_date.is_month_start )
    row["IsMonthEnd"] = int( forecast_date.is_month_end)
    row["MonthSin"] = np.sin( 2 * np.pi * forecast_date.month / 12)
    row["MonthCos"] = np.cos(2 * np.pi * forecast_date.month / 12 )
    row["WeekdaySin"] = np.sin( 2 * np.pi * (forecast_date.dayofweek + 1) / 7 )
    row["WeekdayCos"] = np.cos( 2 * np.pi * (forecast_date.dayofweek + 1) / 7)

    return row



def calculate_promo_interval_active( promo_interval, forecast_date):
    """
    Reproduce PromoIntervalActive from the feature engineering pipeline.
    """

    if pd.isna(promo_interval):
        return 0

    interval = str(promo_interval).strip()

    if not interval:
        return 0

    month_name = forecast_date.strftime("%b")

    active_months = [
        value.strip()
        for value in interval.split(",")
    ]

    return int( month_name in active_months)



def build_recursive_row(store_id, forecast_date, history, static_features):
    """
    Build one future feature row using ONLY information available before forecast_date.
    Previously predicted sales are added to history after every prediction, making this genuinely recursive.
    """

    if store_id not in static_features.index:
        raise ValueError(
            f"Store {store_id} not found in static features."
        )

    static_row = (
        static_features
        .loc[store_id]
        .copy()
    )

    row = static_row.to_dict()

    row["Store"] = store_id
    row["Date"] = forecast_date


    row = add_calendar_features( row, forecast_date )


    if "PromoInterval" in row:
        row["PromoIntervalActive"] = ( calculate_promo_interval_active(row["PromoInterval"], forecast_date ))


    store_history = history[ history["Store"] == store_id ].sort_values("Date")
    sales_by_date = ( store_history.set_index("Date")["Sales"])



    lag_windows = [ 7, 14, 28,35 ]

    for lag in lag_windows:

        lag_date = (forecast_date - pd.Timedelta(days=lag))

        if lag_date in sales_by_date.index:
            value = sales_by_date.loc[lag_date]

            if isinstance(value, pd.Series):
                value = value.iloc[-1]

            row[f"SalesLag_{lag}"] = float( value )
        else:
            row[f"SalesLag_{lag}"] = np.nan



    previous_sales = store_history[
        store_history["Date"] < forecast_date
    ].sort_values("Date")["Sales"]

    for window in [7, 14, 28]:

        values = previous_sales.tail(window)

        row[f"SalesRollingMean_{window}"] = (
            values.mean()
            if len(values) > 0
            else np.nan
        )

        row[f"SalesRollingStd_{window}"] = (
            values.std()
            if len(values) > 1
            else 0.0
        )

    return pd.DataFrame([row])



def prepare_model_features( row_df, model_feature_names, training_features_df, categorical_metadata ):
    """
    Prepare the future row so that:

    1. Columns exactly match the trained model.
    2. Column order exactly matches the trained model.
    3. Categorical columns use the SAME categories as training.
    4. Object columns are handled consistently.
    """

    X = row_df.copy()


    missing_features = [
        feature
        for feature in model_feature_names
        if feature not in X.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features:\n"
            +
            "\n".join(str(feature) for feature in missing_features)
        )

    X = X[ model_feature_names].copy()
    X = X.replace([np.inf, -np.inf],np.nan)

    # Recreate categorical metadata exactly

    for column, metadata in (categorical_metadata.items() ):

        if column not in X.columns:
            continue

        values = (
            X[column]
            .astype("object")
            .where(
                X[column].notna(),
                "__MISSING__",
            )
            .astype(str)
        )

        X[column] = pd.Categorical(
            values,
            categories=metadata["categories"],
            ordered=metadata["ordered"],
        )


    for column in model_feature_names:

        if column in categorical_metadata:
            continue

        if column not in X.columns:
            continue

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )
        
        if list(X.columns) != model_feature_names:
            raise ValueError(
                "Feature columns do not match the "
                "trained LightGBM model."
            )

    return X



def recursive_forecast_store(
    store_id,
    forecast_start,
    history_df,
    model,
    static_features,
    training_features_df,
    categorical_metadata,
    model_feature_names,
    horizon,
):
    """
    Recursively forecast one store.

    Prediction from day t becomes historical sales for constructing features on day t+1.
    """

    store_history = history_df[
        history_df["Store"] == store_id
    ].copy()

    store_history = store_history.sort_values(
        "Date"
    )

    predictions = []

    for step in range(horizon):

        forecast_date = (
            forecast_start
            + pd.Timedelta(days=step)
        )

        row_df = build_recursive_row(
            store_id=store_id,
            forecast_date=forecast_date,
            history=store_history,
            static_features=static_features,
        )

        X = prepare_model_features(
            row_df=row_df,
            model_feature_names=model_feature_names,
            training_features_df=training_features_df,
            categorical_metadata=categorical_metadata,
        )

        prediction = model.predict(X)

        prediction = float(
            np.asarray(prediction).ravel()[0]
        )

        # Sales cannot be negative.
        prediction = max(
            0.0,
            prediction,
        )

        predictions.append(
            {
                "Store": store_id,
                "Date": forecast_date,
                "Prediction": prediction,
            }
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Feed prediction back into history.
        # ----------------------------------------------------

        new_history_row = pd.DataFrame(
            [
                {
                    "Store": store_id,
                    "Date": forecast_date,
                    "Sales": prediction,
                }
            ]
        )

        store_history = pd.concat(
            [
                store_history,
                new_history_row,
            ],
            ignore_index=True,
        )

    return pd.DataFrame(predictions)



def recursive_forecast_all_stores(
    history_df,
    model,
    static_features,
    training_features_df,
    categorical_metadata,
    model_feature_names,
    forecast_start,
    horizon,
):
    """
    Generate recursive forecasts for every store.
    """

    stores = sorted(
        history_df["Store"]
        .dropna()
        .unique()
    )

    print(
        f"Forecasting {len(stores):,} stores "
        f"for {horizon} days..."
    )

    all_predictions = []

    for index, store_id in enumerate(
        stores,
        start=1,
    ):

        if index == 1 or index % 100 == 0:
            print(
                f"Processing store "
                f"{index:,}/{len(stores):,}..."
            )

        store_predictions = (
            recursive_forecast_store(
                store_id=store_id,
                forecast_start=forecast_start,
                history_df=history_df,
                model=model,
                static_features=static_features,
                training_features_df=training_features_df,
                categorical_metadata=categorical_metadata,
                model_feature_names=model_feature_names,
                horizon=horizon,
            )
        )

        all_predictions.append(
            store_predictions
        )

    if not all_predictions:
        return pd.DataFrame(
            columns=[
                "Store",
                "Date",
                "Prediction",
            ]
        )

    return pd.concat(
        all_predictions,
        ignore_index=True,
    )



def evaluate_recursive_forecast(
    predictions,
    actual_sales,
    forecast_origin,
):
    """
    Evaluate recursive forecasts against actual sales.
    """

    merged = predictions.merge(
        actual_sales[
            [
                "Store",
                "Date",
                "Sales",
            ]
        ],
        on=[
            "Store",
            "Date",
        ],
        how="left",
    )

    merged = merged.dropna(
        subset=["Sales"]
    )

    metrics = evaluate_forecast(
        merged["Sales"],
        merged["Prediction"],
    )

    metrics["Horizon"] = int(
        (
            merged["Date"].max()
            - forecast_origin
        ).days
    )

    metrics["ForecastOrigin"] = (
        forecast_origin
    )

    metrics["Rows"] = len(merged)

    return metrics, merged




def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    training_features_df, sales_df, model = (
        load_inputs()
    )

    print()
    print(
        f"Training feature rows: "
        f"{len(training_features_df):,}"
    )

    print(
        f"Historical sales rows: "
        f"{len(sales_df):,}"
    )

    model_feature_names = (
        get_model_feature_names(model)
    )

    print(
        f"Model features: "
        f"{len(model_feature_names)}"
    )


    categorical_metadata = (
        build_categorical_metadata(
            training_features_df,
            model,
            model_feature_names
        )
    )

    print(
        "Categorical features:",
        list(
            categorical_metadata.keys()
        ),
    )

    static_features = (
        build_store_static_features(
            training_features_df
        )
    )

    # Forecast origin
    #
    # We use the final 30 days of the historical dataset
    # as the evaluation period.

    max_date = sales_df["Date"].max()

    forecast_origin = (
        max_date
        - pd.Timedelta(days=30)
    )

    forecast_start = (
        forecast_origin
        + pd.Timedelta(days=1)
    )

    print()
    print(
        f"Forecast origin: "
        f"{forecast_origin.date()}"
    )

    print(
        f"Forecast start: "
        f"{forecast_start.date()}"
    )

    print(
        f"Historical maximum date: "
        f"{max_date.date()}"
    )


    # Only use data available before forecast origin when
    # generating the recursive forecast.


    history_df = sales_df[
        sales_df["Date"] <= forecast_origin
    ].copy()

    actual_sales = sales_df[
        sales_df["Date"] > forecast_origin
    ].copy()

    print(
        f"Training history rows used: "
        f"{len(history_df):,}"
    )

    print(
        f"Actual evaluation rows: "
        f"{len(actual_sales):,}"
    )


    # Forecast 30 days once.
    #
    # Because this is recursive, the 7-day and 14-day
    # results can be obtained from the same 30-day run.


    print()
    print(
        "Starting recursive 30-day forecast..."
    )

    predictions = (
        recursive_forecast_all_stores(
            history_df=history_df,
            model=model,
            static_features=static_features,
            training_features_df=training_features_df,
            categorical_metadata=categorical_metadata,
            model_feature_names=model_feature_names,
            forecast_start=forecast_start,
            horizon=30,
        )
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print()
    print(
        f"Saved predictions to:"
        f"\n{PREDICTIONS_PATH}"
    )


    # Evaluate each horizon


    all_metrics = []
    all_evaluated_predictions = []

    for horizon in HORIZONS:

        horizon_end = (
            forecast_origin
            + pd.Timedelta(days=horizon)
        )

        horizon_predictions = predictions[
            predictions["Date"] <= horizon_end
        ].copy()

        metrics, evaluated = (
            evaluate_recursive_forecast(
                predictions=horizon_predictions,
                actual_sales=actual_sales,
                forecast_origin=forecast_origin,
            )
        )

        metrics["Horizon"] = horizon

        all_metrics.append(metrics)

        evaluated["Horizon"] = horizon

        all_evaluated_predictions.append(
            evaluated
        )

        print()
        print(
            f"{horizon}-day recursive forecast:"
        )

        print(
            f"MAE: "
            f"{metrics['MAE']:.4f}"
        )

        print(
            f"RMSE: "
            f"{metrics['RMSE']:.4f}"
        )

        print(
            f"WAPE: "
            f"{metrics['WAPE']:.4f}%"
        )


    # Save metrics


    metrics_df = pd.DataFrame(
        all_metrics
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    print()
    print(
        f"Saved metrics to:"
        f"\n{METRICS_PATH}"
    )


    # Replace prediction file with evaluated version


    if all_evaluated_predictions:

        evaluated_predictions = pd.concat(
            all_evaluated_predictions,
            ignore_index=True,
        )

        evaluated_predictions = (
            evaluated_predictions[
                [
                    "Store",
                    "Date",
                    "Sales",
                    "Prediction",
                    "Horizon",
                ]
            ]
        )

        evaluated_predictions.to_csv(
            PREDICTIONS_PATH,
            index=False,
        )



    print()
    print("=" * 60)
    print("RECURSIVE FORECAST SUMMARY")
    print("=" * 60)

    for _, row in metrics_df.iterrows():

        print(
            f"Horizon {int(row['Horizon']):2d} days | "
            f"MAE: {row['MAE']:.4f} | "
            f"RMSE: {row['RMSE']:.4f} | "
            f"WAPE: {row['WAPE']:.4f}%"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()