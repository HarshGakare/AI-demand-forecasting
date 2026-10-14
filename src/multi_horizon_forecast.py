from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.evaluation import evaluate_forecast

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (PROJECT_ROOT / "data" / "processed" / "features_train.csv")
MODEL_PATH = (PROJECT_ROOT / "models" / "lightgbm_demand_model.joblib")
REPORT_DIR = PROJECT_ROOT / "reports"

HORIZONS = [7,14,30]
RANDOM_STATE = 42


def load_data():
    """Load features data"""
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature data: {DATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained model: {MODEL_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["Date"],
    )

    df = df.sort_values(
        ["Store", "Date"]
    ).reset_index(drop=True)

    model = joblib.load(
        MODEL_PATH
    )

    return df, model


def prepare_features(df):
    """
    Prepare features in exactly the same general form
    used by the LightGBM training pipeline.
    """
    
    excluded_columns = {"Sales","Customers", "Date", "IsTrain",}

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    X = df[feature_columns].copy()

    categorical_columns = (
        X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
    )

    for column in categorical_columns:
        X[column] = X[column].astype("category")

    X = X.replace([np.inf, -np.inf],np.nan,)

    return X


def align_with_model(X,model):
    """
    Align prediction columns with the columns used by the saved LightGBM model.
    """
    
    if not hasattr(model, "feature_name_"):
        return X
    
    expected_columns = (model.feature_name_)
    
    for column in expected_columns:
        if column not in X.columns:
            X[column] = np.nan
            
    
    extra_columns = [column for column in X.columns if column not in expected_columns]
    
    if expected_columns:
        X = X.drop(columns=extra_columns)
    
    X= X[expected_columns]
    
    return X



def choose_forecast_origin(df, horizon):
    """
    Select a forecast origin sufficiently far from the end of the dataset to provide a complete future horizon.
    """
    
    max_date = df["Date"].max()
    
    origin = (max_date - pd.Timedelta(days=horizon))
    
    return origin


def create_fixed_horizon_dataset(df,origin,horizon):
    """
    Create one fixed forecast horizon.
    Only observations at before the forecast origin are considered historical information.
    """
    
    forecast_start = (origin + pd.Timedelta(days=1))
    forecast_end = (origin + pd.Timedelta(days=horizon))
    
    historical = df[df["Date"] <= origin].copy()
    
    future = df[(df["Date"] >=forecast_start) & (df["Date"] <= forecast_end)].copy()
    
    return historical, future


def forecast_horizon(df,model,origin, horizon):
    """
    Evaluate a fixed-origin forecast.

    Important:
    The model predictions are generated using features already stored for each future date. We do not update the feature values using actual future sales.
    """
    historical, future = (
        create_fixed_horizon_dataset(df, origin, horizon)
    )
    
    if future.empty:
        raise ValueError(f"No future observation available for horizon {horizon}")
        
    if( future["Date"].max() < origin + pd.Timedelta(days=horizon)):
        raise ValueError("Future data does not contain a complete forecast horizon")
    
    X_future = prepare_features(future)
    X_future = align_with_model(X_future, model)
    
    predictions = model.predict(X_future)
    predictions = np.clip(predictions,0,None)
    
    results = future[["Store","Date","Sales"]].copy()
    results["Prediction"] = predictions
    results["Horizon"] = horizon
    results["ForecastOrigin"] = origin
    
    return results


def evaluate_horizon(results):
    """Calculate metrics for one forecast horizon"""
    
    metrics = evaluate_forecast(results["Sales"],results["Prediction"])
    metrics["Horizon"] = int(results["Horizon"].iloc[0])
    
    return metrics



def evaluate_store_level(results):
    """
    Calculate store-level WAPE.
    This is useful because inventory decisions are made at individual store level rather than only globally.
    """
    
    store_results = []
    
    for store_id , group in results.groupby("Store"):
        metrics = evaluate_forecast(group["Sales"], group["Prediction"])
        
        store_results.append({
            "Store": store_id,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "WAPE": metrics["WAPE"]
        })
    
    return pd.DataFrame(store_results)

def save_results(horizon_results, horizon_metrics):
    """Save forecast results"""
    
    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    all_results = pd.concat(horizon_results, ignore_index=True)
    metrics_df = pd.DataFrame(horizon_metrics)
    all_results.to_csv(REPORT_DIR / "multi_horizon_predictions.csv", index=False)
    metrics_df.to_csv(REPORT_DIR / "multi_horizon_metrics.csv", index=False)
    
    return (all_results,metrics_df)


def main():

    df, model = load_data()

    print( "\nMulti-horizon forecasting")

    print(   "=" * 60 )

    horizon_results = []
    horizon_metrics = []

    for horizon in HORIZONS:

        origin = choose_forecast_origin( df, horizon,  )

        print( f"\nHorizon: {horizon} days")

        print( f"Forecast origin: " f"{origin.date()}")

        print(f"Forecast end: "f"{(origin + pd.Timedelta(days=horizon)).date()}")

        results = forecast_horizon(
            df,
            model,
            origin,
            horizon,
        )

        metrics = evaluate_horizon( results)
        horizon_results.append( results)
        horizon_metrics.append(  metrics )

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

    all_results, metrics_df = (
        save_results(
            horizon_results,
            horizon_metrics,
        )
    )

    print("\n" + "=" * 60 )
    print("MULTI-HORIZON RESULTS" )
    print( "=" * 60)

    print(metrics_df.to_string(
            index=False ) )

    print( "\nSaved:"  )

    print( REPORT_DIR / "multi_horizon_predictions.csv")

    print( REPORT_DIR / "multi_horizon_metrics.csv")


if __name__ == "__main__":
    main()