from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, early_stopping
from statsmodels.tsa.holtwinters import ExponentialSmoothing

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

FEATURES_TRAIN_PATH = PROCESSED_DIR / "features_train.csv"
BEST_PARAMS_PATH = REPORTS_DIR / "lightgbm_best_params.csv"
LIGHTGBM_MODEL_PATH = MODELS_DIR / "lightgbm_demand_model.joblib"
LIGHTGBM_PREDICTIONS_PATH = REPORTS_DIR / "multi_horizon_predictions.csv"

ENSEMBLE_TUNING_PATH = REPORTS_DIR / "ensemble_tuning.csv"
ENSEMBLE_METRICS_PATH = REPORTS_DIR / "ensemble_final_metrics.csv"
ENSEMBLE_PREDICTIONS_PATH = REPORTS_DIR / "ensemble_predictions.csv"
ENSEMBLE_WEIGHT_PATH = REPORTS_DIR / "ensemble_best_weight.csv"


FINAL_HORIZON = 30
TUNING_HORIZON = 30
SEASONAL_PERIOD = 7

ENSEMBLE_WEIGHTS = np.round(np.arange(0.0, 1.01, 0.05), 2)




def calculate_mae(y_true, y_pred): 
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.mean(np.abs(y_true - y_pred))


def calculate_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_wape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return np.nan

    return (np.sum(np.abs(y_true - y_pred))/ denominator* 100)


def evaluate_predictions(y_true, y_pred):
    return{ "MAE": calculate_mae(y_true,y_pred),
            "RMSE": calculate_rmse(y_true,y_pred),
             "WAPE": calculate_wape(y_true,y_pred)}
    

def prepare_features(df):
    """
    Convert categorical/object columns into numerical category codes.This must match the approach used when training LightGBM.
    """
    df = df.copy()

    object_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in object_columns:
        df[column] = df[column].astype("category").cat.codes

    return df


def seasonal_naive_forecast(history, horizon):
    """Fallback forecast using the last complete weekly pattern"""
    
    history = np.asarray(history, dtype=float)
    
    if len(history) == 0:
        return np.zeros(horizon)
    
    if len(history) < SEASONAL_PERIOD:
        return np.repeat(np.mean(history), horizon)
    
    last_week = history[-SEASONAL_PERIOD:]
    
    repetitions = int(np.ceil(horizon / SEASONAL_PERIOD))
    
    forecast = np.tile(last_week, repetitions)
    
    return forecast[:horizon]

def ets_forecast(history, horizon):
    """
    Forecast a single store using additive Holt-Winters exponential smoothing with weekly seasonality.
    Falls back to seasonal naice if the ETS model fails
    """
    
    history = np.asarray(history,dtype=float)
    
    history = np.nan_to_num(history, nan=0.0, posinf=0.0, neginf=0.0)
    
    if len(history) < 2 * SEASONAL_PERIOD:
        return seasonal_naive_forecast(history, horizon)
    
    
    try : 
        model = ExponentialSmoothing(history,trend=None, seasonal="add", seasonal_periods=SEASONAL_PERIOD, initialization_method="estimated")
        
        fitted_model = model.fit(optimized=True,use_brute=True)
        forecast = fitted_model.forecast(horizon)
        forecast = np.asarray(forecast, dtype=float)
        forecast = np.nan_to_num(forecast, nan=0.0,posinf=0.0, neginf=0.0)
        
        return np.maximum(forecast, 0)
    
    except Exception:
        return seasonal_naive_forecast(history,horizon)
    
    

def build_ets_forecasts(sales_df, origin_date, horizon):
    """
    Generate ETS forecasts for every store.
    Historical data ends strictly before origin_date."""
    
    history_df = sales_df[sales_df["Date"] <= origin_date].copy()
    stores = sorted(history_df["Store"].unique())
    
    forecast_rows = []
    
    future_dates = pd.date_range(start=origin_date + pd.Timedelta(days=1),
                                 periods=horizon, freq='D')
    
    print(f"Generating ETS forecasts for {len(stores):,} stores")
    
    for index, store_id in enumerate(stores, start=1):
        
        store_history = (history_df[history_df["Store"] == store_id].sort_values("Date"))
        
        sales = store_history["Sales"].values
        
        forecast = ets_forecast(sales,horizon)
        
        for date, prediction in zip(future_dates, forecast):
            forecast_rows.append({ "Store": store_id,
                                    "Date": date,
                                    "ETSPrediction": float(max(prediction, 0) ),})
            
        if index % 100 == 0:
            print(f"Processed {index:,} / {len(stores):,} stores")
    
    return pd.DataFrame(forecast_rows)



def load_best_lightgbm_params():
    """
    Load the best hyperparameters discovered by Optuna"""
    
    params_df = pd.read_csv(BEST_PARAMS_PATH)
    
    if params_df.empty:
        raise ValueError("lightbgm_best_params.csv is empty")
    
    params = params_df.iloc[0].to_dict()
    
    integer_parameters = { "n_estimators","num_leaves","max_depth","min_child_samples"}
    
    float_parameters = {"learning_rate","subsample","colsample_bytree","reg_alpha","reg_lambda","min_split_gain"}
    
    cleaned_params = {}
    
    for key, value in params.items():
        
        if key in integer_parameters:
            cleaned_params[key] = int(value)
        
        elif key in float_parameters:
            cleaned_params[key] = float(value)
    
    return cleaned_params


def train_validation_lightgbm(train_df,validation_df,feature_columns, traget_column="Sales"):
    """
    Train LightGBM only on the data preceding the ensemble tuning validation period.
    This prevents final-holdout leakage when selecting the ensemble weight """
    
    X_train = prepare_features(train_df[feature_columns])
    X_valid = prepare_features(validation_df[feature_columns])
    X_valid = X_valid.reindex(columns=X_train.columns,fill_value=0)
    y_train = train_df[traget_column]
    y_valid = validation_df[traget_column]
    
    params = load_best_lightgbm_params()
    
    model = LGBMRegressor(objective="regression",random_state=42,n_jobs=-1,verbosity=-1,**params)
    
    model.fit(X_train,y_train,eval_set=[(X_valid, y_valid)], callbacks=[early_stopping(75, verbose=False)])
    
    predictions = model.predict(X_valid)
    predictions = np.maximum(predictions,0)
    
    return model, predictions

def find_best_weight(y_true,lightgbm_predictions, ets_predictions):
    """
    Search for the LightGBM weight producing the
    lowest MAE on the dedicated tuning validation period.
    """
    
    results = []
    
    for alpha in ENSEMBLE_WEIGHTS:
        
        ensemble_predictions = (alpha * lightgbm_predictions + (1-alpha) * ets_predictions)
        
        metrics = evaluate_predictions(y_true, ensemble_predictions)
        
        results.append({
                "LightGBMWeight": alpha,
                "ETSWeight": 1 - alpha,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "WAPE": metrics["WAPE"],
            })
        
    results_df = pd.DataFrame(results).sort_values("MAE")
    best_row = results_df.iloc[0]
    
    return(float(best_row["LightGBMWeight"]), results_df)


def main():
    
    print("_"*60)
    print(" LIGHTGBM + ETSENSEMBLE")
    print("_"*60)
    
    print("\n Loading features dataset")
    
    df = pd.read_csv(FEATURES_TRAIN_PATH, parse_dates=["Date"])
    df = df.sort_values(["Date","Store"]).reset_index(drop=True)
    
    print(f"Rows: {len(df):,}")
    print(f"Data range {df["Date"].min().date() } to {df["Date"].max().date()}")
    
    merged_path = (PROCESSED_DIR / "train_merged.csv")
    
    sales_df = pd.read_csv(merged_path,parse_dates=["Date"])
    sales_df = sales_df[
        [
            "Store",
            "Date",
            "Sales",
        ]].copy()
    sales_df = sales_df.sort_values(["Date", "Store"] )


    #  Define final holdout

    max_date = df["Date"].max()

    final_holdout_start = (max_date- pd.Timedelta(days=FINAL_HORIZON - 1 ))

    final_origin = (final_holdout_start - pd.Timedelta(days=1) )

    final_holdout = df[ df["Date"] >= final_holdout_start].copy()

    print("\nFinal holdout:")
    print( f"Origin: {final_origin.date()}" )
    print( f"Start: {final_holdout_start.date()}" )
    print( f"End: {max_date.date()}" )
    print(f"Rows: {len(final_holdout):,}")


    #  Define earlier validation window

    tuning_valid_end = (final_holdout_start- pd.Timedelta(days=1))
    tuning_valid_start = (tuning_valid_end - pd.Timedelta( days=TUNING_HORIZON - 1) )
    
    tuning_train_end = (tuning_valid_start- pd.Timedelta(days=1))
    tuning_train = df[ df["Date"] <= tuning_train_end ].copy()
    tuning_valid = df[ (df["Date"] >= tuning_valid_start)
                            &
                        ( df["Date"] <= tuning_valid_end ) ].copy()

    print("\nEnsemble tuning validation:")
    print(f"  Train through: {tuning_train_end.date()}" )
    print(f"  Validation: {tuning_valid_start.date()} to {tuning_valid_end.date()}")


    #  Determine feature columns

    excluded_columns = {"Sales", "Date",}
    feature_columns = [ column for column in df.columns if column not in excluded_columns]


    #  Train validation LightGBM

    print("\nTraining validation LightGBM...")

    _, lgb_validation_predictions = (
        train_validation_lightgbm(
            tuning_train,
            tuning_valid,
            feature_columns,
        )
    )


    #  Generate validation ETS forecasts


    print("\nGenerating validation ETS forecasts..." )

    tuning_ets = build_ets_forecasts(
        sales_df,
        tuning_train_end,
        TUNING_HORIZON,
    )

    tuning_actual = tuning_valid[
        [
            "Store",
            "Date",
            "Sales",
        ]
    ].copy()

    tuning_actual = tuning_actual.sort_values(["Store", "Date"]).reset_index(drop=True)

    tuning_ets = tuning_ets.sort_values( ["Store", "Date"] ).reset_index(drop=True)

    # Ensure identical row ordering.
    
    if not ( tuning_actual["Store"].values == tuning_ets["Store"].values ).all():
        raise ValueError(
            "Store ordering mismatch between "
            "validation actuals and ETS forecasts."
        )

    if not (tuning_actual["Date"].values == tuning_ets["Date"].values).all():
        raise ValueError(
            "Date ordering mismatch between "
            "validation actuals and ETS forecasts."
        )

    y_tuning = tuning_actual[ "Sales"].values

    ets_validation_predictions = (tuning_ets[ "ETSPrediction"].values )


    #  Evaluate individual models


    lgb_metrics = evaluate_predictions( y_tuning, lgb_validation_predictions )
    ets_metrics = evaluate_predictions(  y_tuning,  ets_validation_predictions)

    print("\nValidation model performance")
    print("-" * 50)

    print( f"LightGBM MAE: {lgb_metrics['MAE']:.4f}")
    print(f"ETS MAE: {ets_metrics['MAE']:.4f}")
    
    print(f"LightGBM WAPE: {lgb_metrics['WAPE']:.4f}%")
    print(  f"ETS WAPE: {ets_metrics['WAPE']:.4f}%")

    #  Tune ensemble weight

    print( "\nSearching ensemble weights...")

    best_weight, tuning_results = (
        find_best_weight(
            y_tuning,
            lgb_validation_predictions,
            ets_validation_predictions,
        )
    )

    best_ets_weight = ( 1 - best_weight )

    tuning_results.to_csv(ENSEMBLE_TUNING_PATH, index=False, )

    pd.DataFrame(
        [
            {
                "LightGBMWeight": best_weight,
                "ETSWeight": best_ets_weight,
            }
        ]
    ).to_csv( ENSEMBLE_WEIGHT_PATH, index=False, )

    print( "\nBest ensemble weights")
    print("-" * 50)
    print(f"LightGBM: {best_weight:.2f}" )
    print(f"ETS:  {best_ets_weight:.2f}")


    #  Load final LightGBM predictions

    print("\nLoading final 30-day LightGBM predictions..." )

    lgb_predictions = pd.read_csv( LIGHTGBM_PREDICTIONS_PATH, parse_dates=["Date"])

    lgb_predictions = lgb_predictions[ lgb_predictions["Horizon"] == 30 ].copy()

    lgb_predictions = lgb_predictions[ [
            "Store",
            "Date",
            "Sales",
            "Prediction",
        ]
    ].copy()
    
    lgb_predictions = lgb_predictions.rename(
    columns={
        "Sales": "ActualSales",
        "Prediction": "PredictedSales",
    }
)

    lgb_predictions = lgb_predictions.sort_values( ["Store", "Date"] ).reset_index(drop=True)


    #  Generate final ETS forecasts


    print("\nGenerating final ETS forecasts...")

    final_ets = build_ets_forecasts(
        sales_df,
        final_origin,
        FINAL_HORIZON,
    )

    final_ets = final_ets.sort_values(["Store", "Date"]).reset_index(drop=True)

    #  Merge LightGBM + ETS


    ensemble = lgb_predictions.merge(
        final_ets,
        on=["Store", "Date"],
        how="inner",
    )

    if ensemble.empty:
        raise ValueError(
            "No rows matched between LightGBM "
            "and ETS forecasts."
        )

    ensemble[ "EnsemblePrediction"] = (
        best_weight
        * ensemble["PredictedSales"]
        + best_ets_weight
        * ensemble["ETSPrediction"]
    )

    ensemble[ "EnsemblePrediction"] = ensemble[ "EnsemblePrediction"].clip(lower=0)


    #  Final 30-day evaluation


    y_final = ensemble["ActualSales"].values

    lgb_final = ensemble[ "PredictedSales" ].values
    ets_final = ensemble["ETSPrediction"].values

    ensemble_final = ensemble["EnsemblePrediction" ].values

    lgb_final_metrics = evaluate_predictions( y_final, lgb_final)
    ets_final_metrics = evaluate_predictions( y_final, ets_final)

    ensemble_final_metrics = (evaluate_predictions(  y_final,  ensemble_final ))


    #  Horizon-specific metrics

    ensemble["DaysAhead"] = (ensemble.groupby("Store").cumcount() + 1  )

    metric_rows = []

    for horizon in [7, 14, 30]:
        
        horizon_df = ensemble[ ensemble["DaysAhead"] <= horizon ]

        actual = horizon_df[ "ActualSales"].values

        lgb_pred = horizon_df[ "PredictedSales"].values

        ets_pred = horizon_df[ "ETSPrediction"].values

        ensemble_pred = horizon_df["EnsemblePrediction"].values

        lgb_metrics = evaluate_predictions(actual,lgb_pred, )

        ets_metrics = evaluate_predictions( actual, ets_pred,)

        ensemble_metrics = evaluate_predictions(actual, ensemble_pred,)

        metric_rows.extend(
            [ {
                    "Model": "LightGBM",
                    "Horizon": horizon,
                    **lgb_metrics,
                },
                {
                    "Model": "ETS",
                    "Horizon": horizon,
                    **ets_metrics,
                },
                {
                    "Model": "Ensemble",
                    "Horizon": horizon,
                    **ensemble_metrics,
                },
            ]
        )

    metrics_df = pd.DataFrame(metric_rows)

    metrics_df.to_csv( ENSEMBLE_METRICS_PATH, index=False,)


    # Save predictions

    ensemble.to_csv(ENSEMBLE_PREDICTIONS_PATH,index=False )

    #  Final output


    print("\n")
    print("=" * 70)
    print("FINAL ENSEMBLE RESULTS")
    print("=" * 70)

    print(f"\nSelected weights:" )

    print( f"LightGBM: {best_weight:.2f}" )

    print(f" ETS: {best_ets_weight:.2f}")

    print("\n30-day final holdout:")

    print(f"  LightGBM MAE: {lgb_final_metrics['MAE']:.4f}")

    print(f" ETS MAE: {ets_final_metrics['MAE']:.4f}")

    print(f"Ensemble MAE: {ensemble_final_metrics['MAE']:.4f}" )

    print(f"\n  LightGBM WAPE: {lgb_final_metrics['WAPE']:.4f}%")

    print(f"  ETS WAPE:  {ets_final_metrics['WAPE']:.4f}%")

    print(f"  Ensemble WAPE: {ensemble_final_metrics['WAPE']:.4f}%" )

    print("\nHorizon comparison:")
    print( metrics_df.to_string( index=False)  )

    print("\nSaved files:")
    print(f"  {ENSEMBLE_TUNING_PATH}")
    print(f"  {ENSEMBLE_WEIGHT_PATH}" )
    print( f"  {ENSEMBLE_METRICS_PATH}"  )
    print(f"  {ENSEMBLE_PREDICTIONS_PATH}" )
    print("\nStep 7 completed successfully.")


if __name__ == "__main__":
    main()

        
    
    
                
    
        
    