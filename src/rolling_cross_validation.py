from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

FEATURES_TRAIN_PATH = (PROCESSED_DIR / "features_train.csv")
BEST_PARAMS_PATH = (REPORTS_DIR / "lightgbm_best_params.csv")

ROLLING_RESULTS_PATH = (REPORTS_DIR / "rolling_cv_results.csv")
ROLLING_SUMMARY_PATH = (REPORTS_DIR / "rolling_cv_summary.csv")

FORECAST_HORIZON = 30
N_FOLDS = 4
STEP_SIZE = 30
RANDOM_STATE = 42

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



def load_best_parameters():

    if not BEST_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Missing parameter file: {BEST_PARAMS_PATH}"
        )

    params_df = pd.read_csv(BEST_PARAMS_PATH )

    if params_df.empty:
        raise ValueError(
            "lightgbm_best_params.csv is empty."
        )

    row = params_df.iloc[0]

    integer_parameters = {
        "n_estimators",
        "num_leaves",
        "max_depth",
        "min_child_samples",
    }

    float_parameters = {
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "min_split_gain",
    }

    parameters = {}

    for column in params_df.columns:

        if column in integer_parameters:
            parameters[column] = int(row[column] )

        elif column in float_parameters:
            parameters[column] = float( row[column] )

    return parameters

def prepare_features(df, feature_columns):
    """
    Prepare LightGBM input features.
    Object and category columns are converted to numerical category codes.
    This is intentionally kept compatible with the existing project pipeline.
    """

    X = df[feature_columns].copy()

    categorical_columns = (X.select_dtypes(include=["object", "category"] ).columns.tolist())

    for column in categorical_columns:
        X[column] = (X[column].astype("category").cat.codes )

    X = X.replace( [np.inf, -np.inf], np.nan,)

    return X


def create_rolling_folds(dates, horizon, n_folds, step_size,):
    """
    Create chronological rolling validation folds.
    Each fold predicts the next `horizon` days.
    Example:

        Fold 1:
        train -> validation

        Fold 2:
        expanded train -> later validation

        Fold 3:
        expanded train -> later validation

    No future observations are used for training.
    """

    unique_dates = np.sort(pd.Series( dates).dropna().unique() )

    max_date = pd.Timestamp(unique_dates[-1] )

    folds = []

    # The final validation period is the last `horizon` days.
    final_validation_start = ( max_date - pd.Timedelta( days=horizon - 1 ) )

    for fold_number in range( n_folds):

        validation_start = ( final_validation_start - pd.Timedelta( days=(n_folds - 1 - fold_number)* step_size ) )

        validation_end = ( validation_start + pd.Timedelta(  days=horizon - 1 ) )

        training_end = (validation_start- pd.Timedelta(days=1))

        # Safety check.
        if training_end <= pd.Timestamp(unique_dates[0]):
            continue

        folds.append( {
                "Fold": fold_number + 1,
                "TrainingEnd": training_end,
                "ValidationStart": validation_start,
                "ValidationEnd": validation_end,
            }
        )

    return folds

def train_fold_model( train_df, validation_df, feature_columns, parameters,):
    """
    Train one LightGBM model for one rolling fold.
    """

    X_train = prepare_features( train_df, feature_columns,)
    X_valid = prepare_features( validation_df, feature_columns,)
    X_valid = X_valid.reindex( columns=X_train.columns, fill_value=np.nan, )
    
    y_train = train_df[ "Sales" ].values
    y_valid = validation_df[ "Sales" ].values

    model = LGBMRegressor(
        objective="regression",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        **parameters,
    )

    model.fit(
        X_train,
        y_train,
        eval_X=X_valid,
        eval_y=y_valid,
        callbacks=[early_stopping(75,verbose=False, )]
    )

    predictions = model.predict(  X_valid )

    predictions = np.clip( predictions,0, None )

    return ( model, predictions)

def main():

    print("=" * 70)
    print( "STEP 8 - ROLLING TIME-SERIES " "CROSS-VALIDATION")
    print("=" * 70)


    print("\nLoading feature dataset...")

    if not FEATURES_TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature dataset: "
            f"{FEATURES_TRAIN_PATH}"
        )

    df = pd.read_csv( FEATURES_TRAIN_PATH,parse_dates=["Date"],   )

    df = df.sort_values( [ "Date", "Store",] ).reset_index(drop=True )

    print( f"Rows: {len(df):,}" )

    print( f"Stores: {df['Store'].nunique():,}")

    print(
        f"Date range: "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )


    excluded_columns = {"Sales", "Date", }

    feature_columns = [ 
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    print(
        f"Features: "
        f"{len(feature_columns)}"
    )

    # Load Optuna parameters

    print("\nLoading Optuna-selected parameters...")

    parameters = load_best_parameters()
    print( "\nSelected parameters:")

    for key, value in parameters.items():
        print(
            f"  {key}: {value}"
        )

    #  Create rolling folds

    folds = create_rolling_folds(
        dates=df["Date"],
        horizon=FORECAST_HORIZON,
        n_folds=N_FOLDS,
        step_size=STEP_SIZE,
    )

    if not folds:
        raise ValueError( "No rolling validation folds could be created.")

    print( "\nRolling validation folds:")

    for fold in folds:
        print(
            f" Fold {fold['Fold']}: "
            f"train <= "
            f"{fold['TrainingEnd'].date()}, "
            f"validate "
            f"{fold['ValidationStart'].date()} "
            f"to "
            f"{fold['ValidationEnd'].date()}"
        )

    #  Run rolling validation

    fold_results = []

    for fold in folds:

        fold_number = fold["Fold"]

        training_end = fold[ "TrainingEnd"]

        validation_start = fold["ValidationStart" ]

        validation_end = fold[ "ValidationEnd"]

        print( "\n" + "=" * 70 )

        print(f"FOLD {fold_number}" )

        print( "=" * 70)

        print(
            f"Training through: "
            f"{training_end.date()}"
        )

        print(
            f"Validation: "
            f"{validation_start.date()} "
            f"to "
            f"{validation_end.date()}"
        )


        train_df = df[  df["Date"] <= training_end ].copy()

        validation_df = df[
            (df["Date"] >= validation_start )
            & 
            ( df["Date"] <= validation_end  ) ].copy()

        if train_df.empty:
            raise ValueError(
                f"Fold {fold_number}: "
                "training data is empty."
            )

        if validation_df.empty:
            raise ValueError(
                f"Fold {fold_number}: "
                "validation data is empty."
            )

        print(
            f"Training rows: "
            f"{len(train_df):,}"
        )

        print(
            f"Validation rows: "
            f"{len(validation_df):,}"
        )


        print( "\nTraining LightGBM..." )

        model, predictions = (
            train_fold_model(
                train_df,
                validation_df,
                feature_columns,
                parameters,
            )
        )



        actual = validation_df["Sales" ].values
        mae = calculate_mae(actual,predictions,)
        rmse = calculate_rmse( actual, predictions,)
        wape = calculate_wape( actual,   predictions,)
        best_iteration = getattr( model, "best_iteration_", np.nan, )

        fold_result = {
            "Fold": fold_number,
            "TrainingEnd": training_end,
            "ValidationStart": validation_start,
            "ValidationEnd": validation_end,
            "TrainingRows": len(train_df),
            "ValidationRows": len(validation_df),
            "MAE": mae,
            "RMSE": rmse,
            "WAPE": wape,
            "BestIteration": best_iteration,
        }

        fold_results.append( fold_result )

        print( "\nFold results:" )

        print(
            f"  MAE: "
            f"{mae:.4f}"
        )

        print(
            f"  RMSE: "
            f"{rmse:.4f}"
        )

        print(
            f"  WAPE: "
            f"{wape:.4f}%"
        )

        print(
            f"  Best iteration: "
            f"{best_iteration}"
        )


    results_df = pd.DataFrame(  fold_results  )

    #  Calculate stability summary

    summary_rows = []

    metric_columns = [
        "MAE",
        "RMSE",
        "WAPE",
    ]

    for metric in metric_columns:

        values = results_df[ metric ].dropna()

        summary_rows.append(
            {
                "Metric": metric,
                "Mean": values.mean(),
                "Median": values.median(),
                "Std": values.std(
                    ddof=1
                ),
                "Min": values.min(),
                "Max": values.max(),
            }
        )

    summary_df = pd.DataFrame(summary_rows )

    #  Save results

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        ROLLING_RESULTS_PATH,
        index=False,
    )

    summary_df.to_csv(
        ROLLING_SUMMARY_PATH,
        index=False,
    )

    #  Print final summary

    print( "\n" + "=" * 70)

    print( "ROLLING CROSS-VALIDATION SUMMARY")

    print( "=" * 70 )

    print(
        results_df[
            [
                "Fold",
                "ValidationStart",
                "ValidationEnd",
                "MAE",
                "RMSE",
                "WAPE",
                "BestIteration",
            ]
        ].to_string(
            index=False
        )
    )

    print( "\nMetric stability:" )

    print(summary_df.to_string(   index=False ))

    #  Overall model assessment

    mean_wape = results_df[ "WAPE"].mean()

    std_wape = results_df[ "WAPE" ].std(  ddof=1  )

    mean_mae = results_df[ "MAE"].mean()

    print( "\nOverall rolling-CV performance:" )

    print(
        f"  Mean MAE: "
        f"{mean_mae:.4f}"
    )

    print(
        f"  Mean WAPE: "
        f"{mean_wape:.4f}%"
    )

    print(
        f"  WAPE Std: "
        f"{std_wape:.4f}"
    )

    print( "\nSaved:" )

    print(  ROLLING_RESULTS_PATH )

    print( ROLLING_SUMMARY_PATH)

    print("\nStep 8 completed successfully.")


if __name__ == "__main__":
    main()


    