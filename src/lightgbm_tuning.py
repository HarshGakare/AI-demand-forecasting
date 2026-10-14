from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from src.evaluation import evaluate_forecast


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (PROJECT_ROOT / "data" / "processed" / "features_train.csv")

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42

FINAL_VALIDATION_DAYS = 30
TUNING_VALIDATION_DAYS = 30

N_TRIALS = 100
MAX_TUNING_TRAIN_ROWS = 300_000



def load_features():
    """Load feature-engineered training data."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. "
            "Run src/feature_engineering.py first."
        )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["Date"],
    )

    df = df.sort_values(
        ["Date", "Store"]
    ).reset_index(drop=True)

    return df


def create_time_splits(df):
    """
    Create chronological training, tuning-validation,
    and final-validation datasets.
    """
    
    max_date = df["Date"].max()
    
    final_valid_start = (max_date - pd.Timedelta(days=FINAL_VALIDATION_DAYS-1))
    tuning_valid_end = (final_valid_start - pd.Timedelta(days=1))
    tuning_valid_start = (tuning_valid_end - pd.Timedelta(days=TUNING_VALIDATION_DAYS -1))
    
    train = df[df["Date"] < tuning_valid_start].copy()
    
    tuning_valid = df[(df["Date"] >= tuning_valid_start) & (df["Date"] <= tuning_valid_end)]
    
    final_valid = df[df["Date"] >= final_valid_start].copy()
    
    if train.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    if tuning_valid.empty:
        raise ValueError(
            "Tuning validation dataset is empty."
        )

    if final_valid.empty:
        raise ValueError(
            "Final validation dataset is empty."
        )

    return (
        train,
        tuning_valid,
        final_valid,
    )
    

def prepare_features(df):
    """
    Prepare LightGBM-compatible features.

    Target columns and date are excluded.
    Categorical columns are converted to pandas category dtype.
    Missing categorical values are represented explicitly.
    """
    
    df = df.copy()

    target_columns = {
        "Sales",
        "Customers",
        "Date",
        "IsTrain",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in target_columns
    ]
    
    X = df[feature_columns].copy()

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for column in categorical_columns:
        X[column] = (
            X[column]
            .astype("object")
            .where(
                X[column].notna(),
                "__MISSING__",
            )
            .astype("category")
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )
    
    return X, categorical_columns


def align_categories(train_x, other_x):
    """
    Make categorical columns use the same categories
    between training and validation data.
    """
    
    train_x = train_x.copy()
    other_x = other_x.copy()

    common_columns = (
        set(train_x.columns)
        & set(other_x.columns)
    )
    
    for column in common_columns:

        train_is_category = (
            str(train_x[column].dtype) == "category"
        )

        other_is_category = (
            str(other_x[column].dtype) == "category"
        )

        if not (
            train_is_category
            or other_is_category
        ):
            continue

       
        train_values = (
            train_x[column]
            .astype("object")
            .where(
                train_x[column].notna(),
                "__MISSING__",
            )
            .astype(str)
        )

        other_values = (
            other_x[column]
            .astype("object")
            .where(
                other_x[column].notna(),
                "__MISSING__",
            )
            .astype(str)
        )

        combined_categories = pd.Index(
            pd.concat(
                [
                    train_values,
                    other_values,
                ],
                ignore_index=True,
            ).unique()
        )

        
        combined_categories = (
            combined_categories[
                combined_categories.notna()
            ]
        )

        dtype = pd.api.types.CategoricalDtype(
            categories=combined_categories
        )

        train_x[column] = train_values.astype(dtype)
        other_x[column] = other_values.astype(dtype)

    return train_x, other_x


def sample_training_data(X, y):
    """
    Reduce the tuning dataset when necessary.

    Sampling is only used during hyperparameter search.
    Final training later uses all available training rows.
    """

    if len(X) <= MAX_TUNING_TRAIN_ROWS:
        return X, y
    
    rng = np.random.RandomState(RANDOM_STATE)

    indices = rng.choice(len(X), size=MAX_TUNING_TRAIN_ROWS, replace=False)
    indices = np.sort(indices)
    
    return(X.iloc[indices].copy(), y.iloc[indices].copy())


def create_model(params):
    """Create a LightGBM regression model."""

    return lgb.LGBMRegressor(
        objective="regression",
        random_state=RANDOM_STATE,
        n_jobs=-1,

        # Tuned parameters
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        min_child_samples=params["min_child_samples"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        min_split_gain=params["min_split_gain"],
        verbosity=-1,
    )
    

def build_fixed_tuning_data(train, tuning_valid):
    """
    Prepare data once so Optuna doesn't repeatedly
    perform expensive preprocessing.
    """
    
    train_x, categorical_columns = prepare_features(train)
    tuning_x, _  = prepare_features(tuning_valid)
    
    train_x, tuning_x = align_categories(train_x, tuning_x)
    
    train_y = train["Sales"].astype(float)
    tuning_y = tuning_valid["Sales"].astype(float)
    
    train_x, train_y = sample_training_data(train_x, train_y)
    
    return(
        train_x,
        train_y,
        tuning_x,
        tuning_y,
        categorical_columns,
    )
    
    
def create_objective(
    train_x,
    train_y,
    valid_x,
    valid_y,
    categorical_columns,
):
    """Create the Optuna objective function."""
    
    def objective(trail):
        params = {"n_estimators" : trail.suggest_int("n_estimators",300,1500, step=100),
                  "learning_rate": trail.suggest_float("learning_rate", 0.01, 0.15, log=True),
                  "num_leaves": trail.suggest_int("num_leaves", 20, 200),
                  "max_depth": trail.suggest_int("max_depth", 4, 15),
                  "min_child_samples": trail.suggest_int("min_child_samples", 10, 100),
                  "subsample": trail.suggest_float("subsample", 0.6, 1.0),
                  "colsample_bytree": trail.suggest_float("colsample_bytree", 0.6, 1.0),
                  "reg_alpha": trail.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                  "reg_lambda": trail.suggest_float("reg_lambda",1e-8, 10.0, log=True),
                  "min_split_gain": trail.suggest_float("min_split_gain", 0.0, 1.0),             
                  }
        
        model = create_model(params)
        
        model.fit(train_x, train_y, categorical_feature=categorical_columns, eval_set=[(valid_x,valid_y)],callbacks=[lgb.early_stopping(stopping_rounds=75,verbose=False,) ],)

        predictions = model.predict(valid_x, num_iteration=model.best_iteration_)
        predictions = np.clip(predictions,0,None)

        metrics = evaluate_forecast(valid_y,predictions)
        
        return metrics["MAE"]
    
    return objective


def run_optimization(train, tuning_valid):
    """Run the Optuna hyperparameter search."""

    (train_x, train_y, valid_x, valid_y, categorical_columns) =(build_fixed_tuning_data(train, tuning_valid))
    
    print("\n Strainf Optuna Optimization")
    print(f"Trials : {N_TRIALS}")
    
    study = optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    
    objective = create_objective(train_x, train_y, valid_x, valid_y, categorical_columns)
    
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    return (
        study,
        train_x,
        train_y,
        valid_x,
        valid_y,
        categorical_columns,
    )
    
def train_final_model(train,final_valid,best_params):
    """
    Train the final tuned model using all historical data
    before the final holdout.
    """
    
    train_x, categorical_columns = prepare_features(train)
    valid_x,_= prepare_features(final_valid)
    train_x, valid_x = align_categories(train_x, valid_x)
    train_y = train["Sales"].astype(float)
    valid_y = final_valid["Sales"].astype(float)
    
    model = create_model(best_params)
    
    model.fit(train_x, train_y, categorical_feature=categorical_columns,eval_set=[(valid_x, valid_y)], callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)])
    
    predictions = model.predict(valid_x, num_iteration=model.best_iteration_)
    predictions = np.clip(predictions,0,None)
    
    return(model, predictions, valid_y,valid_x,categorical_columns)


def save_outputs(study,model,predictions, final_valid,metrics):
    """Save tuning results, model, and predictions."""
    
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    
    model_path = (MODEL_DIR / "lightbgm_demand_model.joblib")
    
    joblib.dump(model,model_path)
    
    predictions_df = final_valid[["Store","Date","Sales"]].copy()
    predictions_df["LightGBMPrediction"] = predictions
    predictions_df.to_csv(REPORT_DIR / "lightgbm_predictions.csv", index=False)
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(REPORT_DIR / "lightgbm_metrics.csv", index=False)
    
    study.trials_dataframe().to_csv(REPORT_DIR/ "optuna_trials.csv",index=False, )

    best_params_df = pd.DataFrame( [ study.best_params ])
    best_params_df.to_csv(REPORT_DIR/ "lightgbm_best_params.csv",index=False,)

    print(f"\nModel saved to: {model_path}")
    print("Metrics saved to: "f"{REPORT_DIR / 'lightgbm_metrics.csv'}")
    print("Optuna trials saved to: "f"{REPORT_DIR / 'optuna_trials.csv'}")


def main():
    df = load_features()

    (train,tuning_valid,final_valid,) = create_time_splits(df)

    print("\nDataset split")
    print("-" * 50)
    print(
        f"Training: "
        f"{train['Date'].min().date()} "
        f"-> "
        f"{train['Date'].max().date()}"
    )

    print(
        f"Tuning validation: "
        f"{tuning_valid['Date'].min().date()} "
        f"-> "
        f"{tuning_valid['Date'].max().date()}"
    )

    print(
        f"Final validation: "
        f"{final_valid['Date'].min().date()} "
        f"-> "
        f"{final_valid['Date'].max().date()}"
    )

    print(
        f"\nTraining rows: "
        f"{len(train):,}"
    )

    print(
        f"Tuning validation rows: "
        f"{len(tuning_valid):,}"
    )

    print(
        f"Final validation rows: "
        f"{len(final_valid):,}"
    )

    (study,_,_,_,_,_,) = run_optimization(train,tuning_valid  )

    print("\nBest Optuna result")
    print("-" * 50)

    print(
        f"Best MAE: "
        f"{study.best_value:.4f}"
    )

    print("\nBest parameters:")

    for key, value in study.best_params.items():
        print(
            f"{key}: {value}"
        )

    print(
        "\nTraining final tuned LightGBM..."
    )

    (model,predictions,valid_y,valid_x,categorical_columns,) = train_final_model( train, final_valid, study.best_params,)

    metrics = evaluate_forecast(valid_y,predictions,)

    metrics["BestIteration"] = ( model.best_iteration_ )

    print("\nFinal LightGBM results")
    print("-" * 50)

    for key, value in metrics.items():
        print(
            f"{key}: {value:.4f}"
            if isinstance(value, (int, float))
            else f"{key}: {value}"
        )

    save_outputs( study, model, predictions, final_valid, metrics,)


if __name__ == "__main__":
    main()