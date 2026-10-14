from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.evaluation import evaluate_forecast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features_train.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

VALIDATION_DAYS = 30
RANDOM_STATE = 42

MAX_TRAIN_ROWS = 300_000

def load_features():
    """Load feature-engineered training data."""
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH} Run feature_engineering.py first")
    
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    
    return df.sort_values(["Date", "Store"]).reset_index(drop=True)


def chronological_split(df, validation_days = VALIDATION_DAYS):
    """Split by calendar date, not by random rows."""
    
    last_date = df["Date"].max()
    cutoff = last_date - pd.Timedelta(days=validation_days-1)
    
    train = df[df["Date"]< cutoff].copy()
    valid = df[df["Date"] >= cutoff].copy()
    
    if train.empty or valid.empty:
        raise ValueError("Training or validation data is empty.")
    
    return train, valid, cutoff

def make_model_features(df):
    """
    Select features available when the daily forecast is made.

    Customers and Sales are excluded because they are target-day or post-outcome information.
    """
    
    excluded = {
        "Sales",
        "Customers",
        "Date",
        "IsTrain"
    }
    
    features = df.drop( columns=[col for col in excluded if col in df.columns], errors = "ignore").copy()
    
    if "Store" in features.columns:
        features["Store"] = features["Store"].astype(str)
        
    categorical_columns = features.select_dtypes(include=["object","category","bool"]).columns.tolist()
    
    features = pd.get_dummies(
        features,
        columns=categorical_columns,
        dummy_na=True,
        dtype=np.uint8
    )
    
    features = features.replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)
    
    return features



def align_features(train_features, valid_features):
    """Ensure train and validation have identical columns"""
    
    return train_features.align(
        valid_features,
        join="left",
        axis=1,
        fill_value=0
    
    )
    
def build_baseline_preictions(valid):
    """Create naive and moving avg forecasts"""
    
    prediction = valid.copy()
    
    prediction["NaivePrediction"] = (prediction["SalesLag_7"].fillna(0))
    prediction["MovingAveragePrediction"] = (prediction["SalesRollingMean_28"].fillna(0))
    prediction["NaivePrediction"] = (prediction["NaivePrediction"].clip(lower=0))
    prediction["MovingAveragePrediction"] = (prediction["MovingAveragePrediction"]).clip(lower=0)
    
    return prediction

def train_random_forest(train, valid):
    """Train and predict with a baseline Random Forest"""
    
    train_x = make_model_features(train)
    valid_x = make_model_features(valid)
    
    train_x, valid_x = align_features(train_x, valid_x)
    
    train_y = train["Sales"].astype(float)
    
    if len(train_x) > MAX_TRAIN_ROWS:
        sample_indices = train_x.sample(n=MAX_TRAIN_ROWS, random_state=RANDOM_STATE).index
        
        train_x = train_x.loc[sample_indices]
        train_y = train_y.loc[sample_indices]
        
    
    model = RandomForestRegressor(n_estimators=100, max_depth=18, min_samples_leaf=5, max_features=0.8, n_jobs=-1,random_state=RANDOM_STATE)
    
    model.fit(train_x, train_y)
    
    predictions = model.predict(valid_x)
    
    return model, np.clip(predictions, 0, None), valid_x.columns


def compare_models(valid, rf_predictions):
    """Evaluate all three approaches on the same rows"""
    
    actual = valid["Sales"].to_numpy()
    
    prediction_map = {
        "Naive_7_Day": valid["SalesLag_7"].fillna(0).to_numpy(),
        "Moving_Average_28_Day": (valid["SalesRollingMean_28"].fillna(0).to_numpy()),
        'Random_Forest': rf_predictions
    }
    
    results = []
    
    for model_name, predictions in prediction_map.items():
        predictions = np.clip(predictions, 0, None)
        
        metrics = evaluate_forecast(actual, predictions)
        
        results.append({"Model": model_name, **metrics})
        
    return pd.DataFrame(results)


def plot_model_comparison(results):
    """Plot WAPE for the baseline models"""
    
    FIGURE_DIR.mkdir(parents=True,exist_ok=True)
    
    plt.figure(figsize=(9,5))
    
    plt.bar(results["Model"], results["WAPE"])
    plt.title("Baseline Model Comparison: WAPE")
    plt.xlabel("Model")
    plt.ylabel("WAPE (%)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    plt.savefig(FIGURE_DIR/ "baseline_model_comparison.png",dpi=150, bbox_inches="tight")
    
    plt.close()
    

def main():
    df = load_features()
    
    train, valid, cutoff = chronological_split(df)
    
    print(f"Validation starts: {cutoff.date()}")
    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(valid):,}")
    
    baseline = build_baseline_preictions(valid)
    
    print("\nTraining Random Forest...")
    model, rf_predictions, feature_columns = train_random_forest(
        train,
        valid,
    )

    results = compare_models(
        valid,
        rf_predictions,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        REPORT_DIR / "baseline_metrics.csv",
        index=False,
    )

    baseline["RandomForestPrediction"] = rf_predictions

    baseline[
        [
            "Store",
            "Date",
            "Sales",
            "NaivePrediction",
            "MovingAveragePrediction",
            "RandomForestPrediction",
        ]
    ].to_csv(
        REPORT_DIR / "baseline_predictions.csv",
        index=False,
    )

    plot_model_comparison(results)

    print("\nBaseline results:")
    print(results.to_string(index=False))

    print("\nSaved:")
    print(REPORT_DIR / "baseline_metrics.csv")
    print(REPORT_DIR / "baseline_predictions.csv")
    print(FIGURE_DIR / "baseline_model_comparison.png")


if __name__ == "__main__":
    main()

    
    