from pathlib import Path
import pandas as pd 
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_DIR / "train_merged.csv"
TEST_PATH = PROCESSED_DIR / "test_merged.csv"

OUTPATH_TRAIN = PROCESSED_DIR / "features_train.csv"
OUTPATH_TEST = PROCESSED_DIR / "features_test.csv"

LAG_DAYS = [7,14,28,35]
ROLLING_WINDOWS = [7,14,28]

def load_data():
    """Load merged training and test datasets"""
    
    
    
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TRAIN_PATH} Run src/data_loader.py first"
        )
    
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TEST_PATH}. Run src/data_loader.py first"
        )
        
    train = pd.read_csv(
    TRAIN_PATH,
    parse_dates=["Date"],
    low_memory=False,
    )

    test = pd.read_csv(
        TEST_PATH,
        parse_dates=["Date"],
        low_memory=False,
    )
    return train, test


def add_calendar_features(df):
    """Create Date-based Features"""
    
    df  = df.copy()
    
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["WeekOfYear"] = (df["Date"].dt.isocalendar().week.astype(int))
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Quarter"] = df["Date"].dt.quarter
    df["IsWeekend"] = ( df["DayOfWeek"].isin([5,6]).astype(int))
    df["IsMonthStart"] = (df["Date"].dt.is_month_start.astype(int))
    df["IsMonthEnd"] = (df["Date"].dt.is_month_end.astype(int))
    df["MonthSin"] = np.sin(2* np.pi * df["Month"] / 12)
    df["MonthCos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["WeekdaySin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["WeekdayCos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
    
    return df


def add_store_features(df):
    """Create and clean store-related features"""
    
    df = df.copy()
    
    if{"CompetitionOpenSinceYear","CompetitionOpenSinceMonth"}.issubset(df.columns):
        df["CompetitionOpenSinceYear"] = (df["CompetitionOpenSinceYear"].fillna(0))
        df["CompetitionOpenSinceMonth"] = (df["CompetitionOpenSinceMonth"].fillna(0))
        df["CompetitionOpenMonths"] = (
            (df["Year"] - df["CompetitionOpenSinceYear"]) * 12
            + df["Month"]
            - df["CompetitionOpenSinceMonth"]
        )
        df["CompetitionOpenMonths"] = (
            df["CompetitionOpenMonths"].clip(lower=0)
        )
        
        df["HasCompetition"] = (
            df["CompetitionDistance"].notna().astype(int)
            if "CompetitionDistance" in df.columns
            else 0
        )
        
    if "CompetitionDistance" in df.columns:
        df["CompetitionDistance"] = (df["CompetitionDistance"].fillna(-1))
        
    if{ "Promo2SinceYear","Promo2SinceWeek"}.issubset(df.columns):
        df["Promo2SinceYear"] = (df["Promo2SinceYear"].fillna(0))
        df["Promo2SinceWeek"] = (df["Promo2SinceWeek"].fillna(0))
        
        df["Promo2Active"] = (
            (df["Promo2"] == 1) & ( (df["Year"] > df["Promo2SinceYear"]) )
            |
            ((df["Year"] == df["Promo2SinceYear"]) & (df["WeekOfYear"]>=df["Promo2SinceWeek"]))
        ).astype(int)
        
    if "PromoInterval" in df.columns:
        month_names = df["Date"].dt.strftime("%b")

        df["PromoIntervalActive"] = [
            int(
                month in str(interval).split(",")
                if pd.notna(interval) and str(interval).strip()
                else False
            )
            for interval, month in zip(
                df["PromoInterval"],
                month_names,
            )
        ]
        
    return df


def add_lag_features(df):
    """
    Create historical sales features.
    Sales are shifted within each store, so each row only uses
    sales observed before its forecast date.
    
    """
    df = df.copy()
    df =df.sort_values(["Store","Date"]).reset_index(drop=True)
    
    grouped_sales = df.groupby("Store", sort=False)["Sales"]
    
    for lag in LAG_DAYS:
        df[f"SalesLag_{lag}"] = grouped_sales.shift(lag)
    
    for window in ROLLING_WINDOWS:
        df[f"SalesRollingMean_{window}"] = (
            df.groupby("Store", sort=False)["Sales"]
            .transform(lambda series: (series.shift(1).rolling(window=window, min_periods=window).mean()))
        )
        df[f"SalesRollingStd_{window}"] = (
                    df.groupby("Store", sort=False)["Sales"]
                    .transform(lambda series: (series.shift(1).rolling(window=window, min_periods=window).std()))
                )
    
    return df


def build_training_features(train):
    """Build features for historical training obervation"""
    
    train = train.copy()
    
    train = add_calendar_features(train)
    train = add_store_features(train)
    train = add_lag_features(train)
    
    train = train.dropna(
        subset=[
            f"SalesLag_{max(LAG_DAYS)}",
            f"SalesRollingMean_{max(ROLLING_WINDOWS)}"
        ]
    )
    
    return train.reset_index(drop=True)


def build_test_features(train, test):
    """
    Build test features using the combined history available before the test period.
    For test rows, actual future sales are unavailable.This fuction uses the historical train data for lag
    """
    
    train = train.copy()
    test = test.copy()
    
    train["IsTrain"] = 1
    test["IsTrain"] = 0
    
    test["Sales"] = np.nan
    test["Customers"] = np.nan
    
    combined = pd.concat(
        [train, test],
        ignore_index=True,
        sort = False
    )
    
    combined = combined.sort_values(
        ["Store","Date"]
    ).reset_index(drop=True)
    
    combined = add_calendar_features(combined)
    combined = add_store_features(combined)
    combined = add_lag_features(combined)
    
    test_features = combined.loc[combined["IsTrain"]==0].copy()
    
    return test_features.reset_index(drop=True)


def chronological_split(df, validation_days=30):
    """Split data using the final dates as validation."""
    
    cutoff = ( df["Date"].max() - pd.Timedelta(days=validation_days - 1))
    
    train_part = df[df["Date"] < cutoff].copy()
    valid_part = df[df["Date"] >= cutoff].copy()
    
    if train_part.empty or valid_part.empty:
        raise ValueError("Chronological split produced an empty partition")
    
    return train_part, valid_part, cutoff
    
def main():
    train, test = load_data()

    # Impute known missing Open values for Store 622.
    # This is an inference based on historical weekday patterns,
    # not verified ground truth.
    missing_open = test["Open"].isna()

    if missing_open.any():
        affected = missing_open & (test["Store"] == 622)
        unexpected = missing_open & (test["Store"] != 622)

        if unexpected.any():
            stores = sorted(
                test.loc[unexpected, "Store"].unique().tolist()
            )
            raise ValueError(
                f"Unexpected missing Open values for stores: {stores}. "
                "Investigate before imputing."
            )

        test.loc[affected, "Open"] = 1

        print(
            "Feature engineering inferred Open=1 for "
            f"{int(affected.sum())} missing rows for Store 622."
        )

    print("Building training features...")


    print("Building training features...")
    train_features = build_training_features(train)

    print("Building test features...")
    test_features = build_test_features(train, test)
    
     # Diagnose missing store-opening status in test features.
    if "Open" in test_features.columns:
        missing_open = int(test_features["Open"].isna().sum())

        print(f"Missing Open values in test features: {missing_open}")

        if missing_open > 0:
            print(
                "WARNING: Open is missing for some test rows. "
                "Resolve these rows before generating final predictions."
            )
        else:
            print("PASS: No missing Open values in test features.")
    else:
        print("WARNING: Open column is absent from test features.")


    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_features.to_csv(
        OUTPATH_TRAIN,
        index=False,
    )

    test_features.to_csv(
        OUTPATH_TEST,
        index=False,
    )

    train_part, valid_part, cutoff = chronological_split(
        train_features,
        validation_days=30,
    )

    print("\nFeature engineering complete.")
    print(f"Training features: {train_features.shape}")
    print(f"Test features: {test_features.shape}")
    print(f"Validation cutoff: {cutoff.date()}")
    print(f"Train partition: {train_part.shape}")
    print(f"Validation partition: {valid_part.shape}")

    print(f"\nSaved: {OUTPATH_TRAIN}")
    print(f"Saved: {OUTPATH_TEST}")

    print("\nGenerated feature columns:")
    for column in train_features.columns:
        print(column)


if __name__ == "__main__":
    main()
