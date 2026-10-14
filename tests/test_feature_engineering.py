import numpy as np
import pandas as pd

from src.feature_engineering import(
    add_calendar_features,
    add_lag_features,
    chronological_split
)

def make_sample_data():
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    
    rows = []
    
    for store in [1,2]:
        for i , date in enumerate(dates):
            rows.append({"Store": store, "Date": date, "Sales": float(100 + i + store)})
            
    return pd.DataFrame(rows)

def test_calendar_features():
    df = make_sample_data()
    
    result = add_calendar_features(df)
    
    expected = [
        "Year",
        "Month",
        "Day",
        "DayOfWeek",
        "WeekOfYear",
        "IsWeekend",
        "MonthSin",
        "MonthCos",
    ]

    for column in expected:
        assert column in result.columns
        
def test_lag_features():
    df = make_sample_data()
    
    result = add_lag_features(df)
    
    store_one = (
        result[result["Store"] == 1].sort_values("Date").reset_index(drop=True)
    )
    
    assert np.isnan(store_one.loc[0, "SalesLag_7"])
    assert store_one.loc[7, "SalesLag_7"] == 101.0
    

def test_lag_does_not_cross_stores():
    df = make_sample_data()
    result = add_lag_features(df)
    
    store_two = (
        result[result["Store"] == 2]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    assert store_two.loc[7, "SalesLag_7"] == 102.0
    

def test_rolling_mean_uses_past_only():
    df = make_sample_data()
    
    result = add_lag_features(df)
    
    store_one = (
        result[result["Store"] == 1]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # At index 7, the rolling mean uses indices 0 through 6.
    expected = np.mean([101, 102, 103, 104, 105, 106, 107])

    assert np.isclose(
        store_one.loc[7, "SalesRollingMean_7"],
        expected,
    )
    

def test_chronological_split():
    df = make_sample_data()

    train, valid, cutoff = chronological_split(
        df,
        validation_days=10,
    )

    assert train["Date"].max() < cutoff
    assert valid["Date"].min() >= cutoff
    assert train["Date"].max() < valid["Date"].min()