
import numpy as np
import pandas as pd
import pytest

from src.recursive_forecaster import (
    add_calendar_features,
    build_recursive_row,
    calculate_promo_interval_active,
    evaluate_recursive_forecast,
    get_model_feature_names,
    prepare_model_features,
    recursive_forecast_store,
)
from src.evaluation import calculate_mae, calculate_rmse, calculate_wape


class DummyModel:
    """A small stand-in for a trained model."""

    feature_name_ = ["Store", "Year", "Month", "Day", "SalesLag_7"]

    def predict(self, X):
        # Predict the 7-day lag, or zero if that feature is missing.
        values = pd.to_numeric(X["SalesLag_7"], errors="coerce").fillna(0)
        return values.to_numpy(dtype=float)


def sample_static_features():
    """Static/store-level feature row indexed by Store."""
    return pd.DataFrame(
        {
            "Store": [1],
            "PromoInterval": ["Jan,Apr,Jul,Oct"],
            "StoreType": pd.Categorical(["a"], categories=["a", "b"]),
        }
    ).set_index("Store")


def sample_history():
    dates = pd.date_range("2025-01-01", periods=14, freq="D")
    return pd.DataFrame(
        {
            "Store": [1] * len(dates),
            "Date": dates,
            "Sales": np.arange(100, 114, dtype=float),
        }
    )


def test_get_model_feature_names_from_feature_name_attribute():
    model = DummyModel()
    assert get_model_feature_names(model) == model.feature_name_


def test_get_model_feature_names_from_booster():
    class Booster:
        def feature_name(self):
            return ["Store", "SalesLag_7"]

    class Model:
        booster_ = Booster()

    assert get_model_feature_names(Model()) == ["Store", "SalesLag_7"]


def test_get_model_feature_names_raises_if_names_unavailable():
    with pytest.raises(AttributeError, match="feature names"):
        get_model_feature_names(object())


def test_add_calendar_features_creates_expected_values():
    row = pd.Series({"Store": 1})
    date = pd.Timestamp("2025-01-04")  # Saturday
    result = add_calendar_features(row, date)

    assert result["Year"] == 2025
    assert result["Month"] == 1
    assert result["Day"] == 4
    assert result["DayOfWeek"] == 6  # Monday=1 ... Sunday=7
    assert result["IsWeekend"] == 1
    assert result["Quarter"] == 1
    assert "MonthSin" in result.index
    assert "WeekdayCos" in result.index


@pytest.mark.parametrize(
    ("interval", "date", "expected"),
    [
        ("Jan,Apr,Jul,Oct", "2025-01-15", 1),
        ("Jan,Apr,Jul,Oct", "2025-02-15", 0),
        ("", "2025-01-15", 0),
        (None, "2025-01-15", 0),
    ],
)
def test_calculate_promo_interval_active(interval, date, expected):
    assert calculate_promo_interval_active(
        interval, pd.Timestamp(date)
    ) == expected


def test_build_recursive_row_uses_calendar_lags_and_rolling_features():
    history = sample_history()
    result = build_recursive_row(
        store_id=1,
        forecast_date=pd.Timestamp("2025-01-15"),
        history=history,
        static_features=sample_static_features(),
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["Store"] == 1
    assert row["Date"] == pd.Timestamp("2025-01-15")
    assert row["SalesLag_7"] == 107.0  # Jan 8 sales
    assert row["SalesLag_14"] == 100.0  # Jan 1 sales
    assert row["SalesRollingMean_7"] == pytest.approx(110.0)
    assert row["SalesRollingStd_7"] == pytest.approx(
        pd.Series([107, 108, 109, 110, 111, 112, 113], dtype=float).std()
    )
    assert row["PromoIntervalActive"] == 1


def test_build_recursive_row_raises_for_unknown_store():
    with pytest.raises(ValueError, match="not found"):
        build_recursive_row(
            store_id=99,
            forecast_date=pd.Timestamp("2025-01-15"),
            history=sample_history(),
            static_features=sample_static_features(),
        )


def test_prepare_model_features_orders_columns_and_sets_categories():
    model_features = ["Store", "Year", "StoreType"]
    row_df = pd.DataFrame(
        {
            "Store": [1],
            "Year": [2025],
            "StoreType": ["a"],
            "UnusedFeature": [123],
        }
    )
    metadata = {
        "StoreType": {
            "categories": pd.Index(["a", "b"]),
            "ordered": False,
        }
    }

    result = prepare_model_features(
        row_df=row_df,
        model_feature_names=model_features,
        training_features_df=pd.DataFrame({"Store": [1]}),
        categorical_metadata=metadata,
    )

    assert list(result.columns) == model_features
    assert str(result["StoreType"].dtype) == "category"
    assert list(result["StoreType"].cat.categories) == ["a", "b"]
    assert result["Year"].iloc[0] == 2025


def test_prepare_model_features_raises_when_feature_is_missing():
    with pytest.raises(ValueError, match="Missing model features"):
        prepare_model_features(
            row_df=pd.DataFrame({"Store": [1]}),
            model_feature_names=["Store", "Year"],
            training_features_df=pd.DataFrame(),
            categorical_metadata={},
        )


def test_recursive_forecast_store_feeds_predictions_back_into_history():
    history = sample_history()
    model = DummyModel()
    static_features = sample_static_features()
    metadata = {}

    predictions = recursive_forecast_store(
        store_id=1,
        forecast_start=pd.Timestamp("2025-01-15"),
        history_df=history,
        model=model,
        static_features=static_features,
        training_features_df=pd.DataFrame(),
        categorical_metadata=metadata,
        model_feature_names=model.feature_name_,
        horizon=3,
    )

    assert len(predictions) == 3
    assert predictions["Date"].tolist() == list(
        pd.date_range("2025-01-15", periods=3, freq="D")
    )
    assert (predictions["Prediction"] >= 0).all()
    # Dummy model predicts lag-7 when available, otherwise zero.
    assert predictions["Prediction"].tolist() == [107.0, 108.0, 109.0]


def test_evaluate_recursive_forecast_returns_metrics_and_matched_rows():
    predictions = pd.DataFrame(
        {
            "Store": [1, 1, 1],
            "Date": pd.date_range("2025-01-15", periods=3, freq="D"),
            "Prediction": [10.0, 20.0, 30.0],
        }
    )
    actuals = pd.DataFrame(
        {
            "Store": [1, 1, 1],
            "Date": pd.date_range("2025-01-15", periods=3, freq="D"),
            "Sales": [12.0, 18.0, 33.0],
        }
    )

    metrics, merged = evaluate_recursive_forecast(
        predictions=predictions,
        actual_sales=actuals,
        forecast_origin=pd.Timestamp("2025-01-14"),
    )

    assert len(merged) == 3
    assert metrics["Rows"] == 3
    assert metrics["Horizon"] == 3
    assert metrics["MAE"] == pytest.approx(calculate_mae(actuals["Sales"], predictions["Prediction"]))
    assert metrics["RMSE"] == pytest.approx(calculate_rmse(actuals["Sales"], predictions["Prediction"]))
    assert metrics["WAPE"] == pytest.approx(calculate_wape(actuals["Sales"], predictions["Prediction"]))


def test_evaluation_metrics_ignore_rows_without_actual_sales():
    predictions = pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": pd.to_datetime(["2025-01-15", "2025-01-16"]),
            "Prediction": [10.0, 20.0],
        }
    )
    actuals = pd.DataFrame(
        {
            "Store": [1],
            "Date": pd.to_datetime(["2025-01-15"]),
            "Sales": [12.0],
        }
    )

    metrics, merged = evaluate_recursive_forecast(
        predictions=predictions,
        actual_sales=actuals,
        forecast_origin=pd.Timestamp("2025-01-14"),
    )

    assert len(merged) == 1
    assert metrics["Rows"] == 1
    assert metrics["MAE"] == pytest.approx(2.0)
