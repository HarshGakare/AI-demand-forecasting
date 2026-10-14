import numpy as np
import pandas as pd

from src.lightgbm_tuning import (
    create_time_splits,
    prepare_features,
    create_model,
)


def make_sample_data():
    dates = pd.date_range(
        "2024-01-01",
        periods=120,
        freq="D",
    )

    rows = []

    for store in [1, 2]:

        for i, date in enumerate(dates):

            rows.append({
                "Store": store,
                "Date": date,
                "Sales": float(
                    100
                    + i
                    + store
                ),
                "Promo": i % 2,
                "StoreType": "a",
                "SalesLag_7": (
                    90 + i
                ),
                "SalesRollingMean_7": (
                    95 + i
                ),
            })

    return pd.DataFrame(rows)


def test_time_split_is_chronological():

    df = make_sample_data()

    (
        train,
        tuning_valid,
        final_valid,
    ) = create_time_splits(df)

    assert (
        train["Date"].max()
        < tuning_valid["Date"].min()
    )

    assert (
        tuning_valid["Date"].max()
        < final_valid["Date"].min()
    )


def test_time_split_has_no_overlap():

    df = make_sample_data()

    (
        train,
        tuning_valid,
        final_valid,
    ) = create_time_splits(df)

    train_dates = set(
        train["Date"]
    )

    tuning_dates = set(
        tuning_valid["Date"]
    )

    final_dates = set(
        final_valid["Date"]
    )

    assert train_dates.isdisjoint(
        tuning_dates
    )

    assert train_dates.isdisjoint(
        final_dates
    )

    assert tuning_dates.isdisjoint(
        final_dates
    )


def test_prepare_features():

    df = make_sample_data()

    X, categorical_columns = (
        prepare_features(df)
    )

    assert "Sales" not in X.columns
    assert "Date" not in X.columns

    assert "StoreType" in X.columns

    assert (
        "StoreType"
        in categorical_columns
    )


def test_model_creation():

    params = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 8,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.01,
        "reg_lambda": 0.01,
        "min_split_gain": 0.0,
    }

    model = create_model(params)

    assert (
        model.get_params()[
            "n_estimators"
        ]
        == 100
    )

    assert np.isclose(
        model.get_params()[
            "learning_rate"
        ],
        0.05,
    )