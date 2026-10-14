import numpy as np
import pandas as pd

from src.multi_horizon_forecast import (
    create_fixed_horizon_dataset,
    evaluate_horizon,
)


def make_sample_data():

    dates = pd.date_range(
        "2024-01-01",
        periods=50,
        freq="D",
    )

    rows = []

    for store in [1, 2]:

        for i, date in enumerate(dates):

            rows.append({
                "Store": store,
                "Date": date,
                "Sales": float(
                    100 + i
                ),
            })

    return pd.DataFrame(rows)


def test_fixed_horizon():

    df = make_sample_data()

    origin = pd.Timestamp(
        "2024-02-10"
    )

    historical, future = (
        create_fixed_horizon_dataset(
            df,
            origin,
            7,
        )
    )

    assert (
        historical["Date"].max()
        <= origin
    )

    assert (
        future["Date"].min()
        == origin
        + pd.Timedelta(days=1)
    )

    assert (
        future["Date"].max()
        == origin
        + pd.Timedelta(days=7)
    )


def test_no_overlap():

    df = make_sample_data()

    origin = pd.Timestamp(
        "2024-02-10"
    )

    historical, future = (
        create_fixed_horizon_dataset(
            df,
            origin,
            7,
        )
    )

    historical_dates = set(
        historical["Date"]
    )

    future_dates = set(
        future["Date"]
    )

    assert historical_dates.isdisjoint(
        future_dates
    )


def test_evaluate_horizon():

    results = pd.DataFrame({
        "Store": [1, 1, 2],
        "Date": pd.date_range(
            "2024-01-01",
            periods=3,
        ),
        "Sales": [100, 200, 300],
        "Prediction": [110, 190, 280],
        "Horizon": [7, 7, 7],
    })

    metrics = evaluate_horizon(
        results
    )

    assert metrics["Horizon"] == 7

    assert np.isclose(
        metrics["MAE"],
        13.3333333333,
    )