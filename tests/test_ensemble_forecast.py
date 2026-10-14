import numpy as np

from src.ensemble_forecast import (
    calculate_mae,
    calculate_rmse,
    calculate_wape,
    ets_forecast,
    seasonal_naive_forecast,
)


def test_mae():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 33])

    result = calculate_mae(
        y_true,
        y_pred,
    )

    assert result > 0


def test_rmse():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([10, 20, 30])

    result = calculate_rmse(
        y_true,
        y_pred,
    )

    assert result == 0


def test_wape():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([90, 210, 290])

    result = calculate_wape(
        y_true,
        y_pred,
    )

    assert result >= 0


def test_seasonal_naive_length():
    history = np.arange(14)

    forecast = seasonal_naive_forecast(
        history,
        30,
    )

    assert len(forecast) == 30


def test_ets_forecast_length():
    history = np.tile(
        np.array(
            [
                100,
                120,
                140,
                130,
                110,
                90,
                80,
            ]
        ),
        10,
    )

    forecast = ets_forecast(
        history,
        14,
    )

    assert len(forecast) == 14
    assert np.all(
        forecast >= 0
    )