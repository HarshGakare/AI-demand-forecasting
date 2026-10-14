import numpy as np
import pandas as pd

from src.rolling_cross_validation import (
    calculate_mae,
    calculate_rmse,
    calculate_wape,
    create_rolling_folds,
)


def test_mae_zero_error():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([10, 20, 30])

    assert calculate_mae(
        y_true,
        y_pred,
    ) == 0


def test_rmse_zero_error():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([10, 20, 30])

    assert calculate_rmse(
        y_true,
        y_pred,
    ) == 0


def test_wape_zero_error():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([100, 200, 300])

    assert calculate_wape(
        y_true,
        y_pred,
    ) == 0


def test_create_rolling_folds():

    dates = pd.date_range(
        "2015-01-01",
        "2015-12-31",
        freq="D",
    )

    folds = create_rolling_folds(
        dates=dates,
        horizon=30,
        n_folds=4,
        step_size=30,
    )

    assert len(folds) == 4

    for fold in folds:

        assert (
            fold["TrainingEnd"]
            < fold["ValidationStart"]
        )

        validation_days = (
            fold["ValidationEnd"]
            - fold["ValidationStart"]
        ).days + 1

        assert validation_days == 30


def test_rolling_folds_are_chronological():

    dates = pd.date_range(
        "2015-01-01",
        "2015-12-31",
        freq="D",
    )

    folds = create_rolling_folds(
        dates=dates,
        horizon=30,
        n_folds=4,
        step_size=30,
    )

    validation_starts = [
        fold["ValidationStart"]
        for fold in folds
    ]

    assert validation_starts == sorted(
        validation_starts
    )