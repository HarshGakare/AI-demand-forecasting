
import numpy as np


def calculate_mae(y_true, y_pred):
   
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(np.abs(y_true - y_pred))


def calculate_rmse(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_wape(y_true, y_pred):
    

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return np.nan

    return (
        np.sum(np.abs(y_true - y_pred))
        / denominator
        * 100
    )


def evaluate_forecast(y_true, y_pred):


    return {
        "MAE": calculate_mae(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "WAPE": calculate_wape(y_true, y_pred),
    }

