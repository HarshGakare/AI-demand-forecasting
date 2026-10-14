import numpy as np

from src.evaluation import(
    calculate_mae,
    calculate_rmse,
    calculate_wape,
    evaluate_forecast
)

def test_mae():
    actual = [100,200,300]
    predicted = [110,180,330]
    
    
    assert np.isclose(calculate_mae(actual, predicted),20)
    
    
def test_rmse():
    actual = [100,200,300]
    predicted = [110,180,330]
    
    expected = (((10**2 + 20**2 + 30**2) / 3) ** 0.5)
    
    assert np.isclose(calculate_rmse(actual, predicted), expected)
    

def test_wape():
    actual = [100, 200, 300]
    predicted = [110, 180, 330]

    expected = 60 / 600 * 100

    assert np.isclose( calculate_wape(actual, predicted), expected, )


def test_zero_actual_sales():
    actual = [0, 0]
    predicted = [0, 0]

    assert np.isnan( calculate_wape(actual, predicted))


def test_evaluation_dictionary():
    result = evaluate_forecast([100, 200],[110, 190],)

    assert set(result.keys()) == { "MAE", "RMSE", "WAPE",}
