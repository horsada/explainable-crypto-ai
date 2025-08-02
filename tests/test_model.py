import pandas as pd
import joblib

def test_model_predicts_valid_class():
    model = joblib.load("models/price_predictor.pkl")
    sample = pd.DataFrame([{
        "return": 0.01,
        "volatility": 0.02,
        "sma_5": 105,
        "sma_10": 103,
        "sma_diff": 2.0
    }])
    pred = model.predict(sample)[0]
    assert pred in [0, 1]
