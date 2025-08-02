import pandas as pd
import joblib
import shap

model = joblib.load("models/price_predictor.pkl")

def lambda_handler(event, context):
    features = ["return", "volatility", "sma_5", "sma_10", "sma_diff"]
    df = pd.read_csv("data/features.csv", index_col="open_time", parse_dates=True)
    latest = df[features].dropna().iloc[[-1]]

    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0][pred]

    return {
        "prediction": "UP" if pred else "DOWN",
        "confidence": round(prob, 4)
    }
