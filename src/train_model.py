import pandas as pd
from lightgbm import LGBMClassifier
import joblib

def label_data(df):
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna()

def train_model(df):
    features = ["return", "volatility", "sma_5", "sma_10", "sma_diff"]
    X = df[features]
    y = df["target"]

    model = LGBMClassifier()
    model.fit(X, y)
    return model

if __name__ == "__main__":
    df = pd.read_csv("data/features.csv", index_col="open_time", parse_dates=True)
    df = label_data(df)
    model = train_model(df)
    joblib.dump(model, "models/price_predictor.pkl")
    print("Model trained and saved to models/price_predictor.pkl")
