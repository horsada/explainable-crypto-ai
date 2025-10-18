import pandas as pd
import joblib
from src.eval.backtester import Backtester

FEATURES = ["return", "volatility", "sma_5", "sma_10", "sma_diff"]
MODEL_PATH = "models/price_predictor.pkl"
DATA_PATH = "data/features.csv"

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, index_col="open_time", parse_dates=True)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna()

    model = joblib.load(MODEL_PATH)
    bt = Backtester(model, FEATURES)
    results = bt.run(df)

    acc, f1 = Backtester.evaluate(results)
    print(f"✅ Backtest complete — Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")

    results.to_csv("data/backtest_results.csv", index=False)
    print("📈 Results saved to data/backtest_results.csv")
