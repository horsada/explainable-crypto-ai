import pandas as pd
from src.compare.model_comparator import ModelComparator

if __name__ == "__main__":
    df = pd.read_csv("data/features.csv", index_col="open_time", parse_dates=True)

    model_paths = {
        "with_sentiment": "models/price_predictor_sentiment.pkl",
        "without_sentiment": "models/price_predictor_no_sentiment.pkl"
    }

    features = ["return", "volatility", "sma_5", "sma_10", "sma_diff", "sentiment_score"]

    comparator = ModelComparator(model_paths, features, df)
    results_df = comparator.evaluate()
    print("📊 Model Comparison:")
    print(results_df)
