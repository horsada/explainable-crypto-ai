import pandas as pd

def engineer_features(df):
    df["return"] = df["close"].pct_change()
    df["volatility"] = df["return"].rolling(window=5).std()
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["sma_diff"] = df["sma_5"] - df["sma_10"]
    df = df.dropna()
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/binance_data.csv", index_col="open_time", parse_dates=True)
    df_feat = engineer_features(df)
    df_feat.to_csv("data/features.csv")
    print("Saved engineered features to data/features.csv")
