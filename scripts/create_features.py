import pandas as pd
import os
import argparse
from datetime import datetime
from src.agents.sentiment_scout import RedditSentimentAgent

def engineer_features(df):
    df["return"] = df["close"].pct_change()
    df["volatility"] = df["return"].rolling(window=5).std()
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["sma_diff"] = df["sma_5"] - df["sma_10"]
    return df.dropna()

def fetch_sentiment_score():
    agent = RedditSentimentAgent(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent="crypto-sentiment-agent"
    )
    return agent.get_score()

def full_preprocess():
    df = pd.read_csv("data/binance_data.csv", index_col="open_time", parse_dates=True)
    df_feat = engineer_features(df)

    # Add sentiment only to the last row
    score = fetch_sentiment_score()
    df_feat.loc[df_feat.index[-1], "sentiment_score"] = score

    df_feat.to_csv("data/features.csv")
    print("✅ Full preprocess complete → data/features.csv")

def append_latest():
    df = pd.read_csv("data/binance_data.csv", index_col="open_time", parse_dates=True)
    df_feat = engineer_features(df)
    last_row = df_feat.tail(1).copy()

    score = fetch_sentiment_score()
    last_row["sentiment_score"] = score

    path = "data/features.csv"
    if os.path.exists(path):
        pd.concat([pd.read_csv(path), last_row]).to_csv(path, index=False)
    else:
        last_row.to_csv(path, index=False)

    print(f"✅ Appended new row with sentiment: {round(score, 3)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "append"], default="full")
    args = parser.parse_args()

    if args.mode == "full":
        full_preprocess()
    else:
        append_latest()
