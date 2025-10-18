import argparse
import pandas as pd
import joblib
from src.explain.explainer import ModelExplainer
from src.utils.config import load_config

def main(window, config_path):
    config = load_config(config_path)
    df = pd.read_csv(config["features_path"], index_col="open_time", parse_dates=True)
    df = df.dropna()

    if window:
        df = df.tail(window)

    model = joblib.load(config["model_path"])
    explainer = ModelExplainer(model, config["features"])
    explainer.explain_global(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, help="Number of most recent rows to explain")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    main(args.window, args.config)
