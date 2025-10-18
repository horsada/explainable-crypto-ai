import pandas as pd
from src.training.trainer import ModelTrainer

FEATURES = ["return", "volatility", "sma_5", "sma_10", "sma_diff"]
DATA_PATH = "data/features.csv"
MODEL_PATH = "models/price_predictor.pkl"

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, index_col="open_time", parse_dates=True)
    trainer = ModelTrainer(features=FEATURES, model_output_path=MODEL_PATH)
    trainer.train(df)
    trainer.save_model()
