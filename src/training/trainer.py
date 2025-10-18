import pandas as pd
import joblib
import os
from datetime import datetime
from lightgbm import LGBMClassifier
import json

class ModelTrainer:
    def __init__(self, features, output_dir="models/"):
        self.features = features
        self.output_dir = output_dir
        self.model = LGBMClassifier()

    def label_data(self, df):
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        return df.dropna()

    def train(self, df):
        df = self.label_data(df)
        X = df[self.features]
        y = df["target"]
        self.model.fit(X, y)
        return self.model

    def save_model(self):
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        model_path = f"{self.output_dir}/price_predictor_{timestamp}.pkl"
        meta_path = f"{self.output_dir}/price_predictor_{timestamp}_meta.json"

        joblib.dump(self.model, model_path)

        metadata = {
            "timestamp": timestamp,
            "features": self.features,
            "model_path": model_path,
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Model saved to {model_path}")
        print(f"📄 Metadata saved to {meta_path}")
