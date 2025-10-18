import pandas as pd
import joblib
import os
import json
from datetime import datetime

from src.utils.config import load_config
from src.utils.model_loader import load_latest_model
from src.explain.explainer import ModelExplainer

class CryptoPredictor:
    def __init__(self, config_path="config/config.yaml"):
        self.config = load_config(config_path)
        self.model, self.metadata = self.load_model()
        self.features = self.config["features"]

        self.log_path = "logs/inference_log.jsonl"
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)


    def load_model(self):
        model_path = self.config.get("model_path")
        if model_path:
            print(f"📦 Loading model from config path: {model_path}")
            model = joblib.load(model_path)
            return model, {"model_path": model_path}
        else:
            print("📦 Loading latest available model...")
            return load_latest_model()

    def load_latest_features(self):
        df = pd.read_csv(self.config["features_path"], index_col="open_time", parse_dates=True)
        df = df.dropna()
        return df[self.features].iloc[[-1]]

    def predict(self, X):
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0][pred]
        return pred, proba

    def explain(self, X):
        explainer = ModelExplainer(self.model, self.features)
        explainer.explain_instance(X)

    def run(self):
        X = self.load_latest_features()
        pred, proba = self.predict(X)

        sentiment = X.get("sentiment_score", [None])[0]  # optional

        print(f"📈 Prediction: {'UP' if pred else 'DOWN'} (confidence: {proba:.2f})")
        self.explain(X)

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_path": self.metadata["model_path"],
            "features": X.to_dict(orient="records")[0],
            "prediction": int(pred),
            "probability": float(proba),
            "sentiment_score": sentiment,
            "actual": None, # to be filled after
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

