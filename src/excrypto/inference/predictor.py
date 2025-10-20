from pathlib import Path
import pandas as pd, json, joblib
from datetime import datetime
from excrypto.preprocessing import FeaturePipeline, FeatureConfig
from excrypto.utils import load_config, load_latest_model
from excrypto.explain.explainer import ModelExplainer

class CryptoPredictor:
    def __init__(self, config_path="config/config.yaml"):
        self.config = load_config(config_path)
        self.model, self.metadata = self.load_model()
        self.pipe = FeaturePipeline(FeatureConfig(**self.config["features"]))
        self.features = self.pipe.cfg.features  # single source of truth
        self.log_path = Path("logs/inference_log.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        model_path = self.config.get("model_path")
        if model_path:
            print(f"📦 Loading model from config path: {model_path}")
            return joblib.load(model_path), {"model_path": model_path}
        print("📦 Loading latest available model...")
        return load_latest_model()

    def load_latest_features(self):
        # read RAW data, let pipeline build engineered features
        df = pd.read_csv(self.config["data_path"], parse_dates=["open_time"])
        X, _ = self.pipe.transform(df, compute_label=False)
        if X.empty:
            raise ValueError("No valid feature rows after preprocessing.")
        return X.iloc[[-1]]  # last valid row

    def predict(self, X):
        pred = int(self.model.predict(X)[0])
        proba = (
            float(self.model.predict_proba(X)[0][pred])
            if hasattr(self.model, "predict_proba")
            else None
        )
        return pred, proba

    def explain(self, X):
        ModelExplainer(self.model, self.features).explain_instance(X)

    def run(self):
        X = self.load_latest_features()
        pred, proba = self.predict(X)
        sentiment = X["sentiment_score"].iloc[0] if "sentiment_score" in X.columns else None
        print(f"📈 Prediction: {'UP' if pred else 'DOWN'}"
              f"{'' if proba is None else f' (confidence: {proba:.2f})'}")
        self.explain(X)

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_path": self.metadata["model_path"],
            "features": X.to_dict(orient="records")[0],
            "prediction": pred,
            "probability": proba,
            "sentiment_score": sentiment,
            "actual": None,
        }
        self.log_path.write_text(
            (self.log_path.read_text() if self.log_path.exists() else "") + json.dumps(log_entry) + "\n",
            encoding="utf-8"
        )
