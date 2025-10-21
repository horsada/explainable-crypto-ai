from pathlib import Path
import json, joblib
import pandas as pd
from datetime import datetime

from excrypto.preprocessing import FeaturePipeline, FeatureConfig
from excrypto.utils import load_config, load_latest_model
from excrypto.explain.explainer import ModelExplainer

class CryptoPredictor:
    def __init__(self, config_path="config/predict.yaml"):
        self.config = load_config(config_path)
        self.model, self.metadata = self.load_model()

        # --- features: single source of truth = model metadata ---
        feat_list = self.metadata.get("features")
        if not feat_list:
            raise ValueError("Model metadata missing 'features'. "
                            "Ensure the trainer saved features in the _meta.json.")

        # allow other pipeline knobs from config but enforce feature list from metadata
        feat_cfg = {**self.config.get("features", {})}
        feat_cfg["features"] = list(feat_list)

        self.pipe = FeaturePipeline(FeatureConfig(**feat_cfg))
        self.features = self.pipe.cfg.features

        # logging
        log_path = (self.config.get("logging", {}) or {}).get("log_path", "logs/inference_log.jsonl")
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # optional: allow disabling explain via config
        self.do_explain = (self.config.get("logging", {}) or {}).get("explain", True)

    def load_model(self):
        model_path = self.config.get("model_path")
        if model_path:
            print(f"📦 Loading model from config path: {model_path}")
            model = joblib.load(model_path)
            # minimal metadata if file-only provided; trainer should have saved features in meta
            meta = {"model_path": model_path}
            # try to load sibling _meta.json to fetch features
            meta_path = Path(model_path).with_name(Path(model_path).stem + "_meta.json")
            if meta_path.exists():
                meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
            return model, meta
        print("📦 Loading latest available model...")
        return load_latest_model()  # expected to return (model, metadata)

    def load_latest_features(self):
        df = pd.read_csv(self.config["data_path"], parse_dates=["open_time"])
        X, _ = self.pipe.transform(df, compute_label=False)
        if X.empty:
            raise ValueError("No valid feature rows after preprocessing.")
        return X.iloc[[-1]]

    def predict(self, X):
        pred = int(self.model.predict(X)[0])
        proba = float(self.model.predict_proba(X)[0][pred]) if hasattr(self.model, "predict_proba") else None
        return pred, proba

    def explain(self, X):
        if self.do_explain:
            ModelExplainer(self.model, self.features).explain_instance(X)

    def run(self):
        X = self.load_latest_features()
        pred, proba = self.predict(X)
        sentiment = X["sentiment_score"].iloc[0] if "sentiment_score" in X.columns else None

        msg = f"📈 Prediction: {'UP' if pred else 'DOWN'}"
        if proba is not None:
            msg += f" (confidence: {proba:.2f})"
        print(msg)

        self.explain(X)

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_path": self.metadata.get("model_path"),
            "features": X.to_dict(orient="records")[0],
            "prediction": pred,
            "probability": proba,
            "sentiment_score": sentiment,
            "actual": None,
        }
        # append JSONL
        self.log_path.write_text(
            (self.log_path.read_text(encoding="utf-8") if self.log_path.exists() else "")
            + json.dumps(log_entry) + "\n",
            encoding="utf-8",
        )
