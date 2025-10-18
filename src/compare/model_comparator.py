import pandas as pd
from joblib import load
from src.eval.evaluator import Evaluator

class ModelComparator:
    def __init__(self, model_paths: dict, features: list, df: pd.DataFrame):
        self.model_paths = model_paths
        self.features = features
        self.df = df.dropna()
        self.evaluator = Evaluator(output_path=None)  # disable per-model file saving

    def evaluate(self):
        results = {}
        X = self.df[self.features]
        y = self.df["target"]

        for name, path in self.model_paths.items():
            model = load(path)
            preds = model.predict(X)
            probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
            metrics = self.evaluator.evaluate(y, preds, y_prob=probs)
            results[name] = metrics

        return pd.DataFrame(results).T  # models as rows
