from __future__ import annotations
import joblib
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier

@dataclass
class SKLearnClassifier:
    model: object = None

    @classmethod
    def make(cls, kind: str = "rf", **kwargs) -> "SKLearnClassifier":
        if kind == "rf":
            m = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1, **kwargs)
        else:
            raise ValueError(f"Unknown model kind: {kind}")
        return cls(model=m)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_score(self, X) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(X)
            if p.shape[1] == 2:
                return p[:, 1]
            # multi-class: take long-vs-rest prob as score
            return p[:, p.shape[1]-1]
        if hasattr(self.model, "decision_function"):
            s = self.model.decision_function(X)
            return s if s.ndim == 1 else s[:, -1]
        return self.model.predict(X)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "SKLearnClassifier":
        return cls(model=joblib.load(path))
