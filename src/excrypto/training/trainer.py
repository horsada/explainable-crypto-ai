from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import json
import joblib
import pandas as pd
from datetime import datetime
from lightgbm import LGBMClassifier

# Expect: from excrypto.pipeline import FeaturePipeline, FeatureConfig
# (import inside methods to avoid circulars if needed)

@dataclass
class TrainerConfig:
    output_dir: Path = Path("models")
    random_state: Optional[int] = 42
    lgbm_params: Optional[dict] = None  # e.g., {"num_leaves": 31, "learning_rate": 0.05}

class ModelTrainer:
    def __init__(self, pipeline, config: Optional[TrainerConfig] = None):
        """
        pipeline: FeaturePipeline instance (with FeatureConfig)
        """
        self.pipeline = pipeline
        self.cfg = config or TrainerConfig()
        params = {"random_state": self.cfg.random_state}
        if self.cfg.lgbm_params:
            params.update(self.cfg.lgbm_params)
        self.model = LGBMClassifier(**params)
        self.features: Sequence[str] = []  # filled after transform

    def prepare(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Run feature pipeline and return (X, y)."""
        X, y = self.pipeline.transform(df_raw, compute_label=True)
        if y is None:
            raise ValueError("Pipeline returned y=None; set compute_label=True in transform.")
        self.features = list(X.columns)
        return X, y

    def train(self, df_raw: pd.DataFrame):
        """End-to-end: features -> fit model."""
        X, y = self.prepare(df_raw)
        self.model.fit(X, y)
        return self.model

    def fit_Xy(self, X: pd.DataFrame, y: pd.Series):
        """Optional: fit directly on precomputed features/labels."""
        self.features = list(X.columns)
        self.model.fit(X, y)
        return self.model

    def save_model(self) -> Path:
        """Persist model and metadata next to it."""
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        model_path = self.cfg.output_dir / f"price_predictor_{ts}.pkl"
        meta_path  = self.cfg.output_dir / f"price_predictor_{ts}_meta.json"

        joblib.dump(self.model, str(model_path))
        metadata = {
            "timestamp": ts,
            "features": list(self.features),
            "model_path": str(model_path),
            "lgbm_params": getattr(self.model, "get_params", lambda: {})(),
        }
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(f"✅ Model saved to {model_path}")
        print(f"📄 Metadata saved to {meta_path}")
        return model_path
