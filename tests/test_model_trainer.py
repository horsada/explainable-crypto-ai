# tests/test_trainer_with_stub.py
import json
import pandas as pd
import numpy as np
import pytest
from datetime import datetime as _dt
from pathlib import Path

from excrypto.training.trainer import ModelTrainer, TrainerConfig
import excrypto.training.trainer as trainer_module  # for datetime patch

class StubPipe:
    def __init__(self, cols=("f1","f2")):
        self.cols = list(cols)
    def transform(self, df, compute_label=False):
        X = pd.DataFrame({c: df.index.to_series().astype(float) for c in self.cols}, index=df.index)
        y = (df["close"].shift(-1) > df["close"]).astype(int) if compute_label else None
        return (X.dropna(), y.loc[X.dropna().index] if y is not None else None)

class _FixedDatetime:
    @classmethod
    def now(cls): return _dt(2025,1,2,3,4,5)

def _make_df(n=20):
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(size=n))
    return pd.DataFrame({"close": close})

def test_train_with_stub_pipeline_and_predict():
    df = _make_df(50)
    trainer = ModelTrainer(StubPipe(("f1","f2")), TrainerConfig())
    model = trainer.train(df)
    assert hasattr(model, "classes_")
    # features captured from pipeline
    assert trainer.features == ["f1","f2"]

def test_save_model_with_stub_pipeline(tmp_path, capsys):
    df = _make_df(30)
    trainer = ModelTrainer(StubPipe(("a","b")), TrainerConfig(output_dir=tmp_path / "models", random_state=42))
    trainer.train(df)

    # patch deterministic timestamp
    orig_dt = trainer_module.datetime
    trainer_module.datetime = _FixedDatetime
    try:
        model_path = trainer.save_model()
        out = (capsys.readouterr().out + capsys.readouterr().err)

        ts = "2025-01-02_030405"
        assert model_path.name == f"price_predictor_{ts}.pkl"

        meta_path = model_path.with_name(model_path.stem + "_meta.json")
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["timestamp"] == ts
        assert meta["features"] == ["a","b"]

        # path compare OS-independently
        assert Path(meta["model_path"]).resolve() == model_path.resolve()
        assert isinstance(meta.get("lgbm_params", {}), dict)

        # print contains filenames (don’t assert absolute paths)
        assert "Model saved to " in out and model_path.name in out
        assert "Metadata saved to " in out and meta_path.name in out
    finally:
        trainer_module.datetime = orig_dt
