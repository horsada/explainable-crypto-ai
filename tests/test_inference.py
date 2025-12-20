# tests/test_crypto_predictor.py
import json
import pandas as pd
import numpy as np
from pathlib import Path
import pytest

# ✅ Correct module paths
from excrypto.inference import CryptoPredictor
import excrypto.inference.predictor as predictor_mod
import excrypto.data as preprocessing_mod


# ---------- STUBS ----------
class FakeModel:
    def __init__(self, pred=1, proba=0.8):
        self._pred = int(pred)
        self._proba = float(proba)

    def predict(self, X):
        return [self._pred]

    def predict_proba(self, X):
        p1 = self._proba
        return [[1 - p1, p1]]


class StubExplainer:
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def explain_instance(self, X):
        pass


class StubFeaturePipeline:
    """Pass-through of configured features from raw df."""
    def __init__(self, cfg):
        self.cfg = cfg

    def transform(self, df, compute_label=False):
        X = df[self.cfg.features].copy()
        return X.dropna(), None


# ---------- FIXTURES ----------
@pytest.fixture
def raw_csv(tmp_path):
    t = pd.date_range("2024-01-01 00:00:00", periods=5, freq="T")
    df = pd.DataFrame(
        {
            "open_time": t,
            "close": np.linspace(100, 101, len(t)),
            "f1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "f2": [1, 2, 3, 4, 5],
            "sentiment_score": [0.0, 0.2, 0.1, 0.3, 0.4],
        }
    )
    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def features_cfg():
    return {
        "features": ["f1", "f2", "sentiment_score"],
        "returns_windows": [],
        "lag_features": {},
        "rolling_mean": {},
        "rolling_std": {},
        "dropna": True,
        "fillna": None,
        "align_on_open_time": False,
    }


# ---------- TESTS ----------
def test_predictor_loads_model_from_config_and_logs(tmp_path, monkeypatch, raw_csv, features_cfg, capsys):
    # Patch names **inside excrypto.inference.predictor**
    def fake_load_config(_):
        return {
            "model_path": str(tmp_path / "some_model.pkl"),
            "data_path": str(raw_csv),
            "features": features_cfg,
        }

    class FakeJoblib:
        @staticmethod
        def load(_):
            return FakeModel(pred=1, proba=0.85)

    def fake_load_latest_model():
        raise AssertionError("Should not call load_latest_model when model_path is provided.")

    monkeypatch.setattr(predictor_mod, "load_config", fake_load_config, raising=True)
    monkeypatch.setattr(predictor_mod, "load_latest_model", fake_load_latest_model, raising=True)
    monkeypatch.setattr(predictor_mod, "joblib", FakeJoblib, raising=True)

    # Stub pipeline & explainer
    monkeypatch.setattr(preprocessing_mod, "FeaturePipeline", StubFeaturePipeline, raising=True)

    calls = {"explained": False}

    class ExplainerWithFlag(StubExplainer):
        def explain_instance(self, X):
            calls["explained"] = True

    monkeypatch.setattr(predictor_mod, "ModelExplainer", ExplainerWithFlag, raising=True)

    pred = CryptoPredictor(config_path="unused.yaml")
    X = pred.load_latest_features()
    assert not X.empty and set(["f1", "f2", "sentiment_score"]).issubset(X.columns)

    pred.run()
    out = (capsys.readouterr().out + capsys.readouterr().err)
    assert "Prediction:" in out and "confidence:" in out
    assert calls["explained"] is True

    log_path = Path("logs/inference_log.jsonl")
    assert log_path.exists()
    last_line = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    entry = json.loads(last_line)
    assert entry["prediction"] in (0, 1)
    assert isinstance(entry["probability"], float)
    for k in ["f1", "f2", "sentiment_score"]:
        assert k in entry["features"]


def test_predictor_uses_latest_model_when_no_model_path(tmp_path, monkeypatch, raw_csv, features_cfg):
    def fake_load_config(_):
        return {"data_path": str(raw_csv), "features": features_cfg}

    def fake_load_latest_model():
        return FakeModel(pred=0, proba=0.6), {"model_path": "latest.pkl"}

    monkeypatch.setattr(predictor_mod, "load_config", fake_load_config, raising=True)
    monkeypatch.setattr(predictor_mod, "load_latest_model", fake_load_latest_model, raising=True)
    monkeypatch.setattr(preprocessing_mod, "FeaturePipeline", StubFeaturePipeline, raising=True)

    pred = CryptoPredictor(config_path="unused.yaml")
    assert pred.metadata["model_path"] == "latest.pkl"

    X = pred.load_latest_features()
    assert not X.empty
    p, prob = pred.predict(X)
    assert p in (0, 1) and isinstance(prob, float)
