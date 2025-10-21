# tests/test_cli_end_to_end.py
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from typer.testing import CliRunner

from excrypto.scripts.train import app as train_app
from excrypto.scripts.evaluate import app as eval_app
from excrypto.scripts.predict import app as predict_app
import excrypto.inference.predictor as pred_mod

def _write_train_config(tmp, data_path):
    cfg = {
        "data_path": str(data_path),
        "features": {
            "features": ["ret_1","lag_close_1","roll_mean_close_2","roll_std_close_2"],
            "returns_windows": [1],
            "lag_features": {"close":[1]},
            "rolling_mean": {"close":[2]},
            "rolling_std": {"close":[2]},
            "dropna": True, "fillna": None, "align_on_open_time": True,
        },
        "trainer": {"random_state": 42, "output_dir": str(tmp / "models")}
    }
    (tmp / "train.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return tmp / "train.yaml"

def _write_predict_config(tmp, data_path, model_path=None):
    cfg = {
        **({"model_path": str(model_path)} if model_path else {}),
        "data_path": str(data_path),
        "features": {
            "features": ["ret_1","lag_close_1","roll_mean_close_2","roll_std_close_2","sentiment_score"],
            "returns_windows": [1],
            "lag_features": {"close":[1]},
            "rolling_mean": {"close":[2]},
            "rolling_std": {"close":[2]},
            "dropna": True, "fillna": None, "align_on_open_time": True,
        },
    }
    (tmp / "predict.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return tmp / "predict.yaml"

def _write_raw_csv(tmp, n=40):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="T"),
        "close": 100 + np.cumsum(rng.normal(size=n)),
        "sentiment_score": np.clip(rng.normal(0.2, 0.1, n), 0, 1),
    })
    path = tmp / "raw.csv"
    df.to_csv(path, index=False)
    return path

def test_train_eval_predict_end_to_end(tmp_path, monkeypatch):
    runner = CliRunner()

    # 1) write tiny raw dataset + configs
    raw = _write_raw_csv(tmp_path)
    train_yaml = _write_train_config(tmp_path, raw)

    # 2) TRAIN
    res_train = runner.invoke(train_app, ["--config", str(train_yaml)])
    assert res_train.exit_code == 0, res_train.output
    models_dir = tmp_path / "models"
    pkl = sorted(models_dir.glob("price_predictor_*.pkl"))[-1]
    meta = pkl.with_name(pkl.stem + "_meta.json")
    assert pkl.exists() and meta.exists()

    # 3) EVALUATE (uses same train.yaml)
    res_eval = runner.invoke(eval_app, ["--config", str(train_yaml)])
    assert res_eval.exit_code == 0, res_eval.output
    # sanity: printed metrics and logs/metrics_*.json created
    assert "accuracy" in res_eval.output
    logs = Path("logs")
    metrics_files = list(logs.glob("metrics_*.json"))
    assert metrics_files, "No metrics file written"

    # 4) PREDICT (point at the saved model explicitly)
    predict_yaml = _write_predict_config(tmp_path, raw, model_path=pkl)

    class _NoopExplainer:
        def __init__(self, model, features): ...
        def explain_instance(self, X): ...
    
    monkeypatch.setattr(pred_mod, "ModelExplainer", _NoopExplainer, raising=True)

    res_pred = runner.invoke(predict_app, ["--config", str(predict_yaml)])
    assert res_pred.exit_code == 0, res_pred.output
    # sanity: printed UP/DOWN and confidence; logged JSONL
    assert "Prediction:" in res_pred.output
    log_file = Path("logs/inference_log.jsonl")
    assert log_file.exists() and log_file.stat().st_size > 0
