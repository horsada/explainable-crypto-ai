from pathlib import Path
from datetime import datetime
import json
import yaml
import pandas as pd
import numpy as np
import typer
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

from excrypto.data import FeaturePipeline, FeatureConfig
from excrypto.training.trainer import ModelTrainer, TrainerConfig

app = typer.Typer(help="Evaluate model with a chronological split and baselines")

def _chronological_split(X, y, ratio=0.8):
    n = len(X)
    cut = max(1, int(n * ratio))
    return (X.iloc[:cut], y.iloc[:cut]), (X.iloc[cut:], y.iloc[cut:])

@app.command()
def main(config: str = typer.Option(..., "--config", "-c", help="YAML config path")):
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    df = pd.read_csv(cfg["data_path"], parse_dates=["open_time"])

    pipe = FeaturePipeline(FeatureConfig(**cfg["features"]))
    X, y = pipe.transform(df, compute_label=True)
    if y is None or len(X) < 10:
        raise SystemExit("Not enough data for evaluation.")

    (Xtr, ytr), (Xte, yte) = _chronological_split(X, y)

    # train
    tcfg = cfg.get("trainer", {})
    trainer = ModelTrainer(pipeline=None, config=TrainerConfig(
        output_dir=Path(tcfg.get("output_dir", "models")),
        random_state=tcfg.get("random_state", 42),
        lgbm_params=tcfg.get("lgbm_params"),
    ))
    trainer.fit_Xy(Xtr, ytr)

    # preds
    yhat = trainer.model.predict(Xte)
    acc = float(accuracy_score(yte, yhat))

    proba = None
    auc = None
    brier = None
    if hasattr(trainer.model, "predict_proba"):
        proba = trainer.model.predict_proba(Xte)[:, 1]
        # AUC needs both classes present
        if len(np.unique(yte)) > 1:
            auc = float(roc_auc_score(yte, proba))
        brier = float(brier_score_loss(yte, proba))

    # baselines
    last_move = (yte.shift(1).fillna(0)).astype(int)  # previous label
    acc_last = float(accuracy_score(yte, last_move))

    majority = int(ytr.mean() >= 0.5)
    acc_major = float(accuracy_score(yte, np.full_like(yte, majority)))

    metrics = {
        "samples_train": int(len(Xtr)),
        "samples_test": int(len(Xte)),
        "accuracy": acc,
        "auc": auc,
        "brier": brier,
        "baseline_last_move_acc": acc_last,
        "baseline_majority_acc": acc_major,
    }

    Path("logs").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = Path("logs") / f"metrics_{ts}.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    typer.echo(json.dumps(metrics, indent=2))
    typer.echo(f"Saved metrics to {out}")

if __name__ == "__main__":
    app()
