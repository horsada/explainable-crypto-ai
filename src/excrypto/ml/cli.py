# src/excrypto/ml/cli.py
from __future__ import annotations
import json, time
from pathlib import Path
import pandas as pd
import typer

from excrypto.ml.datasets import load_xy
from excrypto.ml.splitters import PurgedKFold
from excrypto.ml.models_sklearn import SKLearnClassifier
from excrypto.ml.evaluate import cls_metrics
from excrypto.utils.paths import RunPaths
from excrypto.utils.config import load_cfg, cfg_hash

app = typer.Typer(help="ML: train/predict from YAML configs")

def _write_manifest(paths: RunPaths, meta: dict):
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    target = getattr(paths, "manifest", paths.report_dir / "metadata.json")
    target.write_text(json.dumps(meta, indent=2, sort_keys=True))

def _feat_params_from_yaml(path: Path) -> dict:
    cfg = load_cfg(path)
    return {"hash": cfg_hash({"specs": cfg["specs"]})}

def _label_params_from_yaml(path: Path) -> dict:
    cfg = load_cfg(path)
    if cfg.get("kind", "fh") == "fh":
        canon = {"kind":"fh","h": int(cfg["h"]), "thr": float(cfg.get("thr",0.0)), "mode": cfg.get("mode","cls")}
    else:
        canon = {"kind":"tb","h": int(cfg["h"]), "u": float(cfg["u"]), "d": float(cfg["d"]), "w": int(cfg["w"])}
    return {"hash": cfg_hash(canon), **canon}

@app.command("train")
def train(cfg: Path = typer.Option(..., "--config", exists=True, dir_okay=False, help="ML train YAML")):
    mlcfg = load_cfg(cfg)

    # datasetcfg: Optional[Path] = typer.Option(None, "--config", help="YAML with 'specs' list"),
    ds = mlcfg["dataset"]
    snapshot  = ds["snapshot"]
    timeframe = ds["timeframe"]
    syms      = tuple(ds["symbols"])

    # model/cv/run
    model     = mlcfg.get("model", "rf")
    cvcfg     = mlcfg.get("cv", {})
    n_splits  = int(cvcfg.get("n_splits", 5))
    purge     = int(cvcfg.get("purge", 0))
    embargo   = int(cvcfg.get("embargo", 0))
    runs_root = Path(mlcfg.get("runs_root", "runs"))
    extra     = mlcfg.get("extra_params", {})

    # resolve features/labels via content-only params
    feat_cfg_path = Path(mlcfg["features_cfg"])
    lbl_cfg_path  = Path(mlcfg["labels_cfg"])
    feat_params   = _feat_params_from_yaml(feat_cfg_path)
    label_params  = _label_params_from_yaml(lbl_cfg_path)

    fpaths = RunPaths(snapshot, "features", syms, timeframe, params=feat_params, runs_root=Path("data/features"))
    lpaths = RunPaths(snapshot, "labels",   syms, timeframe, params=label_params, runs_root=Path("data/labels"))
    features_path = fpaths.features
    labels_path   = lpaths.labels

    # load data
    X, y, keys, label_col = load_xy(str(features_path), str(labels_path))

    # CV
    cv = PurgedKFold(n_splits=n_splits, purge=purge, embargo=embargo)
    folds = []
    for tr, te in cv.split(X):
        m = SKLearnClassifier.make(kind=model).fit(X.iloc[tr], y.iloc[tr])
        scr = m.predict_score(X.iloc[te])
        folds.append(cls_metrics(y.iloc[te], scr))
    cv_report = {f"fold_{i}": s for i, s in enumerate(folds)}
    if folds:
        cv_report["mean_acc"] = float(sum(s["acc"] for s in folds)/len(folds))
        cv_report["mean_f1"]  = float(sum(s["f1"]  for s in folds)/len(folds))

    # final fit
    clf = SKLearnClassifier.make(kind=model).fit(X, y)

    # save run
    paths = RunPaths(snapshot=snapshot, strategy=f"ml_{model}", symbols=syms, timeframe=timeframe,
                     params={"label": label_col, "feat": feat_params, "lbl": label_params, **extra},
                     runs_root=runs_root)
    paths.ensure()
    model_path = paths.base / "model.joblib"
    clf.save(str(model_path))

    meta = {
        "snapshot": snapshot, "timeframe": timeframe, "symbols": list(syms), "universe": paths.universe,
        "params": {"label_col": label_col, "cv": {"n_splits": n_splits, "purge": purge, "embargo": embargo},
                   "feat": feat_params, "lbl": label_params, **extra},
        "features_path": str(features_path), "labels_path": str(labels_path),
        "model": model, "model_path": str(model_path),
        "cv_report": cv_report, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_manifest(paths, meta)
    typer.echo(f"✅ Trained {model} → {model_path}")

@app.command("predict")
def predict(cfg: Path = typer.Option(..., "--config", exists=True, dir_okay=False)):
    pcfg = load_cfg(cfg)
    manifest_path = Path(pcfg["manifest"])
    if not manifest_path.exists():
        raise typer.BadParameter(f"Manifest not found: {manifest_path}")

    model_dir = manifest_path.parent                           # runs/.../p-XXXX/

    model_bin = model_dir / "model.joblib"
    if not model_bin.exists():
        # helpful error: show what p-* dirs exist
        base = model_dir.parent
        existing = [p.name for p in base.iterdir() if p.is_dir()]
        raise typer.BadParameter(f"Model not found: {model_bin}\nAvailable under {base} -> {existing}")

    # read manifest to get data paths & metadata
    man = json.loads(manifest_path.read().decode() if hasattr(manifest_path, "read") else manifest_path.read_text())
    snapshot  = man["n"] if "n" in man else man.get("snapshot")
    if snapshot is None:
        snapshot = (model_dir.parent.parent.parent).name  # fallback if older manifest
    timeframe = man.get("timeframe") or (model_dir.parent.name)
    syms      = tuple(man.get("symbols") or [model_dir.parent.name])  # fallback: single symbol from dir name
    model     = man.get("model") or man.get("model_kind")

    features_path = Path(man.get("features_path") or pcfg.get("features_path",""))
    if not features_path.exists():
        raise typer.BadParameter(f"Features file not found: {features_path}")

    threshold     = float(pcfg.get("threshold", 0.5))
    runs_root_out = Path(pcfg.get("runs_root_out", "runs"))

    # score
    Xdf = pd.read_parquet(features_path)
    keys = Xdf[["timestamp","symbol"]]
    X = Xdf.drop(columns=["timestamp","symbol"])
    clf = SKLearnClassifier.load(str(model_bin))
    score = clf.predict_score(X)

    import numpy as np
    signals = keys.copy()
    signals["signal"] = (score >= threshold).astype(float)
    signals["score"]  = score

    out_paths = RunCaller = RunPaths(
        snapshot=snapshot, strategy="ml_signals", symbols=syms, timeframe=timeframe,
        params={"src": f"ml_{model}", "thr": threshold}, runs_root=runs_root_out
    )
    out_paths.ensure()
    signals.to_parquet(out_paths.signals, index=False)

    feat_panel = features_path.with_name("panel.parquet")
    if feat_panel.exists():
        panel = pd.read_parquet(feat_panel)
        panel.merge(signals, on=["timestamp","symbol"], how="inner").to_parquet(out_paths.panel, index=False)

    meta = {
        "model_used": str(model_bin), "features_used": str(features_path),
        "signals_path": str(out_paths.signals), "panel_path": str(out_paths.panel),
        "threshold": threshold, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (getattr(out_paths, "manifest", out_paths.report_dir / "metadata.json")
     ).write_text(json.dumps(meta, indent=2, sort_keys=True))
    typer.echo(f"✅ Wrote signals → {out_paths.signals}")
