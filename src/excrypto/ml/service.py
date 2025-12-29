# src/excrypto/ml/service.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from excrypto.ml.datasets import load_xy
from excrypto.ml.evaluate import cls_metrics
from excrypto.ml.models_sklearn import SKLearnClassifier
from excrypto.ml.resolve import read_latest_pointer, load_manifest, write_latest_pointer
from excrypto.ml.splitters import PurgedKFold
from excrypto.utils.config import load_cfg, cfg_hash
from excrypto.utils.paths import RunPaths


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _abs_from_runs_root(runs_root: Path, p: str | Path) -> Path:
    pp = Path(p)

    if pp.is_absolute():
        return pp

    # already relative to runs_root → do NOT prefix again
    if pp.parts and pp.parts[0] == runs_root.name:
        return pp.resolve()

    return (runs_root / pp).resolve()


def _universe_for(snapshot: str, stage: str, symbols: list[str], timeframe: str, runs_root: Path) -> str:
    return RunPaths(
        snapshot=snapshot,
        strategy=stage,
        symbols=tuple(symbols),
        timeframe=timeframe,
        params=None,
        runs_root=runs_root,
    ).universe


@dataclass(frozen=True)
class TrainResult:
    model_path: Path
    manifest_path: Path
    metrics_path: Path


def train_model(
    *,
    snapshot: str,
    symbols: list[str],
    exchange: str,
    timeframe: str,
    runs_root: Path,
    config: Path | None,
    features_manifest: Path | None,
    labels_manifest: Path | None,
) -> TrainResult:
    # Resolve latest pointers per (timeframe, universe) like Features
    universe = _universe_for(snapshot, "features", symbols, timeframe, runs_root)

    feat_man_path = features_manifest or read_latest_pointer(
        runs_root, snapshot, "features", timeframe=timeframe, universe=universe
    )

    lbl_man_path = labels_manifest or read_latest_pointer(
        runs_root, snapshot, "labels", timeframe=timeframe, universe=universe
    )

    feat_man = load_manifest(feat_man_path)
    lbl_man = load_manifest(lbl_man_path)

    features_path = _abs_from_runs_root(runs_root, feat_man["paths"]["features"])
    labels_path = _abs_from_runs_root(runs_root, lbl_man["paths"]["labels"])
    label_col = lbl_man["label"]["label_col"]

    train_cfg: dict[str, Any] = {}
    train_cfg_hash = "p-default"
    if config is not None:
        train_cfg = load_cfg(config)
        train_cfg_hash = cfg_hash(train_cfg)

    xy = load_xy(str(features_path), str(labels_path), label_col=label_col)
    X, y, used_label_col = xy.X, xy.y, xy.label_col

    split_cfg = train_cfg.get("split", {})
    splitter = PurgedKFold(**split_cfg) if isinstance(split_cfg, dict) else PurgedKFold()

    model_spec = train_cfg.get("model", "rf")
    if isinstance(model_spec, str):
        clf = SKLearnClassifier.make(model_spec)
    elif isinstance(model_spec, dict):
        kind = str(model_spec.get("name", model_spec.get("kind", "rf")))
        params = model_spec.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError(f"model.params must be a dict, got {type(params)}")
        clf = SKLearnClassifier.make(kind, **params)
    else:
        raise ValueError(f"Invalid model spec type for 'model': {type(model_spec)}")

    threshold = float(train_cfg.get("threshold", 0.5))
    fold_metrics: list[dict[str, Any]] = []
    for _, (tr_idx, te_idx) in enumerate(splitter.split(X)):
        clf.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        score = clf.predict_score(X.iloc[te_idx])
        pred = (score >= threshold).astype(int)
        fold_metrics.append(cls_metrics(y.iloc[te_idx].to_numpy(), pred, score))

    clf.fit(X, y)

    out_paths = RunPaths(
        snapshot=snapshot,
        strategy="ml",
        symbols=tuple(symbols),
        timeframe=timeframe,
        params={"exchange": exchange, "cfg": train_cfg_hash},
        runs_root=runs_root,
    )
    out_paths.ensure(report=False)

    model_bin = out_paths.base / "model.joblib"
    metrics_path = out_paths.base / "metrics.json"
    clf.save(str(model_bin))
    metrics_path.write_text(json.dumps({"folds": fold_metrics}, indent=2, sort_keys=True))

    manifest = {
        "kind": "ml_model",
        "schema_version": 1,
        "created_at": _now_utc(),
        "snapshot": snapshot,
        "timeframe": timeframe,
        "symbols": symbols,
        "exchange": exchange,
        "inputs": {
            "features_manifest": str(feat_man_path),
            "labels_manifest": str(lbl_man_path),
            "features_path": str(features_path),
            "labels_path": str(labels_path),
            "label_col": used_label_col,
        },
        "train": {"threshold": threshold, "train_cfg_hash": train_cfg_hash, "train_cfg": train_cfg},
        "paths": {"model": str(model_bin), "metrics": str(metrics_path), "manifest": str(out_paths.manifest)},
    }
    out_paths.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    # Write latest pointer per (timeframe, universe)
    write_latest_pointer(
        out_paths.runs_root,
        out_paths.snapshot,
        out_paths.strategy,
        out_paths.manifest,
        timeframe=out_paths.timeframe,
        universe=out_paths.universe,
    )

    return TrainResult(model_path=model_bin, manifest_path=out_paths.manifest, metrics_path=metrics_path)


@dataclass(frozen=True)
class PredictResult:
    signals_path: Path
    manifest_path: Path


def predict_signals(
    *,
    snapshot: str,
    symbols: list[str],
    exchange: str,
    timeframe: str,
    runs_root: Path,
    manifest: Path | None,
    threshold: float,
) -> PredictResult:
    universe = _universe_for(snapshot, "ml", symbols, timeframe, runs_root)

    ml_man_path = manifest or read_latest_pointer(
        runs_root, snapshot, "ml", timeframe=timeframe, universe=universe
    )

    ml_man = load_manifest(ml_man_path)

    model_bin = _abs_from_runs_root(runs_root, ml_man["paths"]["model"])
    features_path = _abs_from_runs_root(runs_root, ml_man["inputs"]["features_path"])

    Xdf = pd.read_parquet(features_path)
    key_cols = ["timestamp", "symbol"]
    missing = [c for c in key_cols if c not in Xdf.columns]
    if missing:
        raise ValueError(f"Features missing key cols required for signals: {missing}")

    keys = Xdf[key_cols].copy()
    X = Xdf.drop(columns=key_cols, errors="ignore")

    clf = SKLearnClassifier.load(str(model_bin))
    score = clf.predict_score(X)

    signals = keys.copy()
    signals["score"] = score
    signals["signal"] = (score >= float(threshold)).astype(float)

    out_paths = RunPaths(
        snapshot=snapshot,
        strategy="predict",
        symbols=tuple(symbols),
        timeframe=timeframe,
        params={"exchange": exchange, "thr": float(threshold), "src": "ml"},
        runs_root=runs_root,
    )
    out_paths.ensure(report=True)

    signals.to_parquet(out_paths.signals, index=False)

    # Optional panel (kept for now)
    panel = signals.copy()
    if "close" in Xdf.columns:
        panel["close"] = Xdf["close"].values
    panel.to_parquet(out_paths.panel, index=False)

    pred_manifest = {
        "kind": "ml_predict",
        "schema_version": 1,
        "created_at": _now_utc(),
        "snapshot": snapshot,
        "timeframe": timeframe,
        "symbols": symbols,
        "exchange": exchange,
        "inputs": {"ml_manifest": str(ml_man_path), "model": str(model_bin), "features": str(features_path)},
        "params": {"threshold": float(threshold)},
        "paths": {"signals": str(out_paths.signals), "panel": str(out_paths.panel), "manifest": str(out_paths.manifest)},
    }
    out_paths.manifest.write_text(json.dumps(pred_manifest, indent=2, sort_keys=True))

    # Write latest pointer per (timeframe, universe)
    write_latest_pointer(
        out_paths.runs_root,
        out_paths.snapshot,
        out_paths.strategy,
        out_paths.manifest,
        timeframe=out_paths.timeframe,
        universe=out_paths.universe,
    )

    return PredictResult(signals_path=out_paths.signals, manifest_path=out_paths.manifest)
