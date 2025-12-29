# src/excrypto/agents/orchestrator.py
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n  {' '.join(cmd)}\n\nstdout:\n{p.stdout}\n\nstderr:\n{p.stderr}"
        )


def _req(d: dict[str, Any], k: str) -> Any:
    if k not in d:
        raise ValueError(f"Missing required key: {k}")
    return d[k]


@dataclass(frozen=True)
class Plan:
    start: str
    end: str
    snapshot: str                 # canonical snapshot id used by RunPaths + stages
    symbols: list[str]
    exchange: str
    timeframe: str
    runs_root: Path
    nan_policy: str
    features_config: Path | None
    labels_config: Path | None
    labels_kind: str
    ml_config: Path | None
    ml_threshold: float


def _load_plan(config_path: Path) -> Plan:
    cfg = yaml.safe_load(config_path.read_text()) or {}

    dataset = _req(cfg, "dataset")
    start = str(_req(dataset, "start"))
    end = str(_req(dataset, "end"))

    # Canonical snapshot id: matches data/snapshot raw folder naming
    snapshot_id = f"{start}_to_{end}"

    symbols = _req(dataset, "symbols")
    if not isinstance(symbols, list) or not all(isinstance(s, str) for s in symbols):
        raise ValueError("dataset.symbols must be a list[str]")

    exchange = str(dataset.get("exchange", "binance"))
    timeframe = str(dataset.get("timeframe", "1h"))

    runs_root = Path(cfg.get("runs_root", "runs"))
    nan_policy = str(cfg.get("nan_policy", "drop_any"))

    features_config = cfg.get("features", {}).get("config")
    labels_section = cfg.get("labels", {})
    labels_config = labels_section.get("config")
    labels_kind = str(labels_section.get("kind", "fixed_horizon_return"))

    ml_section = cfg.get("ml", {})
    ml_config = ml_section.get("config")
    ml_threshold = float(ml_section.get("threshold", 0.5))

    return Plan(
        start=start,
        end=end,
        snapshot=snapshot_id,
        symbols=symbols,
        exchange=exchange,
        timeframe=timeframe,
        runs_root=runs_root,
        nan_policy=nan_policy,
        features_config=Path(features_config) if features_config else None,
        labels_config=Path(labels_config) if labels_config else None,
        labels_kind=labels_kind,
        ml_config=Path(ml_config) if ml_config else None,
        ml_threshold=ml_threshold,
    )


def run_plan(config_path: Path) -> None:
    plan = _load_plan(config_path)
    sym_csv = ",".join(plan.symbols)

    # 1) raw snapshot download
    _run([
        "excrypto", "data", "snapshot",
        "--start", plan.start,
        "--end", plan.end,
        "--exchange", plan.exchange,
        "--symbols", sym_csv,
        "--timeframe", plan.timeframe,
    ])

    # 1b) build panel artifact in runs/ for downstream stages
    _run([
        "excrypto", "data", "panel",
        "--snapshot", plan.snapshot,
        "--exchange", plan.exchange,
        "--symbols", sym_csv,
        "--timeframe", plan.timeframe,
        "--runs-root", str(plan.runs_root),
    ])

    # 2) baselines (generate signals only)
    _run([
        "excrypto", "baseline", "momentum",
        "--snapshot", plan.snapshot,
        "--symbols", sym_csv,
        "--exchange", plan.exchange,
        "--timeframe", plan.timeframe,
        "--fast", "20",
        "--slow", "60",
    ])

    _run([
        "excrypto", "baseline", "hodl",
        "--snapshot", plan.snapshot,
        "--symbols", sym_csv,
        "--exchange", plan.exchange,
        "--timeframe", plan.timeframe,
    ])


    # 3) features
    feat_cmd = [
        "excrypto", "features", "build",
        "--snapshot", plan.snapshot,
        "--symbols", sym_csv,
        "--exchange", plan.exchange,
        "--timeframe", plan.timeframe,
        "--runs-root", str(plan.runs_root),
        "--nan-policy", plan.nan_policy,
    ]
    if plan.features_config:
        feat_cmd += ["--config", str(plan.features_config)]
    _run(feat_cmd)

    # 4) labels
    lbl_cmd = [
        "excrypto", "labels", "build",
        "--snapshot", plan.snapshot,
        "--symbols", sym_csv,
        "--exchange", plan.exchange,
        "--timeframe", plan.timeframe,
        "--runs-root", str(plan.runs_root),
        "--nan-policy", plan.nan_policy,
        "--kind", plan.labels_kind,
    ]
    if plan.labels_config:
        lbl_cmd += ["--config", str(plan.labels_config)]
    _run(lbl_cmd)

    # 5) ml train (uses latest pointers by default)
    train_cmd = [
        "excrypto", "ml", "train",
        "--snapshot", plan.snapshot,
        "--symbols", sym_csv,
        "--exchange", plan.exchange,
        "--timeframe", plan.timeframe,
        "--runs-root", str(plan.runs_root),
    ]
    if plan.ml_config:
        train_cmd += ["--config", str(plan.ml_config)]
    _run(train_cmd)

    # 6) ml predict (uses ml latest pointer by default)
    _run([
        "excrypto", "ml", "predict",
        "--snapshot", plan.snapshot,
        "--symbols", sym_csv,
        "--exchange", plan.exchange,
        "--timeframe", plan.timeframe,
        "--runs-root", str(plan.runs_root),
        "--threshold", str(plan.ml_threshold),
    ])

    # 7) backtests
    for strategy in ["predict", "momentum", "hodl"]:
        _run([
            "excrypto", "backtest", "run",
            plan.snapshot,
            sym_csv,
            "--exchange", plan.exchange,
            "--timeframe", plan.timeframe,
            "--signals-strategy", strategy,
            "--config", "configs/agents/full.yaml",
        ])


