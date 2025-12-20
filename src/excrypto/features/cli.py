# src/excrypto/features/cli.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import typer

from excrypto.utils.loader import load_snapshot
from excrypto.utils.config import load_cfg, cfg_hash
from excrypto.utils.paths import RunPaths
from excrypto.features import FeaturePipeline

app = typer.Typer(help="Features: generate feature panels from registry snapshots")

def _parse_symbols(symbols: str) -> list[str]:
    return [s.strip() for s in symbols.split(",") if s.strip()]

def _default_specs() -> List[Dict[str, Any]]:
    return [
        {"name": "log_returns", "input_cols": ["close"], "output_col": "ret_log"},
        {"name": "rolling_volatility", "input_cols": ["ret_log"], "output_col": "vol_30", "params": {"window": 30}},
        {"name": "rsi", "input_cols": ["close"], "output_col": "rsi_14", "params": {"window": 14}},
    ]

def _load_specs(specs_json: Optional[str]) -> List[Dict[str, Any]]:
    if not specs_json:
        return _default_specs()
    try:
        obj = json.loads(specs_json)
        if not isinstance(obj, list):
            raise ValueError("Specs JSON must be a list of dicts.")
        return obj
    except Exception as e:
        raise typer.BadParameter(f"Invalid --specs JSON: {e}")

import json, time

def _write_outputs(paths: RunPaths, panel: pd.DataFrame, features: pd.DataFrame, write_panel: bool) -> None:
    paths.ensure(report=False)

    # write features + optional merged panel
    features.to_parquet(paths.features, index=False)
    if write_panel:
        base = panel.reset_index()[["timestamp","symbol","close"]]
        merged = base.merge(features, on=["timestamp","symbol"], how="inner")
        merged.to_parquet(paths.panel, index=False)

    # write metadata
    ts_min = str(panel.index.min())
    ts_max = str(panel.index.max())
    meta = {
        "snapshot": paths.snapshot,
        "namespace": paths.strategy,           # "features"
        "timeframe": paths.timeframe,
        "universe": paths.universe,
        "params": paths.params or {},
        "rows": int(len(features)),
        "symbols": sorted(features["symbol"].unique().tolist()),
        "timestamps": {"start": ts_min, "end": ts_max},
        "columns": [c for c in features.columns if c not in ("timestamp","symbol")],
        "artifacts": {
            "features_parquet": str(paths.features),
            "panel_parquet": str(paths.panel) if write_panel else None,
            "report_dir": str(paths.report_dir),
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    paths.manifest.write_text(json.dumps(meta, indent=2, sort_keys=True))
    typer.echo(f"Wrote {paths.features.name}" + (f", {paths.panel.name}" if write_panel else "") + f", and {paths.manifest.name}")


@app.command("build")
def build_features(
    snapshot: str = typer.Option(..., help="registry snapshot_id"),
    symbols: str = typer.Option("BTC/USDT,ETH/USDT", help="CSV list of symbols"),
    exchange: str = typer.Option("binance"),
    timeframe: str = typer.Option("1h", help="Timesteps (e.g., 1h, 15m)"),
    cfg: Optional[Path] = typer.Option(None, "--config", help="YAML with 'specs' list"),
    specs: Optional[str] = typer.Option(None, "--specs", "-s", help="JSON list of feature specs"),
    write_panel: bool = typer.Option(True, help="Also write merged panel with close + features"),
):
    """
    Create features per symbol and save under data/features/{snapshot}/features/{timeframe}/...
    """
    syms = _parse_symbols(symbols)
    panel = load_snapshot(snapshot, syms, exchange=exchange, timeframe=timeframe).sort_index()  # index=timestamp, cols include symbol, close

    # load specs from YAML or JSON or defaults
    if cfg:
        feat_cfg   = load_cfg(cfg)
        feat_specs = feat_cfg["specs"]
        feat_params = {"hash": cfg_hash({"specs": feat_specs})}   # <— only content
    elif specs:
        feat_specs = _load_specs(specs)
        feat_params = {"hash": cfg_hash({"specs": feat_specs})}
    else:
        feat_specs = _default_specs()
        feat_params = {"hash": cfg_hash({"specs": feat_specs})}

    pipe = FeaturePipeline(feat_specs).fit(panel)

    out_frames: list[pd.DataFrame] = []
    for sym, g in panel.groupby("symbol"):
        X = pipe.transform(g)
        X = X.assign(timestamp=g.index, symbol=sym)
        out_frames.append(X.reset_index(drop=True))

    features = pd.concat(out_frames, axis=0, ignore_index=True).sort_values(["timestamp", "symbol"])

    # Save using RunPaths with runs_root overridden to data/features
    paths = RunPaths(
        snapshot=snapshot,
        strategy="features",                 # namespace, not a trading strategy
        symbols=tuple(syms),
        timeframe=timeframe,
        params=feat_params,
        runs_root=Path("data/features"),     # <<< root redirected here
    )
    _write_outputs(paths, panel, features, write_panel)
