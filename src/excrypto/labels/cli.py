# src/excrypto/labels/cli.py
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from excrypto.utils.loader import load_snapshot
from excrypto.utils.config import load_cfg, cfg_hash
from excrypto.utils.paths import RunPaths
from excrypto.labels.labelers import fixed_horizon_return, triple_barrier

app = typer.Typer(help="Labels: generate supervised learning labels from registry snapshots")

def _parse_symbols(symbols: str) -> list[str]:
    return [s.strip() for s in symbols.split(",") if s.strip()]

def _write_outputs(paths: RunPaths, panel: pd.DataFrame, labels: pd.DataFrame, write_panel: bool) -> None:
    paths.ensure(report=False)
    labels.to_parquet(paths.labels, index=False)
    if write_panel:
        base = panel.reset_index()[["timestamp","symbol","close"]]
        base.merge(labels, on=["timestamp","symbol"], how="inner").to_parquet(paths.panel, index=False)

    meta = {
        "snapshot": paths.snapshot,
        "namespace": paths.strategy,     # "labels"
        "timeframe": paths.timeframe,
        "universe": paths.universe,
        "params": paths.params or {},
        "rows": int(len(labels)),
        "symbols": sorted(labels["symbol"].unique().tolist()),
        "timestamps": {"start": str(panel.index.min()), "end": str(panel.index.max())},
        "columns": [c for c in labels.columns if c not in ("timestamp","symbol")],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifacts": {
            "labels_parquet": str(paths.labels),
            "panel_parquet": str(paths.panel) if write_panel else None,
            "report_dir": str(paths.report_dir),
        },
    }
    paths.manifest.write_text(json.dumps(meta, indent=2, sort_keys=True))
    typer.echo(f"Wrote {paths.labels.name}" + (f", {paths.panel.name}" if write_panel else "") + f", and {paths.manifest.name}")

@app.command("fh")
def build_fixed_horizon(
    snapshot: str = typer.Option(...),
    symbols: str = typer.Option("BTC/USDT,ETH/USDT"),
    timeframe: str = typer.Option("1h"),
    cfg: Optional[Path] = typer.Option(None, "--config", help="YAML with {kind: fh, h, thr, mode}"),
    horizon: int = typer.Option(24),
    thr: float = typer.Option(0.0),
    regression: bool = typer.Option(False, "--regression"),
    write_panel: bool = typer.Option(True),
):
    syms = _parse_symbols(symbols)
    panel = load_snapshot(snapshot, syms, timeframe).sort_index()

    if cfg:
        yml = load_cfg(cfg)
        assert yml.get("kind", "fh") == "fh"
        horizon = int(yml.get("h", horizon))
        thr = float(yml.get("thr", thr))
        mode = yml.get("mode", "cls")
    else:
        mode = "reg" if regression else "cls"

    canon = {"kind": "fh", "h": horizon, "thr": thr, "mode": mode}
    label_params = {"hash": cfg_hash(canon), **canon}


    frames = []
    for sym, g in panel.groupby("symbol"):
        s = fixed_horizon_return(g["close"], horizon=horizon, as_class=not regression, thr=thr)
        frames.append(pd.DataFrame({"timestamp": g.index, "symbol": sym, s.name: s.values}))
    labels = pd.concat(frames).sort_values(["timestamp","symbol"]).reset_index(drop=True)

    paths = RunPaths(snapshot=snapshot, strategy="labels", symbols=tuple(syms), timeframe=timeframe,
                     params=label_params, runs_root=Path("data/labels"))
    _write_outputs(paths, panel, labels, write_panel)

@app.command("tb")
def build_triple_barrier(
    snapshot: str = typer.Option(...),
    symbols: str = typer.Option("BTC/USDT,ETH/USDT"),
    timeframe: str = typer.Option("1h"),
    cfg: Optional[Path] = typer.Option(None, "--config", help="YAML with {kind: tb, h,u,d,w}"),
    horizon: int = typer.Option(24),
    up_mult: float = typer.Option(2.0),
    dn_mult: float = typer.Option(2.0),
    vol_window: int = typer.Option(50),
    write_panel: bool = typer.Option(True),
):
    syms = _parse_symbols(symbols)
    panel = load_snapshot(snapshot, syms, timeframe).sort_index()

    if cfg:
        yml = load_cfg(cfg)
        assert yml.get("kind","tb") == "tb"
        horizon   = int(yml.get("h", horizon))
        up_mult   = float(yml.get("u", up_mult))
        dn_mult   = float(yml.get("d", dn_mult))
        vol_window= int(yml.get("w", vol_window))
        label_params = {"cfg": yml.get("name","anon"), "hash": cfg_hash(yml),
                        "kind":"tb","h":horizon,"u":up_mult,"d":dn_mult,"w":vol_window}
    else:
        label_params = {"cfg":"flags","hash": cfg_hash({"h":horizon,"u":up_mult,"d":dn_mult,"w":vol_window}),
                        "kind":"tb","h":horizon,"u":up_mult,"d":dn_mult,"w":vol_window}

    frames = []
    for sym, g in panel.groupby("symbol"):
        s = triple_barrier(g["close"], horizon=horizon, up_mult=up_mult, dn_mult=dn_mult, vol_window=vol_window)
        frames.append(pd.DataFrame({"timestamp": g.index, "symbol": sym, s.name: s.values}))
    labels = pd.concat(frames).sort_values(["timestamp","symbol"]).reset_index(drop=True)

    paths = RunPaths(snapshot=snapshot, strategy="labels", symbols=tuple(syms), timeframe=timeframe,
                     params=label_params, runs_root=Path("data/labels"))
    _write_outputs(paths, panel, labels, write_panel)
