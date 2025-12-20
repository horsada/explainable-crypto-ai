# src/excrypto/baselines/cli.py
from __future__ import annotations
import typer
import pandas as pd
from pathlib import Path
from excrypto.baseline import momentum, hodl, vt_hodl  # optional
from excrypto.utils.loader import load_snapshot
from excrypto.utils.paths import RunPaths

app = typer.Typer(help="Baselines: generate signals/panels")

def _write_outputs(paths: RunPaths, panel: pd.DataFrame, signals: pd.DataFrame, write_panel: bool):
    # if overriding, mirror the default structure under that base
    paths.ensure()
    signals_path = paths.signals
    panel_path   = paths.panel

    signals.to_parquet(signals_path, index=False)
    if write_panel:
        merged = (panel.reset_index()[["timestamp","symbol","close"]]
                  .merge(signals, on=["timestamp","symbol"], how="inner"))
        merged.to_parquet(panel_path, index=False)

    typer.echo(f"Wrote {signals_path}" + (f" and {panel_path.name}" if write_panel else ""))

@app.command("momentum")
def momentum_signals(
    snapshot: str = typer.Option(..., help="registry snapshot_id"),
    symbols: str = typer.Option("BTC/USDT,ETH/USDT", help="CSV list"),
    exchange: str = typer.Option("binance"),
    timeframe: str = typer.Option("1h", help="Timesteps"),
    fast: int = 20,
    slow: int = 60,
    write_panel: bool = True,
):
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    panel = load_snapshot(snapshot, syms, exchange=exchange, timeframe=timeframe)  # index=timestamp, has 'symbol','close'
    # signals per symbol (PIT-safe inside momentum)
    sigs = []
    for sym, g in panel.groupby("symbol"):
        s = momentum._sma_sig(g["close"], fast=fast, slow=slow).rename("signal")
        sigs.append(pd.DataFrame({"timestamp": s.index, "symbol": sym, "signal": s.values}))
    signals = pd.concat(sigs).sort_values(["timestamp","symbol"])
    paths = RunPaths(snapshot=snapshot, strategy="momentum", symbols=tuple(syms), timeframe=timeframe,
                     params={"fast": fast, "slow": slow})
    _write_outputs(paths, panel, signals, write_panel)

@app.command("hodl")
def hodl_signals(
    snapshot: str = typer.Option(..., help="registry snapshot_id"),
    symbols: str = typer.Option("BTC/USDT,ETH/USDT", help="CSV list"),
    exchange: str = typer.Option("binance"),
    timeframe: str = typer.Option("1h", help="Timesteps"),
    write_panel: bool = True,
):
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    panel = load_snapshot(snapshot, syms, exchange=exchange, timeframe=timeframe)
    sig = pd.DataFrame({"timestamp": panel.index, "symbol": panel["symbol"].values, "signal": 1.0})
    paths = RunPaths(snapshot=snapshot, strategy="hodl", symbols=tuple(syms), timeframe=timeframe, params=None)
    _write_outputs(paths, panel, sig, write_panel)
