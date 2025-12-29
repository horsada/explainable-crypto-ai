# src/excrypto/baselines/cli.py
from __future__ import annotations

from pathlib import Path
import typer

from excrypto.baseline.signals import momentum_signals, hodl_signals
from excrypto.baseline.writer import write_baseline_artifact
from excrypto.utils.loader import load_snapshot
from excrypto.utils.paths import RunPaths

app = typer.Typer(help="Baselines: generate signals + manifest")


@app.command("momentum")
def momentum_cli(
    snapshot: str = typer.Option(...),
    symbols: str = typer.Option("BTC/USDT,ETH/USDT"),
    exchange: str = typer.Option("binance"),
    timeframe: str = typer.Option("1h"),
    runs_root: str = typer.Option("runs"),
    fast: int = 20,
    slow: int = 60,
):
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    panel = load_snapshot(snapshot, syms, exchange=exchange, timeframe=timeframe)
    signals = momentum_signals(panel, fast=fast, slow=slow)

    runpaths = RunPaths(
        snapshot=snapshot,
        strategy="momentum",
        symbols=tuple(syms),
        timeframe=timeframe,
        params={"exchange": exchange, "fast": fast, "slow": slow},
        runs_root=Path(runs_root),
    )

    artifact = write_baseline_artifact(
        runpaths,
        signals,
        inputs={"exchange": exchange},
    )
    typer.echo(f"Wrote {artifact.signals_path}")
    typer.echo(f"Wrote {artifact.manifest_path}")


@app.command("hodl")
def hodl_cli(
    snapshot: str = typer.Option(...),
    symbols: str = typer.Option("BTC/USDT,ETH/USDT"),
    exchange: str = typer.Option("binance"),
    timeframe: str = typer.Option("1h"),
    runs_root: str = typer.Option("runs"),
):
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    panel = load_snapshot(snapshot, syms, exchange=exchange, timeframe=timeframe)
    signals = hodl_signals(panel)

    runpaths = RunPaths(
        snapshot=snapshot,
        strategy="hodl",
        symbols=tuple(syms),
        timeframe=timeframe,
        params={"exchange": exchange},
        runs_root=Path(runs_root),
    )

    artifact = write_baseline_artifact(
        runpaths,
        signals,
        inputs={"exchange": exchange},
    )
    typer.echo(f"Wrote {artifact.signals_path}")
    typer.echo(f"Wrote {artifact.manifest_path}")
