# src/excrypto/ml/cli.py
from __future__ import annotations

import json
from pathlib import Path

import typer

from excrypto.ml.service import train_model, predict_signals

app = typer.Typer(help="ML: train/predict (thin CLI wrapper)")


def _parse_symbols(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


@app.command("train")
def train(
    snapshot: str = typer.Option(...),
    symbols: str = typer.Option(..., help="Comma-separated symbols"),
    exchange: str = typer.Option("binance"),
    timeframe: str = typer.Option("1h"),
    runs_root: Path = typer.Option(Path("runs")),
    config: Path | None = typer.Option(None, exists=True, dir_okay=False, help="Train config (yaml/json)."),
    features_manifest: Path | None = typer.Option(None, exists=True, dir_okay=False, help="Override features manifest."),
    labels_manifest: Path | None = typer.Option(None, exists=True, dir_okay=False, help="Override labels manifest."),
) -> None:
    syms = _parse_symbols(symbols)
    res = train_model(
        snapshot=snapshot,
        symbols=syms,
        exchange=exchange,
        timeframe=timeframe,
        runs_root=runs_root,
        config=config,
        features_manifest=features_manifest,
        labels_manifest=labels_manifest,
    )
    typer.echo(json.dumps(
        {
            "model": str(res.model_path),
            "metrics": str(res.metrics_path),
            "manifest": str(res.manifest_path),
        },
        indent=2,
    ))


@app.command("predict")
def predict(
    snapshot: str = typer.Option(...),
    symbols: str = typer.Option(..., help="Comma-separated symbols"),
    exchange: str = typer.Option("binance"),
    timeframe: str = typer.Option("1h"),
    runs_root: Path = typer.Option(Path("runs")),
    manifest: Path | None = typer.Option(None, exists=True, dir_okay=False, help="Override ML manifest."),
    threshold: float = typer.Option(0.5, help="Decision threshold on score."),
) -> None:
    syms = _parse_symbols(symbols)
    res = predict_signals(
        snapshot=snapshot,
        symbols=syms,
        exchange=exchange,
        timeframe=timeframe,
        runs_root=runs_root,
        manifest=manifest,
        threshold=threshold,
    )
    typer.echo(json.dumps(
        {
            "signals": str(res.signals_path),
            "manifest": str(res.manifest_path),
        },
        indent=2,
    ))
