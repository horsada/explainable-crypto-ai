# src/excrypto/features/cli.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from excrypto.utils.config import cfg_hash, load_cfg
from excrypto.utils.paths import RunPaths
from excrypto.features.builder import build_and_write_features

app = typer.Typer(add_completion=False)


def _parse_symbols(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _default_specs() -> list[dict[str, Any]]:
    # Keep defaults here (or move to a YAML later)
    return [
        {"name": "log_returns", "input_cols": ["close"], "output_col": "ret_log"},
        {
            "name": "rolling_volatility",
            "input_cols": ["ret_log"],
            "output_col": "vol_30",
            "params": {"window": 30},
        },
        {"name": "rsi", "input_cols": ["close"], "output_col": "rsi_14", "params": {"window": 14}},
    ]


@app.command("build")
def build(
    snapshot: str = typer.Option(..., help="Snapshot id (folder name under runs/)."),
    symbols: str = typer.Option(..., help="Comma-separated symbols, e.g. BTC/USDT,ETH/USDT"),
    exchange: str = typer.Option("binance", help="Exchange name (used for metadata)."),
    timeframe: str = typer.Option("1h", help="Candle timeframe, e.g. 1m, 5m, 1h."),
    config: Path | None = typer.Option(None, exists=True, dir_okay=False, help="YAML/JSON feature spec config."),
    runs_root: Path = typer.Option(Path("runs"), help="Artifact root directory."),
    nan_policy: str = typer.Option("keep", help="NaN handling: keep | drop_any"),
) -> None:
    """
    Build features for a snapshot + symbol universe.
    Thin CLI wrapper: load -> call builder -> print paths.
    """
    syms = _parse_symbols(symbols)

    # Load input panel from snapshot stage (this mirrors your current pattern).
    # If your snapshot artifact path differs, adjust this line.
    snap_paths = RunPaths(
        snapshot=snapshot,
        strategy="snapshot",
        symbols=tuple(syms),
        timeframe=timeframe,
        params={"exchange": exchange},
        runs_root=runs_root,
    )
    panel_path = snap_paths.panel
    if not panel_path.exists():
        raise typer.BadParameter(f"Snapshot panel not found: {panel_path}")

    panel = pd.read_parquet(panel_path)

    specs: list[dict[str, Any]]
    if config is None:
        specs = _default_specs()
        specs_hash = cfg_hash({"specs": specs})
    else:
        cfg = load_cfg(config)
        specs = cfg.get("specs", _default_specs())
        specs_hash = cfg_hash(cfg)

    feat_paths = RunPaths(
        snapshot=snapshot,
        strategy="features",
        symbols=tuple(syms),
        timeframe=timeframe,
        params={"exchange": exchange, "spec_hash": specs_hash},
        runs_root=runs_root,
    )

    artifact = build_and_write_features(
        panel=panel,
        specs=specs,
        runpaths=feat_paths,
        group_col="symbol",
        nan_policy="drop_any" if nan_policy == "drop_any" else "keep",
        extra_manifest={
            "exchange": exchange,
            "input_panel": str(panel_path),
        },
    )

    typer.echo(json.dumps(
        {
            "features": str(artifact.features_path),
            "panel": str(artifact.panel_path),
            "manifest": str(artifact.manifest_path),
            "n_features": artifact.n_features,
            "specs_hash": artifact.specs_hash,
        },
        indent=2,
    ))
