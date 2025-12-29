# src/excrypto/labels/cli.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from excrypto.utils.config import cfg_hash, load_cfg
from excrypto.utils.paths import RunPaths
from excrypto.labels.builder import build_and_write_labels, canonical_label_params

app = typer.Typer(add_completion=False)


def _parse_symbols(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


@app.command("build")
def build(
    snapshot: str = typer.Option(..., help="Snapshot id (folder name under runs/)."),
    symbols: str = typer.Option(..., help="Comma-separated symbols, e.g. BTC/USDT,ETH/USDT"),
    exchange: str = typer.Option("binance", help="Exchange name (used for metadata)."),
    timeframe: str = typer.Option("1h", help="Candle timeframe, e.g. 1m, 5m, 1h."),
    kind: str = typer.Option("fixed_horizon_return", help="Label kind: fixed_horizon_return | triple_barrier"),
    config: Path | None = typer.Option(None, exists=True, dir_okay=False, help="YAML/JSON label config."),
    runs_root: Path = typer.Option(Path("runs"), help="Artifact root directory."),
    nan_policy: str = typer.Option("keep", help="NaN handling: keep | drop_any"),
    # common FH params (also usable as overrides)
    horizon: int = typer.Option(24, help="Forward horizon in bars."),
    thr: float = typer.Option(0.0, help="FH classification threshold (log-return)."),
    as_class: bool = typer.Option(True, help="FH: classification vs regression."),
    # TB params (also usable as overrides)
    up_mult: float = typer.Option(2.0, help="TB: upper barrier multiplier."),
    dn_mult: float = typer.Option(2.0, help="TB: lower barrier multiplier."),
    vol_window: int = typer.Option(50, help="TB: rolling vol window."),
) -> None:
    """
    Build labels for a snapshot + symbol universe.
    Thin CLI wrapper: load -> call builder -> print paths.
    """
    syms = _parse_symbols(symbols)

    # Load snapshot panel (adjust if your snapshot stage uses a different strategy key)
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

    # Load config + apply CLI overrides
    base_params: dict[str, Any] = {}
    cfg_hash_val: str

    if config is not None:
        cfg = load_cfg(config)
        base_params = dict(cfg) if isinstance(cfg, dict) else {}
        cfg_hash_val = cfg_hash(cfg)
    else:
        cfg_hash_val = cfg_hash({"kind": kind})

    # Merge in explicit overrides (kept simple)
    overrides: dict[str, Any] = {"horizon": horizon, "as_class": as_class, "thr": thr,
                                "up_mult": up_mult, "dn_mult": dn_mult, "vol_window": vol_window}
    # Only keep non-None overrides
    merged = {**base_params, **{k: v for k, v in overrides.items() if v is not None}}

    canon = canonical_label_params(kind if config is None else merged.get("kind", kind), merged)

    lbl_paths = RunPaths(
        snapshot=snapshot,
        strategy="labels",
        symbols=tuple(syms),
        timeframe=timeframe,
        params={"exchange": exchange, "label_cfg": cfg_hash_val},
        runs_root=runs_root,
    )

    artifact = build_and_write_labels(
        panel=panel,
        canon=canon,
        runpaths=lbl_paths,
        nan_policy="drop_any" if nan_policy == "drop_any" else "keep",
        extra_manifest={"exchange": exchange, "input_panel": str(panel_path)},
    )

    typer.echo(
        json.dumps(
            {
                "labels": str(artifact.labels_path),
                "panel": str(artifact.panel_path),
                "manifest": str(artifact.manifest_path),
                "label_col": artifact.label_col,
                "params_hash": artifact.params_hash,
            },
            indent=2,
        )
    )