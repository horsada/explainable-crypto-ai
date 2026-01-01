# src/excrypto/data/panel.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from excrypto.utils.loader import load_snapshot
from excrypto.utils.paths import RunPaths
from excrypto.ml.resolve import write_latest_pointer


@dataclass(frozen=True)
class PanelArtifact:
    panel_path: Path
    manifest_path: Path


def build_and_write_panel(
    *,
    snapshot: str,
    symbols: list[str],
    exchange: str,
    timeframe: str,
    runs_root: Path,
) -> PanelArtifact:
    # load_snapshot gives you the standard panel used by features/labels today
    panel = load_snapshot(snapshot, symbols, exchange=exchange, timeframe=timeframe).sort_index()

    # write to runs/…/snapshot/…/panel.parquet (single canonical panel artifact)
    paths = RunPaths(
        snapshot=snapshot,
        strategy="snapshot",
        symbols=tuple(symbols),
        timeframe=timeframe,
        params={"exchange": exchange},
        runs_root=runs_root,
    )
    paths.ensure(report=False)

    out = panel.reset_index()  # timestamp becomes column
    out.to_parquet(paths.panel, index=False)

    meta = {
        "kind": "snapshot_panel",
        "schema_version": 1,
        "snapshot": snapshot,
        "exchange": exchange,
        "timeframe": timeframe,
        "symbols": symbols,
        "rows": int(out.shape[0]),
        "columns": list(out.columns),
        "paths": {"panel": str(paths.panel), "manifest": str(paths.manifest)},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    paths.manifest.write_text(json.dumps(meta, indent=2, sort_keys=True))

    # stable pointer per (timeframe, universe) so other stages resolve without recomputing params
    write_latest_pointer(
        paths.runs_root,
        paths.snapshot,
        paths.strategy,
        paths.manifest,
        timeframe=paths.timeframe,
        universe=paths.universe,
    )

    return PanelArtifact(panel_path=paths.panel, manifest_path=paths.manifest)
