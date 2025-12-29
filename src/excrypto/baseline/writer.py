# src/excrypto/baselines/writer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import pandas as pd

from excrypto.utils.paths import RunPaths


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


@dataclass(frozen=True)
class BaselineArtifact:
    signals_path: Path
    manifest_path: Path
    latest_path: Path
    n_rows: int


def write_latest_manifest(runpaths: RunPaths) -> Path:
    """
    Writes: runs/<snapshot>/<strategy>/<tf>/<universe>/latest_manifest.json
    """
    sym_dir = runpaths.base.parent  # .../<universe>/
    latest = sym_dir / "latest_manifest.json"
    payload = {
        "kind": "latest_pointer",
        "schema_version": 1,
        "paths": {"manifest": str(runpaths.manifest)},
        "p_hash": runpaths.base.name,
    }
    _atomic_write_json(latest, payload)
    return latest


def write_baseline_artifact(
    runpaths: RunPaths,
    signals: pd.DataFrame,
    *,
    inputs: dict[str, Any],
    extra_manifest: dict[str, Any] | None = None,
) -> BaselineArtifact:
    runpaths.ensure(report=False)

    _atomic_write_parquet(signals, runpaths.signals)

    manifest: dict[str, Any] = {
        "kind": "baseline_signals",
        "schema_version": 1,
        "snapshot": runpaths.snapshot,
        "strategy": runpaths.strategy,
        "timeframe": runpaths.timeframe,
        "symbols": list(runpaths.symbols),
        "universe": runpaths.universe,
        "params": runpaths.params,
        "inputs": inputs,
        "paths": {
            "signals": str(runpaths.signals),
            "manifest": str(runpaths.manifest),
        },
        "rows": {"signals_rows": int(signals.shape[0])},
        "cols": {"cols": list(signals.columns)},
    }
    if extra_manifest:
        manifest.update(extra_manifest)

    _atomic_write_json(runpaths.manifest, manifest)
    latest_path = write_latest_manifest(runpaths)

    return BaselineArtifact(
        signals_path=runpaths.signals,
        manifest_path=runpaths.manifest,
        latest_path=latest_path,
        n_rows=int(signals.shape[0]),
    )
