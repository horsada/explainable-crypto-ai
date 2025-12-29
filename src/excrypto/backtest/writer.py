from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import pandas as pd

from excrypto.ml.resolve import write_latest_pointer
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
class BacktestArtifact:
    backtest_path: Path
    summary_path: Path
    manifest_path: Path
    n_rows: int


def write_backtest_artifact(
    runpaths: RunPaths,
    bt: pd.DataFrame,
    *,
    summary: dict[str, Any],
    inputs: dict[str, Any],
    engine: dict[str, Any],
    extra_manifest: dict[str, Any] | None = None,
) -> BacktestArtifact:
    runpaths.ensure(report=False)

    # write parquet
    _atomic_write_parquet(bt.reset_index(drop=True), runpaths.backtest)

    # write summary next to parquet
    summary_path = runpaths.backtest.with_suffix(".summary.json")
    _atomic_write_json(summary_path, summary)

    manifest: dict[str, Any] = {
        "kind": "backtest",
        "schema_version": 1,
        "snapshot": runpaths.snapshot,
        "strategy": runpaths.strategy,
        "timeframe": runpaths.timeframe,
        "symbols": list(runpaths.symbols),
        "universe": runpaths.universe,
        "params": runpaths.params,
        "inputs": inputs,
        "engine": engine,
        "paths": {
            "backtest": str(runpaths.backtest),
            "summary": str(summary_path),
            "manifest": str(runpaths.manifest),
        },
        "rows": {"backtest_rows": int(bt.shape[0])},
        "cols": {"cols": list(bt.columns)},
    }
    if extra_manifest:
        manifest.update(extra_manifest)

    _atomic_write_json(runpaths.manifest, manifest)

    # standard pointer (same as features)
    write_latest_pointer(
        runpaths.runs_root,
        runpaths.snapshot,
        runpaths.strategy,
        runpaths.manifest,
        timeframe=runpaths.timeframe,
        universe=runpaths.universe,
    )


    return BacktestArtifact(
        backtest_path=runpaths.backtest,
        summary_path=summary_path,
        manifest_path=runpaths.manifest,
        n_rows=int(bt.shape[0]),
    )
