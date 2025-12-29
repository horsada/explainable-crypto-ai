# src/excrypto/labels/builder.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import hashlib
import json
import pandas as pd

from excrypto.utils.paths import RunPaths
from excrypto.labels.labelers import fixed_horizon_return, triple_barrier
from excrypto.ml.resolve import write_latest_pointer


NanPolicy = Literal["keep", "drop_any"]


@dataclass(frozen=True)
class LabelsArtifact:
    labels_path: Path
    panel_path: Path
    manifest_path: Path
    n_rows_in: int
    n_rows_out: int
    label_col: str
    params_hash: str


def _hash_obj(obj: dict[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:10]


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)

def _write_latest_pointer(stage_dir: Path, manifest_path: Path) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "latest_manifest.json").write_text(
        json.dumps({"manifest": str(manifest_path)}, indent=2, sort_keys=True)
    )


def canonical_label_params(kind: str, params: dict[str, Any]) -> dict[str, Any]:
    """
    Canonicalize params so hashing/paths are stable.
    Keeps only known keys and forces types.
    """
    k = kind.lower().strip()
    if k in {"fh", "fixed_horizon", "return", "fixed_horizon_return"}:
        return {
            "kind": "fixed_horizon_return",
            "horizon": int(params.get("horizon", 24)),
            "as_class": bool(params.get("as_class", True)),
            "thr": float(params.get("thr", 0.0)),
            "price_col": str(params.get("price_col", "close")),
        }

    if k in {"tb", "triple_barrier"}:
        return {
            "kind": "triple_barrier",
            "horizon": int(params.get("horizon", 24)),
            "up_mult": float(params.get("up_mult", 2.0)),
            "dn_mult": float(params.get("dn_mult", 2.0)),
            "vol_window": int(params.get("vol_window", 50)),
            "price_col": str(params.get("price_col", "close")),
            # labeler options
            "min_periods": params.get("min_periods", None),
            "tail_value": int(params.get("tail_value", 0)),
        }

    raise ValueError(f"Unknown label kind '{kind}'")


def label_col_name(canon: dict[str, Any]) -> str:
    if canon["kind"] == "fixed_horizon_return":
        h = canon["horizon"]
        return f"fh_lbl_{h}" if canon["as_class"] else f"fh_ret_{h}"
    if canon["kind"] == "triple_barrier":
        return f"tb_lbl_h{canon['horizon']}_u{canon['up_mult']}_d{canon['dn_mult']}_w{canon['vol_window']}"
    raise ValueError(f"Unknown canonical kind '{canon['kind']}'")


def build_labels_frame(
    panel: pd.DataFrame,
    *,
    canon: dict[str, Any],
    group_col: str = "symbol",
    time_col: str = "timestamp",
    nan_policy: NanPolicy = "keep",
) -> pd.DataFrame:
    """
    Returns a DataFrame with [time_col, group_col, label_col] (and keeps row order where possible).
    """
    for col in (time_col, group_col):
        if col not in panel.columns:
            raise ValueError(f"panel missing required column '{col}'")

    price_col = canon["price_col"]
    if price_col not in panel.columns:
        raise ValueError(f"panel missing price column '{price_col}'")

    lbl_col = label_col_name(canon)

    def _label_one(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(time_col)
        price = g[price_col]
        if canon["kind"] == "fixed_horizon_return":
            s = fixed_horizon_return(
                price,
                horizon=int(canon["horizon"]),
                as_class=bool(canon["as_class"]),
                thr=float(canon["thr"]),
                name=lbl_col,
            )
        else:
            s = triple_barrier(
                price,
                horizon=int(canon["horizon"]),
                up_mult=float(canon["up_mult"]),
                dn_mult=float(canon["dn_mult"]),
                vol_window=int(canon["vol_window"]),
                min_periods=canon.get("min_periods", None),
                tail_value=int(canon.get("tail_value", 0)),
                name=lbl_col,
            )
        out = g[[time_col, group_col]].copy()
        out[lbl_col] = s.to_numpy()
        return out

    labels = panel.groupby(group_col, group_keys=False).apply(_label_one).reset_index(drop=True)

    if nan_policy == "drop_any":
        labels = labels.dropna(subset=[lbl_col]).reset_index(drop=True)

    return labels


def write_labels_artifact(
    runpaths: RunPaths,
    labels: pd.DataFrame,
    *,
    canon: dict[str, Any],
    extra_manifest: dict[str, Any] | None = None,
    ensure_report_dir: bool = False,
) -> LabelsArtifact:
    runpaths.ensure(report=ensure_report_dir)

    lbl_col = label_col_name(canon)
    if lbl_col not in labels.columns:
        raise ValueError(f"labels output missing expected label column '{lbl_col}'")

    # Write labels.parquet
    _atomic_write_parquet(labels, runpaths.labels)

    # Keep a "panel" artifact for parity with features stage (optional but convenient):
    # here it's just the labels frame.
    _atomic_write_parquet(labels, runpaths.panel)

    params_hash = _hash_obj(canon)

    manifest: dict[str, Any] = {
        "kind": "labels",
        "schema_version": 1,
        "snapshot": runpaths.snapshot,
        "strategy": runpaths.strategy,
        "timeframe": runpaths.timeframe,
        "symbols": list(runpaths.symbols),
        "universe": runpaths.universe,
        "params": runpaths.params,
        "label": {
            "label_col": lbl_col,
            "canon": canon,
            "params_hash": params_hash,
        },
        "paths": {
            "labels": str(runpaths.labels),
            "panel": str(runpaths.panel),
            "manifest": str(runpaths.manifest),
        },
        "rows": {"n_rows": int(labels.shape[0])},
        "time_range": {
            "min": str(labels["timestamp"].min()) if "timestamp" in labels.columns else None,
            "max": str(labels["timestamp"].max()) if "timestamp" in labels.columns else None,
        },
    }
    if extra_manifest:
        manifest.update(extra_manifest)

    _atomic_write_json(runpaths.manifest, manifest)

    write_latest_pointer(
        runpaths.runs_root,
        runpaths.snapshot,
        runpaths.strategy,
        runpaths.manifest,
        timeframe=runpaths.timeframe,
        universe=runpaths.universe,
    )


    return LabelsArtifact(
        labels_path=runpaths.labels,
        panel_path=runpaths.panel,
        manifest_path=runpaths.manifest,
        n_rows_in=int(labels.shape[0]),
        n_rows_out=int(labels.shape[0]),
        label_col=lbl_col,
        params_hash=params_hash,
    )


def build_and_write_labels(
    panel: pd.DataFrame,
    *,
    canon: dict[str, Any],
    runpaths: RunPaths,
    group_col: str = "symbol",
    time_col: str = "timestamp",
    nan_policy: NanPolicy = "keep",
    extra_manifest: dict[str, Any] | None = None,
) -> LabelsArtifact:
    labels = build_labels_frame(
        panel,
        canon=canon,
        group_col=group_col,
        time_col=time_col,
        nan_policy=nan_policy,
    )
    return write_labels_artifact(
        runpaths,
        labels,
        canon=canon,
        extra_manifest=extra_manifest,
    )
