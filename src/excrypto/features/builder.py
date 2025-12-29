# src/excrypto/features/builder.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import hashlib
import json
import pandas as pd

from excrypto.features.pipeline import FeaturePipeline
from excrypto.utils.paths import RunPaths
from excrypto.ml.resolve import write_latest_pointer


NanPolicy = Literal["keep", "drop_any"]


@dataclass(frozen=True)
class FeaturesArtifact:
    features_path: Path
    panel_path: Path
    manifest_path: Path
    n_rows_in: int
    n_rows_out: int
    n_features: int
    specs_hash: str


def _hash_specs(specs: list[dict[str, Any]]) -> str:
    # Stable hash: canonical JSON with sorted keys
    payload = json.dumps(specs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:10]


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)

def build_features_frame(
    panel: pd.DataFrame,
    specs: Iterable[dict[str, Any]],
    *,
    group_col: str = "symbol",
    nan_policy: NanPolicy = "keep",
    return_with_input_cols: bool = True,
) -> pd.DataFrame:
    """
    Pure builder: takes an in-memory panel and returns a DataFrame with feature columns.

    Assumptions:
      - panel contains `group_col` (default: "symbol") if you want per-symbol generation.
      - FeaturePipeline returns a DataFrame indexed like the input frame.

    Notes:
      - We do NOT call pipeline.fit() here to avoid leakage-by-default.
    """
    specs_list = list(specs)
    pipe = FeaturePipeline(specs_list).build()

    if group_col in panel.columns:
        # Per-symbol feature generation
        feats = (
            panel.groupby(group_col, group_keys=False)
            .apply(lambda g: pipe.transform(g))
            .reset_index(drop=True)
        )
    else:
        feats = pipe.transform(panel).reset_index(drop=True)

    # Align with original row order
    feats = feats.reset_index(drop=True)

    if nan_policy == "drop_any":
        mask = ~feats.isna().any(axis=1)
        feats = feats.loc[mask].reset_index(drop=True)
        base = panel.reset_index(drop=True).loc[mask].reset_index(drop=True)
    else:
        base = panel.reset_index(drop=True)

    if return_with_input_cols:
        # Append feature cols onto the original panel (no overwrites)
        overlap = set(base.columns) & set(feats.columns)
        if overlap:
            raise ValueError(f"Feature output cols collide with input cols: {sorted(overlap)}")
        return pd.concat([base, feats], axis=1)

    return feats


def write_features_artifact(
    runpaths: RunPaths,
    panel_with_features: pd.DataFrame,
    *,
    specs: list[dict[str, Any]],
    extra_manifest: dict[str, Any] | None = None,
    ensure_report_dir: bool = False,
) -> FeaturesArtifact:
    """
    Writes:
      - runpaths.panel    (panel_with_features)
      - runpaths.features (features-only subset)
      - runpaths.manifest (metadata)
    """
    runpaths.ensure(report=ensure_report_dir)

    # Infer feature columns from specs
    feature_cols = [s["output_col"] for s in specs]
    missing = [c for c in feature_cols if c not in panel_with_features.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns in output frame: {missing}")

    key_cols = ["timestamp", "symbol"]
    missing_keys = [c for c in key_cols if c not in panel_with_features.columns]
    if missing_keys:
        raise ValueError(f"Panel missing key cols required for ML join: {missing_keys}")

    features_only = panel_with_features[key_cols + feature_cols].copy()


    specs_hash = _hash_specs(specs)

    _atomic_write_parquet(panel_with_features, runpaths.panel)
    _atomic_write_parquet(features_only, runpaths.features)

    manifest: dict[str, Any] = {
        "kind": "features",
        "schema_version": 1,
        "snapshot": runpaths.snapshot,
        "strategy": runpaths.strategy,
        "timeframe": runpaths.timeframe,
        "symbols": list(runpaths.symbols),
        "universe": runpaths.universe,
        "params": runpaths.params,
        "specs_hash": specs_hash,
        "specs": specs,  # keep for reproducibility; remove if you prefer only hash
        "paths": {
            "panel": str(runpaths.panel),
            "features": str(runpaths.features),
            "manifest": str(runpaths.manifest),
        },
        "rows": {
            "panel_rows": int(panel_with_features.shape[0]),
            "feature_rows": int(features_only.shape[0]),
        },
        "cols": {
            "n_features": int(features_only.shape[1]),
            "feature_cols": feature_cols,
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


    return FeaturesArtifact(
        features_path=runpaths.features,
        panel_path=runpaths.panel,
        manifest_path=runpaths.manifest,
        n_rows_in=int(panel_with_features.shape[0]),
        n_rows_out=int(features_only.shape[0]),
        n_features=int(features_only.shape[1]),
        specs_hash=specs_hash,
    )


def build_and_write_features(
    panel: pd.DataFrame,
    specs: list[dict[str, Any]],
    runpaths: RunPaths,
    *,
    group_col: str = "symbol",
    nan_policy: NanPolicy = "keep",
    extra_manifest: dict[str, Any] | None = None,
) -> FeaturesArtifact:
    """
    One-stop API for CLI/orchestrator:
      panel -> compute features -> write artifacts -> return artifact info
    """
    panel_out = build_features_frame(
        panel,
        specs,
        group_col=group_col,
        nan_policy=nan_policy,
        return_with_input_cols=True,
    )
    return write_features_artifact(
        runpaths,
        panel_out,
        specs=specs,
        extra_manifest=extra_manifest,
    )
