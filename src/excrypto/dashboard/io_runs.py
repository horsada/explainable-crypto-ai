# dashboard/io_runs.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_snapshots(runs_root: Path) -> list[str]:
    if not runs_root.exists():
        return []
    snaps = [p.name for p in runs_root.iterdir() if p.is_dir()]
    snaps.sort()
    return snaps


def list_timeframes(runs_root: Path, snapshot: str, stage: str = "snapshot") -> list[str]:
    base = runs_root / snapshot / stage
    if not base.exists():
        return []
    tfs = [p.name for p in base.iterdir() if p.is_dir()]
    tfs.sort()
    return tfs


def list_universes(runs_root: Path, snapshot: str, timeframe: str, stage: str = "snapshot") -> list[str]:
    """
    Your RunPaths uses 'universe' folder names (e.g. BTC_USDT or u-<hash>).
    This replaces list_symbols for run browsing.
    """
    base = runs_root / snapshot / stage / timeframe
    if not base.exists():
        return []
    uni = [p.name for p in base.iterdir() if p.is_dir()]
    uni.sort()
    return uni


def list_p_hashes(
    runs_root: Path,
    snapshot: str,
    timeframe: str,
    universe: str,
    *,
    stage: str = "snapshot",
) -> list[str]:
    uni_dir = runs_root / snapshot / stage / timeframe / universe
    if not uni_dir.exists():
        return []
    p_dirs = [p.name for p in uni_dir.iterdir() if p.is_dir() and p.name.startswith("p-")]
    p_dirs.sort()
    return p_dirs


def _p(path_str: Optional[str], runs_root: Path) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (runs_root / p).resolve()


def resolve_run_dir(
    runs_root: Path,
    snapshot: str,
    stage: str,
    timeframe: str,
    universe: str,
    *,
    prefer: str = "latest",  # "latest" or "pick"
    p_hash: Optional[str] = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Generic resolver that matches the Features pattern:

      runs/<snapshot>/<stage>/<tf>/<universe>/p-xxxxxx/manifest.json
      runs/<snapshot>/<stage>/<tf>/<universe>/latest_manifest.json

    Returns (run_dir, manifest_path).
    """
    uni_dir = runs_root / snapshot / stage / timeframe / universe
    if not uni_dir.exists():
        return None, None

    if prefer == "pick":
        if not p_hash:
            return None, None
        run_dir = uni_dir / p_hash
        man = run_dir / "manifest.json"
        return (run_dir if run_dir.exists() else None, man if man.exists() else None)

    latest = uni_dir / "latest_manifest.json"
    if not latest.exists():
        return None, None

    lm = read_json(latest)
    paths = lm.get("paths") or {}
    manifest_str = paths.get("manifest")
    if not isinstance(manifest_str, str) or not manifest_str:
        return None, None

    man = Path(manifest_str)
    if not man.is_absolute():
        man = (runs_root / man).resolve()

    return (man.parent if man.exists() else None, man if man.exists() else None)

