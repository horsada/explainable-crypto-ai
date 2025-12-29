# src/excrypto/ml/resolve.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


def write_latest_pointer(
    runs_root: Path,
    snapshot: str,
    stage: str,
    manifest_path: Path,
    *,
    timeframe: Optional[str] = None,
    universe: Optional[str] = None,
) -> Path:
    """
    Best-practice location (matches RunPaths + dashboard browsing):

      runs/<snapshot>/<stage>/<timeframe>/<universe>/latest_manifest.json

    If timeframe/universe are omitted, falls back to legacy:
      runs/<snapshot>/<stage>/latest_manifest.json
    (keep this fallback while migrating; remove later if you want strictness)
    """
    if timeframe and universe:
        d = runs_root / snapshot / stage / timeframe / universe
        d.mkdir(parents=True, exist_ok=True)
        latest = d / "latest_manifest.json"
        payload = {"paths": {"manifest": str(manifest_path)}}
        _atomic_write_json(latest, payload)
        return latest

    # legacy fallback
    d = runs_root / snapshot / stage
    d.mkdir(parents=True, exist_ok=True)
    latest = d / "latest_manifest.json"
    payload = {"paths": {"manifest": str(manifest_path)}}
    _atomic_write_json(latest, payload)
    return latest


def read_latest_pointer(
    runs_root: Path,
    snapshot: str,
    stage: str,
    *,
    timeframe: Optional[str] = None,
    universe: Optional[str] = None,
) -> Path:
    """
    Reads best-practice pointer if timeframe+universe given,
    otherwise reads legacy pointer.
    """
    if timeframe and universe:
        p = runs_root / snapshot / stage / timeframe / universe / "latest_manifest.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing latest pointer: {p}")
        obj = json.loads(p.read_text())
        return Path((obj.get("paths") or {}).get("manifest"))

    p = runs_root / snapshot / stage / "latest_manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing latest pointer: {p}")
    obj = json.loads(p.read_text())
    return Path((obj.get("paths") or {}).get("manifest"))


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(path.read_text())
