# src/excrypto/utils/hash_debug.py
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import Any, Dict

from .config import load_cfg, cfg_hash  # reuse your existing helpers

def params_id_from_dict(params: Dict[str, Any]) -> str:
    """
    Reproduce RunPaths.params_id behaviour:
      key = "|".join(f"{k}={params[k]}" for k in sorted(params))
      return "p-" + md5(key)[:6]
    """
    if not params:
        return "p-default"
    key = "|".join(f"{k}={params[k]}" for k in sorted(params))
    h = hashlib.md5(key.encode()).hexdigest()[:6]
    return "p-" + h

def make_feat_params_from_cfg(cfg_path: Path | str) -> Dict[str, Any]:
    """Return the canonical feat params we use (content-only)."""
    cfg = load_cfg(cfg_path)
    # canonical: only specs matter for features
    canon = {"specs": cfg["specs"]}
    return {"hash": cfg_hash(canon)}

def make_lbl_params_from_cfg(cfg_path: Path | str) -> Dict[str, Any]:
    """Return canonical label params (content-only) depending on kind."""
    cfg = load_cfg(cfg_path)
    kind = cfg.get("kind", "fh")
    if kind == "fh":
        canon = {"kind": "fh", "h": int(cfg["h"]), "thr": float(cfg.get("thr", 0.0)), "mode": cfg.get("mode", "cls")}
    else:
        # assume tb
        canon = {"kind": "tb", "h": int(cfg["h"]), "u": float(cfg["u"]), "d": float(cfg["d"]), "w": int(cfg["w"])}
    d = {"hash": cfg_hash(canon)}
    d.update(canon)
    return d

def inspect_run_dirs(root: Path, snapshot: str, namespace: str, timeframe: str, symbols_slug: str) -> Dict[str, Path]:
    """
    List p- folders under the given run root for quick comparison.
    Example root="data/features", namespace="features", symbols_slug="BTC_USDT"
    returns dict { "p-xxxxxx": Path(...) }
    """
    base = Path(root) / snapshot / namespace / timeframe / symbols_slug
    out = {}
    if not base.exists():
        return out
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith("p-"):
            out[p.name] = p
    return out

def explain_diff(feat_cfg: Path | str, lbl_cfg: Path | str, snapshot: str, syms: str, timeframe: str):
    """
    Prints hashes and candidate p- ids and scans data/features and data/labels for matches.
    syms: like 'BTC/USDT' -> slug 'BTC_USDT'
    """
    syms_slug = syms.replace("/", "_")
    feat_params = make_feat_params_from_cfg(feat_cfg)
    lbl_params = make_lbl_params_from_cfg(lbl_cfg)

    feat_pid = params_id_from_dict(feat_params)
    lbl_pid  = params_id_from_dict(lbl_params)

    print("=== FEATURE CFG ===")
    print(f"cfg: {feat_cfg}")
    print("canonical params:", json.dumps(feat_params, indent=2, sort_keys=True))
    print("params_id (p-xxx):", feat_pid)
    print()
    print("=== LABEL CFG ===")
    print(f"cfg: {lbl_cfg}")
    print("canonical params:", json.dumps(lbl_params, indent=2, sort_keys=True))
    print("params_id (p-xxx):", lbl_pid)
    print()

    feat_found = inspect_run_dirs(Path("data/features"), snapshot, "features", timeframe, syms_slug)
    lbl_found  = inspect_run_dirs(Path("data/labels"), snapshot, "labels", timeframe, syms_slug)

    print("=== ON DISK: data/features folders ===")
    if feat_found:
        for k, p in feat_found.items():
            print(k, "->", p)
    else:
        print("no match (data/features path not found).")

    print("\n=== ON DISK: data/labels folders ===")
    if lbl_found:
        for k, p in lbl_found.items():
            print(k, "->", p)
    else:
        print("no match (data/labels path not found).")

    print("\n=== SUGGESTION ===")
    if feat_pid not in feat_found:
        print(f"feature params_id {feat_pid} not present under data/features; consider rebuilding features with this cfg or inspect why the content hash differs.")
    else:
        print(f"feature params_id {feat_pid} found at {feat_found[feat_pid]}")
    if lbl_pid not in lbl_found:
        print(f"label params_id {lbl_pid} not present under data/labels; consider rebuilding labels with this cfg or inspect why the content hash differs.")
    else:
        print(f"label params_id {lbl_pid} found at {lbl_found[lbl_pid]}")
