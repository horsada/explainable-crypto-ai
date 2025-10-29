# src/excrypto/agents/orchestrator.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict

from excrypto.agents.tools import sh
from excrypto.utils.config import load_cfg, cfg_hash
from excrypto.utils.paths import RunPaths, sym_slug, universe_id, params_id
from excrypto.utils.loader import load_snapshot


# -------------------- hashing helpers (single source of truth) --------------------

def _feat_params_from_yaml(path: Path) -> dict:
    """Content-only hash for features: specs list only."""
    cfg = load_cfg(path)
    return {"hash": cfg_hash({"specs": cfg["specs"]})}

def _params_str(params: Optional[Dict[str, str]]) -> str:
    if not params: return ""
    return ",".join(f"{k}={v}" for k, v in sorted(params.items()))


def _label_params_and_col_from_yaml(path: Path) -> tuple[dict, str]:
    """
    Content-only label params + deterministic label_col.
    FH: canon = {kind:'fh', h, thr, mode}  → fh_lbl_{h} | fh_ret_{h}
    TB: canon = {kind:'tb', h, u, d, w}    → tb_lbl_h{h}_u{u}_d{d}_w{w}
    """
    cfg = load_cfg(path)
    kind = cfg.get("kind", "fh").lower()

    if kind == "fh":
        h = int(cfg["h"])
        thr = float(cfg.get("thr", 0.0))
        mode = cfg.get("mode", "cls")
        canon = {"kind": "fh", "h": h, "thr": thr, "mode": mode}
        label_col = f"fh_{'lbl' if mode == 'cls' else 'ret'}_{h}"
        return ({"hash": cfg_hash(canon), **canon}, label_col)

    if kind == "tb":
        h = int(cfg["h"])
        u = float(cfg["u"])
        d = float(cfg["d"])
        w = int(cfg["w"])
        canon = {"kind": "tb", "h": h, "u": u, "d": d, "w": w}
        # canonical float formatting (ensure .0 when integer-like)
        def _fmt(x: float) -> str:
            s = f"{x:.6g}"
            if "." not in s:
                s += ".0"
            return s
        label_col = f"tb_lbl_h{h}_u{_fmt(u)}_d{_fmt(d)}_w{w}"
        return ({"hash": cfg_hash(canon), **canon}, label_col)

    raise ValueError("unknown label kind in labels yaml")


# -------------------- ML config patching --------------------

def _patch_ml_cfg(ml_cfg_path: Path, snapshot: str, timeframe: str, symbols: List[str]) -> Path:
    """Inject dataset fields into train yaml so it matches orchestrator dataset."""
    mlcfg = load_cfg(ml_cfg_path)
    mlcfg.setdefault("dataset", {})
    mlcfg["dataset"]["snapshot"] = snapshot
    mlcfg["dataset"]["timeframe"] = timeframe
    mlcfg["dataset"]["symbols"] = symbols
    patched = ml_cfg_path.parent / f".patched_{ml_cfg_path.name}"
    patched.write_text(json.dumps(mlcfg, indent=2, sort_keys=True))
    return patched


# -------------------- Manifest resolution strategies --------------------
def _manifest_from_exact_hash(full_yaml: Path, runs_root: Path) -> Path | None:
    """
    Reconstruct the exact model run dir from YAMLs by recomputing the params hash.
    params = {"label": label_col, "feat": feat_params, "lbl": label_params, **extra}
    """
    cfg = load_cfg(full_yaml)

    snap = cfg["dataset"]["snapshot"]
    timeframe = cfg["dataset"]["timeframe"]
    symbols = cfg["dataset"]["symbols"]

    features_yaml = Path(cfg["features"]["cfg"])
    labels_yaml = Path(cfg["labels"]["cfg"])
    train_yaml = Path(cfg["ml"]["train_cfg"])

    model = load_cfg(train_yaml).get("model", "rf")
    extra = load_cfg(train_yaml).get("extra_params", {})

    feat_params = _feat_params_from_yaml(features_yaml)
    label_params, label_col = _label_params_and_col_from_yaml(labels_yaml)

    run_params = {"label": label_col, "feat": feat_params, "lbl": label_params, **extra}
    pid = params_id(run_params)

    uni = sym_slug(symbols[0]) if len(symbols) == 1 else f"u-{universe_id(symbols)}"
    run_dir = Path(runs_root) / snap / f"ml_{model}" / timeframe / uni / pid

    m1 = run_dir / "report" / "manifest.json"
    m2 = run_dir / "manifest.json"
    return m1 if m1.exists() else (m2 if m2.exists() else None)


# -------------------- Orchestrator entry --------------------

def run_plan(config: Path) -> None:
    """
    Single-orchestrator pipeline driven by one YAML:
      1) snapshot
      2) features build
      3) labels build
      4) (optional) baselines
      5) ml train (dataset injected)
      6) ml predict (prefer exact-hash; fallback to LAST_RUN; then latest)
    """
    cfg = load_cfg(config)

    # ---- dataset ----
    ds = cfg["dataset"]
    snapshot = ds["snapshot"]
    timeframe = ds["timeframe"]
    symbols = ds["symbols"]
    exchange = ds.get("exchange", "binance")
    runs_root = Path(cfg.get("runs_root", "runs"))

    sym_csv = ",".join(symbols)

    # ---- snapshot (idempotent; supports range or single) ----
    if "_to_" in snapshot:
        start, end = snapshot.split("_to_")
        sh([
            "excrypto", "pipeline", "snapshot",
            "--start", start, "--end", end,
            "--timeframe", timeframe,
            "--exchange", exchange, "--symbols", sym_csv
        ])
    else:
        sh([
            "excrypto", "pipeline", "snapshot",
            "--snapshot", snapshot,
            "--exchange", exchange, "--symbols", sym_csv
        ])

    # ---- snapshot viz ----

    # load panel from the registry
    panel = load_snapshot(snapshot, symbols, timeframe)  # index=timestamp; has 'symbol','close' (maybe 'volume')

    # write a small parquet for viz
    uni = sym_slug(symbols[0]) if len(symbols)==1 else f"u-{universe_id(symbols)}"
    snap_report_dir = runs_root / snapshot / "snapshot_viz" / timeframe / uni
    snap_report_dir.mkdir(parents=True, exist_ok=True)
    snap_parquet = snap_report_dir / "snapshot_panel.parquet"
    cols = ["timestamp", "symbol", "close"]
    if "volume" in panel.columns: cols.append("volume")
    panel.reset_index()[cols].to_parquet(snap_parquet, index=False)

    # run raw viz
    sh(["excrypto", "viz", "raw", str(snap_parquet), "--out-dir", str(snap_report_dir)])

    # ---- features ----
    f_cfg = Path(cfg["features"]["cfg"])
    sh([
        "excrypto","features","build",
        "--snapshot", snapshot, "--symbols", sym_csv, "--timeframe", timeframe,
        "--config", str(f_cfg),
        *(["--write-panel"] if cfg["features"].get("write_panel", False) else []),
    ])
    
    # feature viz
    feat_params = _feat_params_from_yaml(f_cfg)
    fpaths_data = RunPaths(snapshot, "features", tuple(symbols), timeframe,
                       params=feat_params, runs_root=Path("data/features"))
    fpaths_viz  = RunPaths(snapshot, "features", tuple(symbols), timeframe,
                       params=feat_params, runs_root=runs_root)
    sh([
        "excrypto","viz","features",
        str(fpaths_data.features),
        "--out-dir", str(fpaths_viz.report_dir),
    ])


    # ---- labels ----
    l_cfg = Path(cfg["labels"]["cfg"])
    lbl_kind = load_cfg(l_cfg).get("kind", "fh")
    # build labels
    sh([
        "excrypto","labels","tb" if lbl_kind=="tb" else "fh",
        "--snapshot", snapshot, "--symbols", sym_csv, "--timeframe", timeframe,
        "--config", str(l_cfg),
        *(["--write-panel"] if cfg["labels"].get("write_panel", False) else []),
    ])

    # labels viz
    label_params, label_col = _label_params_and_col_from_yaml(l_cfg)
    lpaths_data = RunPaths(snapshot, "labels", tuple(symbols), timeframe,
                       params=label_params, runs_root=Path("data/labels"))
    
    sh([
        "excrypto","viz","features",
        str(fpaths_data.features),
        "--out-dir", str(fpaths_viz.report_dir),
        "--labels-path", str(lpaths_data.labels),
        "--label-col", label_col,
    ])

    # ---- baselines (optional) ----
    bl = cfg.get("baselines", {})
    if "momentum" in bl:
        fast = str(bl["momentum"].get("fast", 20))
        slow = str(bl["momentum"].get("slow", 60))
        mom_params = {"fast": fast, "slow": slow}
        sh([
            "excrypto", "baseline", "momentum",
            "--snapshot", snapshot, "--symbols", sym_csv, "--timeframe", timeframe,
            "--fast", fast, "--slow", slow
        ])
        sh(["excrypto","backtest","run", snapshot, "momentum", sym_csv, "--params", _params_str(mom_params),
            "--config","configs/backtest.yaml"])
        sh(["excrypto","risk","report", snapshot, "momentum", sym_csv, "--params", _params_str(mom_params), "--pnl-col","pnl_net","--title", f"momentum | {snapshot}"])
        
    if bl.get("hodl"):
        sh([
            "excrypto", "baseline", "hodl",
            "--snapshot", snapshot, "--symbols", sym_csv, "--timeframe", timeframe
        ])
        sh(["excrypto","backtest","run", snapshot, "hodl", sym_csv, "--config","configs/backtest.yaml"])
        sh(["excrypto","risk","report", snapshot, "hodl", sym_csv, "--pnl-col","pnl_net","--title", f"hodl | {snapshot}"])
        

    # ---- ML train ----
    train_cfg_in = Path(cfg["ml"]["train_cfg"])
    train_cfg = _patch_ml_cfg(train_cfg_in, snapshot, timeframe, symbols)
    sh(["excrypto", "ml", "train", "--config", str(train_cfg)])

    # ---- ML predict (prefer exact-hash; else LAST_RUN; else latest) ----
    manifest = _manifest_from_exact_hash(config, runs_root)
    if manifest is None:
        raise RuntimeError("Could not resolve a trained model manifest")

    pred_cfg_path = manifest.parent / "predict.yaml"
    pred_cfg = {
        "manifest": str(manifest).replace("\\", "/"),
        "threshold": float(cfg["ml"].get("predict", {}).get("threshold", 0.5)),
        "runs_root_out": str(Path(cfg["ml"].get("predict", {}).get("runs_root_out", "runs"))).replace("\\", "/"),
    }
    pred_cfg_path.write_text(json.dumps(pred_cfg, indent=2, sort_keys=True))
    sh(["excrypto", "ml", "predict", "--config", str(pred_cfg_path)])

    # plotting
    sh(["excrypto", "viz", "from-train", str(manifest)])
