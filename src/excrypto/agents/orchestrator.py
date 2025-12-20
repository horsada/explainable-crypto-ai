# src/excrypto/agents/orchestrator.py

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict

from excrypto.agents.tools import sh
from excrypto.utils.config import load_cfg, cfg_hash
from excrypto.utils.paths import RunPaths, sym_slug, universe_id


# -------------------- hashing helpers --------------------

def _feat_params_from_yaml(path: Path) -> dict:
    cfg = load_cfg(path)
    return {"hash": cfg_hash({"specs": cfg["specs"]})}


def _params_str(params: Optional[Dict[str, str]]) -> str:
    if not params:
        return ""
    return ",".join(f"{k}={v}" for k, v in sorted(params.items()))


def _label_params_and_col_from_yaml(path: Path) -> tuple[dict, str]:
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

        def _fmt(x: float) -> str:
            s = f"{x:.6g}"
            if "." not in s:
                s += ".0"
            return s

        label_col = f"tb_lbl_h{h}_u{_fmt(u)}_d{_fmt(d)}_w{w}"
        return ({"hash": cfg_hash(canon), **canon}, label_col)

    raise ValueError("unknown label kind in labels yaml")


# -------------------- Orchestrator entry --------------------

def run_plan(config: Path) -> None:
    cfg = load_cfg(config)

    # ---- dataset ----
    ds = cfg["dataset"]
    snapshot = ds["snapshot"]
    timeframe = ds["timeframe"]
    symbols = ds["symbols"]
    exchange = ds.get("exchange", "binance")
    runs_root = Path(cfg.get("runs_root", "runs"))
    sym_csv = ",".join(symbols)

    # ---- snapshot build (use new data CLI) ----
    if "_to_" in snapshot:
        start, end = snapshot.split("_to_")
    else:
        start = end = snapshot

    sh([
        "excrypto", "data", "snapshot",
        "--start", start, "--end", end,
        "--exchange", exchange,
        "--symbols", sym_csv,
        "--timeframe", timeframe,
    ])

    # ---- snapshot viz (CLI-driven raw viz) ----
    uni = sym_slug(symbols[0]) if len(symbols) == 1 else f"u-{universe_id(symbols)}"
    snap_report_dir = runs_root / snapshot / "snapshot_viz" / timeframe / uni
    snap_report_dir.mkdir(parents=True, exist_ok=True)
    sh([
        "excrypto", "viz", "raw",
        "--snapshot", snapshot,
        "--symbols", sym_csv,
        "--timeframe", timeframe,
        "--exchange", exchange,
        "--out-dir", str(snap_report_dir),
    ])

    # ---- features ----
    f_cfg = Path(cfg["features"]["cfg"])
    sh([
        "excrypto", "features", "build",
        "--snapshot", snapshot, "--symbols", sym_csv, "--exchange", exchange, "--timeframe", timeframe,
        "--config", str(f_cfg),
        *(["--write-panel"] if cfg["features"].get("write_panel", False) else []),
    ])

    feat_params = _feat_params_from_yaml(f_cfg)
    fpaths_data = RunPaths(
        snapshot, "features", tuple(symbols), timeframe,
        params=feat_params, runs_root=Path("data/features")
    )

    # ---- labels ----
    l_cfg = Path(cfg["labels"]["cfg"])
    lbl_kind = load_cfg(l_cfg).get("kind", "fh")
    sh([
        "excrypto", "labels", ("tb" if lbl_kind == "tb" else "fh"),
        "--snapshot", snapshot, "--symbols", sym_csv, "--exchange", exchange, "--timeframe", timeframe,
        "--config", str(l_cfg),
        *(["--write-panel"] if cfg["labels"].get("write_panel", False) else []),
    ])

    label_params, label_col = _label_params_and_col_from_yaml(l_cfg)
    lpaths_data = RunPaths(
        snapshot, "labels", tuple(symbols), timeframe,
        params=label_params, runs_root=Path("data/labels")
    )

    # ---- labelled feature viz ----
    feat_report_dir = runs_root / snapshot / "feature_viz" / timeframe / uni
    feat_report_dir.mkdir(parents=True, exist_ok=True)
    sh([
        "excrypto", "viz", "features",
        str(fpaths_data.features),
        "--out-dir", str(feat_report_dir),
        "--labels-path", str(lpaths_data.labels),
        "--label-col", label_col,
    ])

    # ---- baselines (unchanged) ----
    bl = cfg.get("baselines", {})
    if "momentum" in bl:
        fast = str(bl["momentum"].get("fast", 20))
        slow = str(bl["momentum"].get("slow", 60))
        mom_params = {"fast": fast, "slow": slow}
        sh([
            "excrypto", "baseline", "momentum",
            "--snapshot", snapshot, "--symbols", sym_csv, "--exchange", exchange, "--timeframe", timeframe,
            "--fast", fast, "--slow", slow
        ])
        sh([
            "excrypto", "backtest", "run",
            snapshot, "momentum", sym_csv,
            "--params", _params_str(mom_params),
            "--config", "configs/backtest.yaml"
        ])
        sh([
            "excrypto", "risk", "report",
            snapshot, "momentum", sym_csv,
            "--params", _params_str(mom_params),
            "--pnl-col", "pnl_net",
            "--title", f"momentum | {snapshot}"
        ])

    if bl.get("hodl"):
        sh([
            "excrypto", "baseline", "hodl",
            "--snapshot", snapshot, "--symbols", sym_csv, "--exchange", exchange, "--timeframe", timeframe
        ])
        sh(["excrypto", "backtest", "run", snapshot, "hodl", sym_csv, "--config", "configs/backtest.yaml"])
        sh([
            "excrypto", "risk", "report",
            snapshot, "hodl", sym_csv,
            "--pnl-col", "pnl_net",
            "--title", f"hodl | {snapshot}"
        ])

    # ---- ML train ----
    train_cfg = Path(cfg["ml"]["train_cfg"])
    sh([
        "excrypto", "ml", "train", "--config", str(train_cfg), "--snapshot", snapshot,
        "--timeframe", timeframe, "--exchange", exchange, "--symbols", sym_csv,
    ])


    # ---- ML predict via pointer written by ml train ----
    ptr = runs_root / snapshot / "latest_manifest.json"
    if not ptr.exists():
        raise FileNotFoundError(f"Expected manifest pointer not found: {ptr}")

    manifest = Path(json.loads(ptr.read_text())["manifest"])

    predict_cfg_path = runs_root / snapshot / ".predict.json"
    predict_cfg = {
        "manifest": str(manifest),
        "threshold": float(cfg.get("ml", {}).get("predict", {}).get("threshold", 0.5)),
        "runs_root_out": str(cfg.get("ml", {}).get("predict", {}).get("runs_root_out", runs_root)),
    }
    predict_cfg_path.write_text(json.dumps(predict_cfg, indent=2, sort_keys=True))

    sh(["excrypto", "ml", "predict", "--config", str(predict_cfg_path)])

    # ---- report from train manifest ----
    sh(["excrypto", "viz", "from-train", str(manifest)])
