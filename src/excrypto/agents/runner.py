# src/excrypto/agents/runner.py
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import json

from excrypto.agents.tools import sh
from excrypto.utils.registry import find
from excrypto.utils.paths import RunPaths, sym_slug, universe_id
from excrypto.utils.config import load_cfg, cfg_hash

# ---------- helpers ----------
def _snapshot_exists(snapshot: str, exchange: str) -> bool:
    df = find(snapshot_id=snapshot)
    return (not df.empty) and (df["exchange"].nunique() == 1) and (df["exchange"].iloc[0] == exchange)

def _has(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0

def _parse_symbols(csv: str) -> List[str]:
    return [s.strip() for s in csv.split(",") if s.strip()]

def _params_str(params: Optional[Dict[str, str]]) -> str:
    if not params: return ""
    return ",".join(f"{k}={v}" for k, v in sorted(params.items()))

def _feat_params_from_yaml(path: Path) -> dict:
    cfg = load_cfg(path)
    return {"hash": cfg_hash({"specs": cfg["specs"]})}

def _label_params_from_yaml(path: Path) -> dict:
    cfg = load_cfg(path)
    if cfg.get("kind", "fh") == "fh":
        canon = {"kind":"fh","h": int(cfg["h"]), "thr": float(cfg.get("thr",0.0)), "mode": cfg.get("mode","cls")}
    else:
        canon = {"kind":"tb","h": int(cfg["h"]), "u": float(cfg["u"]), "d": float(cfg["d"]), "w": int(cfg["w"])}
    return {"hash": cfg_hash(canon), **canon}

def _latest_model_manifest(runs_root: Path, snapshot: str, timeframe: str, syms: List[str], model: str) -> Path | None:
    uni = sym_slug(syms[0]) if len(syms) == 1 else f"u-{universe_id(syms)}"
    base = runs_root / snapshot / f"ml_{model}" / timeframe / uni
    if not base.exists(): 
        return None
    best, best_key = None, ("", 0.0)
    for d in base.iterdir():
        if not (d.is_dir() and d.name.startswith("p-")):
            continue
        # support both locations
        man = d / "manifest.json"
        ts = ""
        if man:
            try: ts = json.loads(man.read_text()).get("created_at", "")
            except: ts = ""
        key = (ts, d.stat().st_mtime)
        if key > best_key:
            best, best_key = (man if man else None), key
    return best

def _make_temp_ml_cfg(ml_train_cfg: Path, snapshot: str, timeframe: str, syms: list[str]) -> Path:
    cfg = load_cfg(ml_train_cfg)
    cfg.setdefault("dataset", {})
    cfg["dataset"]["snapshot"]  = snapshot
    cfg["dataset"]["timeframe"] = timeframe
    cfg["dataset"]["symbols"]   = syms
    # write a temp file next to the original for transparency
    tmp = ml_train_cfg.parent / f".patched_{ml_train_cfg.name}"
    # write as JSON (YAML also fine if you have a dumper)
    tmp.write_text(json.dumps(cfg, indent=2))
    return tmp


# ---------- agents ----------
def run_daily(
    snapshot: Optional[str],
    symbols: str,
    exchange: str = "binance",
    override: bool = False,
    mom_params: Optional[Dict[str, str]] = None,
    *,
    features_cfg: Path | None = None,
    labels_cfg: Path | None = None,
    ml_train_cfg: Path | None = None,
    runs_root: Path = Path("runs"),
):
    """Daily agent: snapshot → baselines → (optional) features/labels → ML train+predict."""
    snap = snapshot or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    syms = _parse_symbols(symbols)
    mom_params = mom_params or {"fast": "20", "slow": "60"}

    # 1) Snapshot
    if _snapshot_exists(snap, exchange) and not override:
        print(f"[agent] snapshot {snap}@{exchange} exists — skipping fetch")
    else:
        sh(["excrypto","pipeline","snapshot","--snapshot",snap,"--exchange",exchange,"--symbols",symbols])

    # -------- Baselines (unchanged) --------
    m_paths = RunPaths(snapshot=snap, strategy="momentum", symbols=tuple(syms), timeframe='1m', params=mom_params)
    m_paths.ensure()
    if not (_has(m_paths.signals) and _has(m_paths.panel)) or override:
        sh(["excrypto","baseline","momentum","--snapshot",snap,"--symbols",symbols,
            "--fast", mom_params.get("fast","20"), "--slow", mom_params.get("slow","60")])

    if not _has(m_paths.backtest) or override:
        sh(["excrypto","backtest","run", snap, "momentum", symbols, "--params", _params_str(mom_params),
            "--config", "configs/backtest.yaml"])

    if not _has(m_paths.report_md) or override:
        sh(["excrypto","risk","report", snap, "momentum", symbols,
            "--params", _params_str(mom_params), "--pnl-col","pnl_net", "--title", f"Momentum | {snap}"])

    # -------- Features + Labels + ML (optional if cfgs provided) --------
    if features_cfg and labels_cfg and ml_train_cfg:
        # features
        f_params = _feat_params_from_yaml(features_cfg)
        f_paths = RunPaths(snap, "features", tuple(syms), "1m", params=f_params, runs_root=Path("data/features"))
        if not _has(f_paths.base / "features.parquet") or override:
            sh(["excrypto","features","build","--snapshot",snap,"--symbols",symbols,"--timeframe","1m",
                "--cfg", str(features_cfg), "--write-panel"])

        # labels
        l_params = _label_params_from_yaml(labels_cfg)
        l_paths = RunPaths(snap, "labels", tuple(syms), "1m", params=l_params, runs_root=Path("data/labels"))
        if not _has(l_paths.base / "labels.parquet") or override:
            kind = load_cfg(labels_cfg).get("kind","fh")
            if kind == "tb":
                sh(["excrypto","labels","tb","--snapshot",snap,"--symbols",symbols,"--timeframe","1m",
                    "--cfg", str(labels_cfg), "--write-panel"])
            else:
                sh(["excrypto","labels","fh","--snapshot",snap,"--symbols",symbols,"--timeframe","1m",
                    "--cfg", str(labels_cfg), "--write-panel"])

        # train
        # Ensure ml_train_cfg.dataset matches (snapshot/timeframe/symbols)
        mlcfg = load_cfg(ml_train_cfg)
        ds = mlcfg.get("dataset", {})
        if (ds.get("snapshot") != snap) or (ds.get("timeframe") != "1m") or (ds.get("symbols") != syms):
            print("[agent] warning: ml train config dataset != agent args")
        sh(["excrypto","ml","train","--config", str(ml_train_cfg)])

        # predict (use latest manifest under the run root)
        model = mlcfg.get("model","rf")
        man = _latest_model_manifest(runs_root, snap, "1m", syms, model)
        if man is None:
            raise RuntimeError("[agent] could not locate a trained model manifest for predict")
        # write a tiny temp predict.yaml next to the run
        pred_yaml = man.parent / "predict.yaml"
        pred_yaml.write_text(json.dumps({"manifest": str(man).replace("\\","/"),
                                         "threshold": 0.5,
                                         "runs_root_out": str(runs_root).replace("\\","/")}))
        sh(["excrypto","ml","predict","--config", str(pred_yaml)])


def run_range(
    start: str,
    end: str,
    timeframe: str,
    symbols: str,
    exchange: str = "binance",
    override: bool = False,
    mom_params: Optional[Dict[str, str]] = None,
    *,
    features_cfg: Path | None = None,
    labels_cfg: Path | None = None,
    ml_train_cfg: Path | None = None,
    runs_root: Path = Path("runs"),
):
    """Range agent: snapshot → baselines → (optional) features/labels → ML train+predict."""
    snap = f"{start}_to_{end}"
    syms = _parse_symbols(symbols)
    mom_params = mom_params or {"fast": "20", "slow": "60"}

    # 1) Combined snapshot
    if _snapshot_exists(snap, exchange) and not override:
        print(f"[agent] combined snapshot {snap}@{exchange} exists — skipping fetch")
    else:
        sh(["excrypto","pipeline","snapshot","--start",start,"--end",end,"--timeframe",timeframe,
            "--exchange",exchange,"--symbols",symbols])

    # --- Momentum ---
    m_paths = RunPaths(snapshot=snap, strategy="momentum", symbols=tuple(syms), timeframe=timeframe, params=mom_params)
    m_paths.ensure()
    if not (_has(m_paths.signals) and _has(m_paths.panel)) or override:
        sh(["excrypto","baseline","momentum","--snapshot",snap,"--symbols",symbols,
            "--fast", mom_params.get("fast","20"), "--slow", mom_params.get("slow","60")])
    if not _has(m_paths.backtest) or override:
        sh(["excrypto","backtest","run", snap, "momentum", symbols, "--params", _params_str(mom_params),
            "--config","configs/backtest.yaml"])
    if not _has(m_paths.report_md) or override:
        sh(["excrypto","risk","report", snap, "momentum", symbols, "--params", _params_str(mom_params),
            "--pnl-col","pnl_net","--title", f"momentum | {snap}"])

    # --- HODL ---
    h_paths = RunPaths(snapshot=snap, strategy="hodl", symbols=tuple(syms), timeframe=timeframe, params=None)
    h_paths.ensure()
    if not (_has(h_paths.signals) and _has(h_paths.panel)) or override:
        sh(["excrypto","baseline","hodl","--snapshot", snap, "--symbols", symbols])
    if not _has(h_paths.backtest) or override:
        sh(["excrypto","backtest","run", snap, "hodl", symbols, "--config","configs/backtest.yaml"])
    if not _has(h_paths.report_md) or override:
        sh(["excrypto","risk","report", snap, "hodl", symbols, "--pnl-col","pnl_net","--title", f"hodl | {snap}"])

    # --- Features + Labels + ML (optional) ---
    if features_cfg and labels_cfg and ml_train_cfg:
        # features
        f_params = _feat_params_from_yaml(features_cfg)
        f_paths = RunPaths(snap, "features", tuple(syms), timeframe, params=f_params, runs_root=Path("data/features"))
        if not _has(f_paths.base / "features.parquet") or override:
            sh(["excrypto","features","build","--snapshot",snap,"--symbols",symbols,"--timeframe",timeframe,
                "--config", str(features_cfg), "--write-panel"])

        # labels
        l_params = _label_params_from_yaml(labels_cfg)
        l_paths = RunPaths(snap, "labels", tuple(syms), timeframe, params=l_params, runs_root=Path("data/labels"))
        if not _has(l_paths.base / "labels.parquet") or override:
            kind = load_cfg(labels_cfg).get("kind","fh")
            if kind == "tb":
                sh(["excrypto","labels","tb","--snapshot",snap,"--symbols",symbols,"--timeframe",timeframe,
                    "--config", str(labels_cfg), "--write-panel"])
            else:
                sh(["excrypto","labels","fh","--snapshot",snap,"--symbols",symbols,"--timeframe",timeframe,
                    "--config", str(labels_cfg), "--write-panel"])

        # train
        ml_train_cfg = _make_temp_ml_cfg(ml_train_cfg, snap, timeframe, syms)
        sh(["excrypto","ml","train","--config", str(ml_train_cfg)])

        # predict using latest manifest
        model = mlcfg.get("model","rf")
        man = _latest_model_manifest(runs_root, snap, timeframe, syms, model)
        if man is None:
            raise RuntimeError("[agent] could not locate a trained model manifest for predict")
        pred_yaml = man.parent / "predict.yaml"
        pred_yaml.write_text(json.dumps({"manifest": str(man).replace("\\","/"),
                                         "threshold": 0.5,
                                         "runs_root_out": str(runs_root).replace("\\","/")}))
        sh(["excrypto","ml","predict","--config", str(pred_yaml)])
