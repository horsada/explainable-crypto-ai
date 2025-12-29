# dashboard/pages/5_Predict.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

from excrypto.dashboard.io_runs import (
    list_p_hashes,
    list_snapshots,
    list_universes,
    list_timeframes,
    read_json,
)

RUNS_ROOT = Path("runs")


def _p(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    return Path(path_str).resolve()


def resolve_predict_run_dir(
    runs_root: Path,
    snapshot: str,
    timeframe: str,
    symbol: str,
    *,
    prefer: str,
    p_hash: Optional[str],
) -> tuple[Optional[Path], Optional[Path]]:
    sym_dir = runs_root / snapshot / "predict" / timeframe / symbol
    if not sym_dir.exists():
        return None, None

    if prefer == "pick" and p_hash:
        run_dir = sym_dir / p_hash
        man = run_dir / "manifest.json"
        return (run_dir if run_dir.exists() else None, man if man.exists() else None)

    latest = sym_dir / "latest_manifest.json"
    if latest.exists():
        lm = read_json(latest)
        man_str = lm.get("manifest")
        if isinstance(man_str, str) and man_str:
            man = _p(man_str)
            return (man.parent if man else None, man if man and man.exists() else None)

    p_dirs = [p for p in sym_dir.iterdir() if p.is_dir() and p.name.startswith("p-")]
    if not p_dirs:
        return None, None
    p_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    run_dir = p_dirs[0]
    man = run_dir / "manifest.json"
    return run_dir, (man if man.exists() else None)


def render_predict_summary(m: dict[str, Any]) -> None:
    st.subheader("Run summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exchange", m.get("exchange", "—"))
    c2.metric("Timeframe", m.get("timeframe", "—"))
    c3.metric("Threshold", str((m.get("params") or {}).get("threshold", "—")))
    c4.metric("Kind", m.get("kind", "—"))

    st.write(f"**Snapshot:** {m.get('snapshot', '—')}")
    syms = m.get("symbols") or []
    st.write(f"**Symbols:** {', '.join(syms) if syms else '—'}")
    st.write(f"**Created:** {m.get('created_at', '—')}")

    inputs = m.get("inputs") or {}
    with st.expander("Inputs", expanded=False):
        for k in ["features", "ml_manifest", "model"]:
            if k in inputs:
                st.write(f"- **{k}**: `{inputs[k]}`")


def pick_selection(runs_root: Path) -> tuple[str, str, str, str, Optional[str]]:
    snapshots = list_snapshots(runs_root)
    if not snapshots:
        st.error(f"No snapshots found under: {runs_root.resolve()}")
        st.stop()
    snapshot = st.selectbox("Snapshot range", snapshots, index=len(snapshots) - 1)

    tfs = list_timeframes(runs_root, snapshot, stage="predict")
    if not tfs:
        st.error("No timeframes found for runs/<snapshot>/predict/")
        st.stop()
    timeframe = st.selectbox("Timeframe", tfs, index=0)

    syms = list_universes(runs_root, snapshot, timeframe, stage="predict")
    if not syms:
        st.error("No symbols found for runs/<snapshot>/predict/<tf>/")
        st.stop()
    symbol = st.selectbox("Symbol", syms, index=0)

    mode = st.radio("Which run?", ["Latest", "Pick p-hash"])
    prefer = "latest" if mode == "Latest" else "pick"

    p_hash: Optional[str] = None
    if prefer == "pick":
        hashes = list_p_hashes(runs_root, snapshot, timeframe, symbol, stage="predict")
        if not hashes:
            st.warning("No p-* runs available to pick.")
        else:
            p_hash = st.selectbox("p-hash", hashes, index=len(hashes) - 1)

    return snapshot, timeframe, symbol, prefer, p_hash


st.set_page_config(page_title="Runs → Predict", layout="wide")
st.title("Runs → Predict")

with st.sidebar:
    st.header("Select run")
    snapshot, timeframe, symbol, prefer, p_hash = pick_selection(RUNS_ROOT)

run_dir, manifest_path = resolve_predict_run_dir(
    RUNS_ROOT, snapshot, timeframe, symbol, prefer=prefer, p_hash=p_hash
)
if not run_dir or not manifest_path:
    st.error("Could not resolve a predict run for this selection.")
    st.stop()

m = read_json(manifest_path)
paths = m.get("paths") or {}

signals_path = _p(paths.get("signals"))
panel_path = _p(paths.get("panel"))

# ---- single-column layout ----
render_predict_summary(m)
st.divider()

with st.expander("Paths", expanded=False):
    st.code(str(manifest_path), language="text")
    st.code(str(signals_path) if signals_path else "—", language="text")
    st.code(str(panel_path) if panel_path else "—", language="text")

with st.expander("Manifest (raw)", expanded=False):
    st.json(m)

st.subheader("Signals")
if not signals_path or not signals_path.exists():
    st.error("signals.parquet missing.")
    st.stop()

sig = pd.read_parquet(signals_path)
st.write(f"Rows: **{len(sig):,}**  |  Cols: **{sig.shape[1]}**")
st.dataframe(sig.head(200), use_container_width=True)

# Pick probability/score column heuristically
candidate_cols = [c for c in sig.columns if c.lower() not in {"timestamp", "symbol"}]
score_col = None
for c in candidate_cols:
    lc = c.lower()
    if any(k in lc for k in ["proba", "prob", "score", "pred", "signal"]):
        score_col = c
        break
if score_col is None and candidate_cols:
    score_col = candidate_cols[0]

thr = float((m.get("params") or {}).get("threshold", 0.5))

if "timestamp" in sig.columns and score_col:
    st.subheader("Signal timeline")
    tmp = sig[["timestamp", score_col]].dropna().copy()
    try:
        tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
        tmp = tmp.sort_values("timestamp").set_index("timestamp")
        st.line_chart(tmp[score_col])
    except Exception:
        st.info("Couldnt parse timestamp to datetime for plotting.")

    st.subheader("Threshold crossings")
    s2 = sig[["timestamp", score_col]].dropna().copy()
    try:
        s2["timestamp"] = pd.to_datetime(s2["timestamp"])
        s2 = s2.sort_values("timestamp")
        s2["above_thr"] = s2[score_col] >= thr
        s2["cross_up"] = s2["above_thr"] & (~s2["above_thr"].shift(1).fillna(False))
        crossings = s2.loc[s2["cross_up"], ["timestamp", score_col]].tail(200)
        st.write(f"Cross-ups (>= {thr}): **{len(crossings):,}**")
        st.dataframe(crossings, use_container_width=True)
    except Exception:
        st.info("Couldn’t compute crossings (timestamp parse or numeric issue).")

st.subheader("Panel + signal (joined)")
if panel_path and panel_path.exists() and "timestamp" in sig.columns:
    panel = pd.read_parquet(panel_path)
    if "timestamp" in panel.columns:
        merged = sig.merge(panel, on="timestamp", how="left", suffixes=("", "_panel"))
        st.write(f"Merged rows: **{len(merged):,}**")
        st.dataframe(merged.head(200), use_container_width=True)
else:
    st.caption("panel.parquet missing or no timestamp column to join.")
