from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from excrypto.dashboard.io_runs import (
    list_p_hashes,
    list_snapshots,
    list_universes,
    list_timeframes,
)

RUNS_ROOT = Path("runs")


def resolve_run_dir(
    runs_root: Path,
    snapshot: str,
    stage: str,
    timeframe: str,
    symbol: str,
    *,
    prefer: str,
    p_hash: Optional[str],
) -> Optional[Path]:
    sym_dir = runs_root / snapshot / stage / timeframe / symbol
    if not sym_dir.exists():
        return None

    if prefer == "pick" and p_hash:
        run_dir = sym_dir / p_hash
        return run_dir if run_dir.exists() else None

    p_dirs = [p for p in sym_dir.iterdir() if p.is_dir() and p.name.startswith("p-")]
    if not p_dirs:
        return None
    p_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return p_dirs[0]


st.set_page_config(page_title="Runs → Baselines", layout="wide")
st.title("Runs → Baselines")

with st.sidebar:
    st.header("Select run")

    snapshots = list_snapshots(RUNS_ROOT)
    if not snapshots:
        st.error(f"No snapshots found under: {RUNS_ROOT.resolve()}")
        st.stop()
    snapshot = st.selectbox("Snapshot range", snapshots, index=len(snapshots) - 1)

    strategy = st.selectbox("Strategy", ["momentum", "hodl"], index=0)

    tfs = list_timeframes(RUNS_ROOT, snapshot, stage=strategy)
    if not tfs:
        st.error(f"No timeframes found for runs/<snapshot>/{strategy}/")
        st.stop()
    timeframe = st.selectbox("Timeframe", tfs, index=0)

    syms = list_universes(RUNS_ROOT, snapshot, timeframe, stage=strategy)
    if not syms:
        st.error(f"No symbols found for runs/<snapshot>/{strategy}/<tf>/")
        st.stop()
    symbol = st.selectbox("Symbol", syms, index=0)

    mode = st.radio("Which run?", ["Latest", "Pick p-hash"])
    prefer = "latest" if mode == "Latest" else "pick"

    p_hash: Optional[str] = None
    if prefer == "pick":
        hashes = list_p_hashes(RUNS_ROOT, snapshot, timeframe, symbol, stage=strategy)
        if not hashes:
            st.warning("No p-* runs available to pick.")
        else:
            p_hash = st.selectbox("p-hash", hashes, index=len(hashes) - 1)

run_dir = resolve_run_dir(RUNS_ROOT, snapshot, strategy, timeframe, symbol, prefer=prefer, p_hash=p_hash)
if not run_dir:
    st.error("Could not resolve a baseline run for this selection.")
    st.stop()

signals_path = run_dir / "signals.parquet"
if not signals_path.exists():
    st.error(f"signals.parquet missing: {signals_path}")
    st.stop()

with st.expander("Paths", expanded=False):
    st.code(str(run_dir), language="text")
    st.code(str(signals_path), language="text")

sig = pd.read_parquet(signals_path)

st.subheader("Signals table")
st.write(f"Rows: **{len(sig):,}**  |  Cols: **{sig.shape[1]}**")
st.dataframe(sig.head(300), use_container_width=True)

if "signal" in sig.columns:
    st.subheader("Signal distribution")
    st.dataframe(sig["signal"].value_counts(dropna=False).to_frame("count"), use_container_width=True)

    if "timestamp" in sig.columns:
        st.subheader("Signal over time")
        tmp = sig[["timestamp", "signal"]].dropna().copy()
        try:
            tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], utc=True)
            tmp = tmp.sort_values("timestamp")
            st.line_chart(tmp.set_index("timestamp")["signal"])
        except Exception:
            st.info("Couldn’t parse timestamp to datetime for plotting.")
