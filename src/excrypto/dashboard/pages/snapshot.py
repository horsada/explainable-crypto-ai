# dashboard/pages/1_Snapshot.py
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
    resolve_run_dir,
)

RUNS_ROOT = Path("runs")


def render_snapshot_summary(manifest: dict[str, Any]) -> None:
    st.subheader("Run summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exchange", manifest.get("exchange", "—"))
    c2.metric("Timeframe", manifest.get("timeframe", "—"))

    rows = manifest.get("rows")
    c3.metric("Rows", f"{rows:,}" if isinstance(rows, int) else "—")

    c4.metric("Kind", manifest.get("kind", "—"))

    st.write(f"**Snapshot:** {manifest.get('snapshot', '—')}")
    syms = manifest.get("symbols") or []
    st.write(f"**Symbols:** {', '.join(syms) if syms else '—'}")
    st.write(f"**Created:** {manifest.get('created_at', '—')}")

    cols = manifest.get("columns") or []
    st.write(f"**Columns:** {', '.join(cols) if cols else '—'}")


def pick_snapshot_selection(runs_root: Path) -> tuple[str, str, str, str, Optional[str]]:
    snapshots = list_snapshots(runs_root)
    if not snapshots:
        st.error(f"No snapshots found under: {runs_root.resolve()}")
        st.stop()

    snapshot = st.selectbox("Snapshot range", snapshots, index=len(snapshots) - 1)

    tfs = list_timeframes(runs_root, snapshot, stage="snapshot")
    if not tfs:
        st.error("No timeframes found for runs/<snapshot>/snapshot/")
        st.stop()
    timeframe = st.selectbox("Timeframe", tfs, index=0)

    syms = list_universes(runs_root, snapshot, timeframe, stage="snapshot")
    if not syms:
        st.error("No symbols found for runs/<snapshot>/snapshot/<tf>/")
        st.stop()
    symbol = st.selectbox("Symbol", syms, index=0)

    mode = st.radio("Which run?", ["Latest", "Pick p-hash"])
    prefer = "latest" if mode == "Latest" else "pick"

    p_hash: Optional[str] = None
    if prefer == "pick":
        hashes = list_p_hashes(runs_root, snapshot, timeframe, symbol, stage="snapshot")
        if not hashes:
            st.warning("No p-* runs available to pick.")
        else:
            p_hash = st.selectbox("p-hash", hashes, index=len(hashes) - 1)

    return snapshot, timeframe, symbol, prefer, p_hash


def resolve_panel_path(run_dir: Path, manifest: Optional[dict[str, Any]]) -> Path:
    if manifest:
        panel = (manifest.get("paths") or {}).get("panel")
        if isinstance(panel, str) and panel:
            return Path(panel).resolve()
    return (run_dir / "panel.parquet").resolve()


st.set_page_config(page_title="Runs → Snapshot", layout="wide")
st.title("Runs → Snapshot")

with st.sidebar:
    st.header("Select run")
    snapshot, timeframe, symbol, prefer, p_hash = pick_snapshot_selection(RUNS_ROOT)

run_dir, manifest_path = resolve_run_dir(
    RUNS_ROOT, snapshot, stage='snapshot', timeframe=timeframe, universe=symbol, prefer=prefer, p_hash=p_hash
)

if not run_dir:
    st.error("Could not resolve a snapshot run directory for this selection.")
    st.stop()

manifest = read_json(manifest_path) if (manifest_path and manifest_path.exists()) else None
panel_path = resolve_panel_path(run_dir, manifest)

# -----------------------------
# Single-column layout
# -----------------------------
if manifest:
    render_snapshot_summary(manifest)
    st.divider()

with st.expander("Paths", expanded=False):
    st.code(str(run_dir), language="text")
    st.code(str(panel_path), language="text")
    st.code(str(manifest_path) if manifest_path else "—", language="text")

if manifest:
    with st.expander("Manifest (raw)", expanded=False):
        st.json(manifest)

st.subheader("Snapshot panel")
if not panel_path.exists():
    st.error("panel.parquet not found.")
    st.stop()

df = pd.read_parquet(panel_path)
st.write(f"Rows: **{len(df):,}**  |  Cols: **{df.shape[1]}**")

nan_pct = (df.isna().mean() * 100).sort_values(ascending=False).head(15)
st.dataframe(nan_pct.to_frame("nan_%"), use_container_width=True)

st.dataframe(df.head(200), use_container_width=True)

if "timestamp" in df.columns:
    value_cols = [c for c in df.columns if c not in {"timestamp", "symbol"}]
    if value_cols:
        vcol = st.selectbox("Plot column", value_cols, index=0)
        tmp = df[["timestamp", vcol]].dropna().copy()
        try:
            tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
            tmp = tmp.sort_values("timestamp")
            st.line_chart(tmp.set_index("timestamp")[vcol])
        except Exception:
            st.info("Couldnt parse timestamp to datetime for plotting.")
