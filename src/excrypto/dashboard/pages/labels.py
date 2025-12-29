# dashboard/pages/3_Labels.py
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


def render_labels_summary(m: dict[str, Any]) -> None:
    st.subheader("Run summary")

    label = m.get("label") or {}
    canon = (label.get("canon") or {}) if isinstance(label, dict) else {}
    rows = m.get("rows") or {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exchange", m.get("exchange", "—"))
    c2.metric("Timeframe", m.get("timeframe", "—"))
    c3.metric("Label col", label.get("label_col", "—"))
    c4.metric("Horizon", str(canon.get("horizon", "—")))

    st.write(f"**Snapshot:** {m.get('snapshot', '—')}")
    syms = m.get("symbols") or []
    st.write(f"**Symbols:** {', '.join(syms) if syms else '—'}")
    st.write(f"**Rows:** {rows.get('n_rows', '—')}")

    st.caption("Label config")
    st.write(f"- kind: `{canon.get('kind', '—')}`")
    st.write(f"- price_col: `{canon.get('price_col', '—')}`")
    st.write(f"- thr: `{canon.get('thr', '—')}`")
    st.write(f"- as_class: `{canon.get('as_class', '—')}`")

    params = m.get("params") or {}
    st.caption("Hashes")
    st.write(f"- label_cfg: `{params.get('label_cfg', '—')}`")
    st.write(f"- params_hash: `{label.get('params_hash', '—')}`")


def resolve_labels_run_dir(
    runs_root: Path,
    snapshot: str,
    timeframe: str,
    symbol: str,
    *,
    prefer: str,
    p_hash: Optional[str],
) -> tuple[Optional[Path], Optional[Path]]:
    sym_dir = runs_root / snapshot / "labels" / timeframe / symbol
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


st.set_page_config(page_title="Runs → Labels", layout="wide")
st.title("Runs → Labels")

with st.sidebar:
    st.header("Select run")

    snapshots = list_snapshots(RUNS_ROOT)
    if not snapshots:
        st.error(f"No snapshots found under: {RUNS_ROOT.resolve()}")
        st.stop()
    snapshot = st.selectbox("Snapshot range", snapshots, index=len(snapshots) - 1)

    tfs = list_timeframes(RUNS_ROOT, snapshot, stage="labels")
    if not tfs:
        st.error("No timeframes found for runs/<snapshot>/labels/")
        st.stop()
    timeframe = st.selectbox("Timeframe", tfs, index=0)

    syms = list_universes(RUNS_ROOT, snapshot, timeframe, stage="labels")
    if not syms:
        st.error("No symbols found for runs/<snapshot>/labels/<tf>/")
        st.stop()
    symbol = st.selectbox("Symbol", syms, index=0)

    mode = st.radio("Which run?", ["Latest", "Pick p-hash"])
    prefer = "latest" if mode == "Latest" else "pick"

    p_hash: Optional[str] = None
    if prefer == "pick":
        hashes = list_p_hashes(RUNS_ROOT, snapshot, timeframe, symbol, stage="labels")
        if not hashes:
            st.warning("No p-* runs available to pick.")
        else:
            p_hash = st.selectbox("p-hash", hashes, index=len(hashes) - 1)

run_dir, manifest_path = resolve_labels_run_dir(
    RUNS_ROOT, snapshot, timeframe, symbol, prefer=prefer, p_hash=p_hash
)
if not run_dir or not manifest_path:
    st.error("Could not resolve a labels run for this selection.")
    st.stop()

m = read_json(manifest_path)

labels_path = _p((m.get("paths") or {}).get("labels"))
panel_path = _p((m.get("paths") or {}).get("panel"))
input_panel_path = _p(m.get("input_panel"))

# ---- single-column layout ----
render_labels_summary(m)
st.divider()

with st.expander("Paths", expanded=False):
    st.code(str(manifest_path), language="text")
    st.code(str(labels_path) if labels_path else "—", language="text")
    st.code(str(panel_path) if panel_path else "—", language="text")
    st.code(str(input_panel_path) if input_panel_path else "—", language="text")

with st.expander("Manifest (raw)", expanded=False):
    st.json(m)

st.subheader("Labels table")
if not labels_path or not labels_path.exists():
    st.error("labels.parquet missing.")
    st.stop()

lbl = pd.read_parquet(labels_path)
st.write(f"Rows: **{len(lbl):,}**  |  Cols: **{lbl.shape[1]}**")
st.dataframe(lbl.head(200), use_container_width=True)

label_col = (m.get("label") or {}).get("label_col")
if isinstance(label_col, str) and label_col in lbl.columns:
    st.subheader("Label distribution")
    counts = lbl[label_col].value_counts(dropna=False).sort_index()
    st.dataframe(counts.to_frame("count"), use_container_width=True)

    st.subheader("Label over time")
    if "timestamp" in lbl.columns:
        tmp = lbl[["timestamp", label_col]].dropna().copy()
        try:
            tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
            tmp = tmp.sort_values("timestamp")
            st.line_chart(tmp.set_index("timestamp")[label_col])
        except Exception:
            st.info("Couldn’t parse timestamp to datetime for plotting.")
else:
    st.info("Label column not found in labels.parquet (or missing in manifest).")

st.subheader("Sanity: close vs label (if available)")
if panel_path and panel_path.exists() and "timestamp" in lbl.columns and isinstance(label_col, str) and label_col in lbl.columns:
    panel = pd.read_parquet(panel_path)
    if "timestamp" in panel.columns and "close" in panel.columns:
        merged = lbl[["timestamp", label_col]].merge(panel[["timestamp", "close"]], on="timestamp", how="inner")
        st.write(f"Merged rows: **{len(merged):,}**")
        st.dataframe(merged.head(200), use_container_width=True)
        try:
            merged["timestamp"] = pd.to_datetime(merged["timestamp"])
            merged = merged.sort_values("timestamp").set_index("timestamp")
            st.line_chart(merged[["close", label_col]])
        except Exception:
            st.info("Couldn’t parse timestamp to datetime for plotting.")
else:
    st.caption("Need labels panel with timestamp + close to show this.")
