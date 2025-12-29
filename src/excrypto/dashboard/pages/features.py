# dashboard/pages/2_Features.py
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


def render_features_summary(m: dict[str, Any]) -> None:
    st.subheader("Run summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exchange", m.get("exchange", "—"))
    c2.metric("Timeframe", m.get("timeframe", "—"))
    c3.metric("Spec hash", (m.get("params") or {}).get("spec_hash", "—"))
    c4.metric("Specs hash", m.get("specs_hash", "—"))

    rows = m.get("rows") or {}
    cols = m.get("cols") or {}

    st.write(f"**Snapshot:** {m.get('snapshot', '—')}")
    syms = m.get("symbols") or []
    st.write(f"**Symbols:** {', '.join(syms) if syms else '—'}")
    st.write(f"**Feature rows:** {rows.get('feature_rows', '—')}")
    st.write(f"**Panel rows:** {rows.get('panel_rows', '—')}")
    st.write(f"**n_features:** {cols.get('n_features', '—')}")


def resolve_features_run_dir(
    runs_root: Path,
    snapshot: str,
    timeframe: str,
    symbol: str,
    *,
    prefer: str,
    p_hash: Optional[str],
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Returns (run_dir, manifest_path) for features stage:
      runs/<snapshot>/features/<tf>/<symbol>/p-xxxxxx/manifest.json
    """
    sym_dir = runs_root / snapshot / "features" / timeframe / symbol
    if not sym_dir.exists():
        return None, None

    if prefer == "pick" and p_hash:
        run_dir = sym_dir / p_hash
        man = run_dir / "manifest.json"
        return (run_dir if run_dir.exists() else None, man if man.exists() else None)

    # prefer == "latest": use latest_manifest.json if present, else newest p-* dir
    latest = sym_dir / "latest_manifest.json"
    if latest.exists():
        lm = read_json(latest)
        paths = lm.get("paths") or {}
        manifest_str = paths.get("manifest")
        if isinstance(manifest_str, str) and manifest_str:
            man = _p(manifest_str)
            return (man.parent if man else None, man if man and man.exists() else None)

        # fallback if latest manifest stores p_hash
        if isinstance(lm.get("p_hash"), str):
            run_dir = sym_dir / lm["p_hash"]
            man = run_dir / "manifest.json"
            return (run_dir if run_dir.exists() else None, man if man.exists() else None)

    p_dirs = [p for p in sym_dir.iterdir() if p.is_dir() and p.name.startswith("p-")]
    if not p_dirs:
        return None, None
    p_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    run_dir = p_dirs[0]
    man = run_dir / "manifest.json"
    return run_dir, (man if man.exists() else None)


st.set_page_config(page_title="Runs → Features", layout="wide")
st.title("Runs → Features")

with st.sidebar:
    st.header("Select run")

    snapshots = list_snapshots(RUNS_ROOT)
    if not snapshots:
        st.error(f"No snapshots found under: {RUNS_ROOT.resolve()}")
        st.stop()
    snapshot = st.selectbox("Snapshot range", snapshots, index=len(snapshots) - 1)

    tfs = list_timeframes(RUNS_ROOT, snapshot, stage="features")
    if not tfs:
        st.error("No timeframes found for runs/<snapshot>/features/")
        st.stop()
    timeframe = st.selectbox("Timeframe", tfs, index=0)

    syms = list_universes(RUNS_ROOT, snapshot, timeframe, stage="features")
    if not syms:
        st.error("No symbols found for runs/<snapshot>/features/<tf>/")
        st.stop()
    symbol = st.selectbox("Symbol", syms, index=0)

    mode = st.radio("Which run?", ["Latest", "Pick p-hash"])
    prefer = "latest" if mode == "Latest" else "pick"

    p_hash: Optional[str] = None
    if prefer == "pick":
        hashes = list_p_hashes(RUNS_ROOT, snapshot, timeframe, symbol, stage="features")
        if not hashes:
            st.warning("No p-* runs available to pick.")
        else:
            p_hash = st.selectbox("p-hash", hashes, index=len(hashes) - 1)

run_dir, manifest_path = resolve_features_run_dir(
    RUNS_ROOT, snapshot, timeframe, symbol, prefer=prefer, p_hash=p_hash
)
if not run_dir or not manifest_path:
    st.error("Could not resolve a features run for this selection.")
    st.stop()

m = read_json(manifest_path)

input_panel_path = _p(m.get("input_panel"))
panel_path = _p((m.get("paths") or {}).get("panel"))
features_path = _p((m.get("paths") or {}).get("features"))

# -----------------------------
# Single-column layout
# -----------------------------
render_features_summary(m)
st.divider()

with st.expander("Paths", expanded=False):
    st.code(str(manifest_path), language="text")
    st.code(str(input_panel_path) if input_panel_path else "—", language="text")
    st.code(str(panel_path) if panel_path else "—", language="text")
    st.code(str(features_path) if features_path else "—", language="text")

st.subheader("Specs")
specs = m.get("specs") or []
if specs:
    st.dataframe(pd.json_normalize(specs), use_container_width=True)
else:
    st.info("No specs found in manifest.")

st.subheader("Features table")
if not features_path or not features_path.exists():
    st.error("features.parquet missing.")
    st.stop()

feat = pd.read_parquet(features_path)
st.write(f"Rows: **{len(feat):,}**  |  Cols: **{feat.shape[1]}**")

nan_pct = (feat.isna().mean() * 100).sort_values(ascending=False)
st.dataframe(nan_pct.to_frame("nan_%"), use_container_width=True, height=300)

st.dataframe(feat.head(200), use_container_width=True)

if "timestamp" in feat.columns:
    feature_cols = [c for c in feat.columns if c not in {"timestamp", "symbol"}]
    if feature_cols:
        col = st.selectbox("Plot feature", feature_cols, index=0)
        tmp = feat[["timestamp", col]].dropna().copy()
        try:
            tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
            tmp = tmp.sort_values("timestamp")
            st.line_chart(tmp.set_index("timestamp")[col])
        except Exception:
            st.info("Couldn’t parse timestamp to datetime for plotting.")
