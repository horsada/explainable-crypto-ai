# dashboard/pages/4_ML.py
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


def resolve_ml_run_dir(
    runs_root: Path,
    snapshot: str,
    timeframe: str,
    symbol: str,
    *,
    prefer: str,
    p_hash: Optional[str],
) -> tuple[Optional[Path], Optional[Path]]:
    sym_dir = runs_root / snapshot / "ml" / timeframe / symbol
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


def render_ml_summary(m: dict[str, Any]) -> None:
    st.subheader("Run summary")

    train = m.get("train") or {}
    cfg = train.get("train_cfg") or {}
    cv = (cfg.get("cv") or {}) if isinstance(cfg, dict) else {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exchange", m.get("exchange", "—"))
    c2.metric("Timeframe", m.get("timeframe", "—"))
    c3.metric("Model", cfg.get("model", "—"))
    c4.metric("Threshold", str(train.get("threshold", "—")))

    st.write(f"**Snapshot:** {m.get('snapshot', '—')}")
    syms = m.get("symbols") or []
    st.write(f"**Symbols:** {', '.join(syms) if syms else '—'}")
    st.write(f"**Created:** {m.get('created_at', '—')}")
    st.write(f"**Label col:** {(m.get('inputs') or {}).get('label_col', '—')}")
    st.write(f"**Train cfg hash:** {train.get('train_cfg_hash', '—')}")

    st.caption("CV")
    st.write(f"- n_splits: `{cv.get('n_splits', '—')}`")
    st.write(f"- purge: `{cv.get('purge', '—')}`")
    st.write(f"- embargo: `{cv.get('embargo', '—')}`")

    extra = cfg.get("extra_params") or {}
    notes = extra.get("notes")
    if notes:
        st.caption("Notes")
        st.write(notes)


def metrics_table(metrics: dict[str, Any]) -> pd.DataFrame:
    folds = metrics.get("folds") or []
    df = pd.DataFrame(folds)
    if df.empty:
        return df
    df.insert(0, "fold", range(1, len(df) + 1))
    return df


st.set_page_config(page_title="Runs → ML", layout="wide")
st.title("Runs → ML")

with st.sidebar:
    st.header("Select run")

    snapshots = list_snapshots(RUNS_ROOT)
    if not snapshots:
        st.error(f"No snapshots found under: {RUNS_ROOT.resolve()}")
        st.stop()
    snapshot = st.selectbox("Snapshot range", snapshots, index=len(snapshots) - 1)

    tfs = list_timeframes(RUNS_ROOT, snapshot, stage="ml")
    if not tfs:
        st.error("No timeframes found for runs/<snapshot>/ml/")
        st.stop()
    timeframe = st.selectbox("Timeframe", tfs, index=0)

    syms = list_universes(RUNS_ROOT, snapshot, timeframe, stage="ml")
    if not syms:
        st.error("No symbols found for runs/<snapshot>/ml/<tf>/")
        st.stop()
    symbol = st.selectbox("Symbol", syms, index=0)

    mode = st.radio("Which run?", ["Latest", "Pick p-hash"])
    prefer = "latest" if mode == "Latest" else "pick"

    p_hash: Optional[str] = None
    if prefer == "pick":
        hashes = list_p_hashes(RUNS_ROOT, snapshot, timeframe, symbol, stage="ml")
        if not hashes:
            st.warning("No p-* runs available to pick.")
        else:
            p_hash = st.selectbox("p-hash", hashes, index=len(hashes) - 1)

run_dir, manifest_path = resolve_ml_run_dir(
    RUNS_ROOT, snapshot, timeframe, symbol, prefer=prefer, p_hash=p_hash
)
if not run_dir or not manifest_path:
    st.error("Could not resolve an ML run for this selection.")
    st.stop()

m = read_json(manifest_path)
paths = m.get("paths") or {}

metrics_path = _p(paths.get("metrics"))
model_path = _p(paths.get("model"))

# ---- single-column layout ----
render_ml_summary(m)
st.divider()

with st.expander("Paths", expanded=False):
    st.code(str(manifest_path), language="text")
    st.code(str(metrics_path) if metrics_path else "—", language="text")
    st.code(str(model_path) if model_path else "—", language="text")

    inputs = m.get("inputs") or {}
    st.caption("Inputs")
    for k in ["features_manifest", "features_path", "labels_manifest", "labels_path", "label_col"]:
        if k in inputs:
            st.write(f"- **{k}**: `{inputs[k]}`")

st.subheader("Metrics")
if not metrics_path or not metrics_path.exists():
    st.error("metrics.json missing.")
    st.stop()

metrics = read_json(metrics_path)
df = metrics_table(metrics)
if df.empty:
    st.info("No fold metrics found.")
    st.stop()

st.dataframe(df, use_container_width=True)

# mean/std
num_cols = [c for c in df.columns if c != "fold" and pd.api.types.is_numeric_dtype(df[c])]
if num_cols:
    summary = pd.DataFrame(
        {
            "mean": df[num_cols].mean(numeric_only=True),
            "std": df[num_cols].std(numeric_only=True),
            "min": df[num_cols].min(numeric_only=True),
            "max": df[num_cols].max(numeric_only=True),
        }
    )
    st.subheader("Summary")
    st.dataframe(summary, use_container_width=True)

    plot_col = st.selectbox("Plot metric", num_cols, index=0)
    st.line_chart(df.set_index("fold")[plot_col])

with st.expander("Manifest (raw)", expanded=False):
    st.json(m)
