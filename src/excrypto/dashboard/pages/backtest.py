# dashboard/pages/6_Backtests.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

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


def resolve_backtest_run_dir(
    runs_root: Path,
    snapshot: str,
    timeframe: str,
    symbol: str,
    *,
    prefer: str,
    p_hash: Optional[str],
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Returns (run_dir, summary_path) for backtest stage:
      runs/<snapshot>/backtest/<tf>/<symbol>/p-xxxxxx/
        backtest.parquet
        backtest.summary.json
    """
    sym_dir = runs_root / snapshot / "backtest" / timeframe / symbol
    if not sym_dir.exists():
        return None, None

    if prefer == "pick" and p_hash:
        run_dir = sym_dir / p_hash
        summ = run_dir / "backtest.summary.json"
        return (run_dir if run_dir.exists() else None, summ if summ.exists() else None)

    latest = sym_dir / "latest_manifest.json"
    if latest.exists():
        lm = read_json(latest)
        # If you store a manifest with paths, try that first
        paths = lm.get("paths") or {}
        summ_str = paths.get("summary")
        if isinstance(summ_str, str) and summ_str:
            summ = _p(summ_str)
            return (summ.parent if summ else None, summ if summ and summ.exists() else None)

        # fallback if latest manifest stores p_hash
        if isinstance(lm.get("p_hash"), str):
            run_dir = sym_dir / lm["p_hash"]
            summ = run_dir / "backtest.summary.json"
            return (run_dir if run_dir.exists() else None, summ if summ.exists() else None)

    p_dirs = [p for p in sym_dir.iterdir() if p.is_dir() and p.name.startswith("p-")]
    if not p_dirs:
        return None, None
    p_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    run_dir = p_dirs[0]
    summ = run_dir / "backtest.summary.json"
    return run_dir, (summ if summ.exists() else None)


def render_backtest_summary(s: dict[str, Any]) -> None:
    st.subheader("Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe", f"{s.get('sharpe', 0.0):.2f}")
    c2.metric("Max drawdown", f"{s.get('max_drawdown', 0.0):.2%}")
    c3.metric("Equity end", f"{s.get('equity_end', 1.0):.3f}")
    c4.metric("Bars", int(s.get('bars', 0)))

    c5, c6, c7 = st.columns(3)
    c5.metric("Mean / bar", f"{s.get('mean_ret_bar', 0.0):.6f}")
    c6.metric("Vol / bar", f"{s.get('vol_bar', 0.0):.6f}")
    c7.metric("Ann factor", f"{s.get('ann_fac', 0.0):.1f}")


st.set_page_config(page_title="Runs → Backtests", layout="wide")
st.title("Runs → Backtests")

with st.sidebar:
    st.header("Select run")

    snapshots = list_snapshots(RUNS_ROOT)
    if not snapshots:
        st.error(f"No snapshots found under: {RUNS_ROOT.resolve()}")
        st.stop()
    snapshot = st.selectbox("Snapshot range", snapshots, index=len(snapshots) - 1)

    tfs = list_timeframes(RUNS_ROOT, snapshot, stage="backtest")
    if not tfs:
        st.error("No timeframes found for runs/<snapshot>/backtest/")
        st.stop()
    timeframe = st.selectbox("Timeframe", tfs, index=0)

    syms = list_universes(RUNS_ROOT, snapshot, timeframe, stage="backtest")
    if not syms:
        st.error("No symbols found for runs/<snapshot>/backtest/<tf>/")
        st.stop()
    symbol = st.selectbox("Symbol", syms, index=0)

    mode = st.radio("Which run?", ["Latest", "Pick p-hash"])
    prefer = "latest" if mode == "Latest" else "pick"

    p_hash: Optional[str] = None
    if prefer == "pick":
        hashes = list_p_hashes(RUNS_ROOT, snapshot, timeframe, symbol, stage="backtest")
        if not hashes:
            st.warning("No p-* runs available to pick.")
        else:
            p_hash = st.selectbox("p-hash", hashes, index=len(hashes) - 1)


run_dir, summary_path = resolve_backtest_run_dir(
    RUNS_ROOT, snapshot, timeframe, symbol, prefer=prefer, p_hash=p_hash
)
if not run_dir:
    st.error("Could not resolve a backtest run for this selection.")
    st.stop()

bt_path = run_dir / "backtest.parquet"
if not bt_path.exists():
    st.error(f"backtest.parquet missing: {bt_path}")
    st.stop()

summary = read_json(summary_path) if summary_path and summary_path.exists() else {}
render_backtest_summary(summary)
st.divider()

with st.expander("Paths", expanded=False):
    st.code(str(run_dir), language="text")
    st.code(str(bt_path), language="text")
    st.code(str(summary_path) if summary_path else "—", language="text")

bt = pd.read_parquet(bt_path)
st.subheader("Backtest table")
st.write(f"Rows: **{len(bt):,}**  |  Cols: **{bt.shape[1]}**")
st.dataframe(bt.head(300), use_container_width=True)

# Equity plot
if "equity" in bt.columns:
    st.subheader("Equity")
    tmp = bt[["equity"]].copy()
    if not isinstance(tmp.index, pd.DatetimeIndex) and "timestamp" in bt.columns:
        try:
            tmp["timestamp"] = pd.to_datetime(bt["timestamp"], utc=True)
            tmp = tmp.set_index("timestamp")[["equity"]]
        except Exception:
            pass
    st.line_chart(tmp["equity"])

# Drawdown plot
if "drawdown" in bt.columns:
    st.subheader("Drawdown")
    st.line_chart(bt["drawdown"])
