# dashboard/app.py
from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="excrypto dashboard", layout="wide")

st.title("excrypto dashboard")
st.caption("Use the sidebar to navigate pages.")

runs_root = Path("runs")
if not runs_root.exists():
    st.error(f"`runs/` not found at: {runs_root.resolve()}")
    st.stop()

# Minimal landing content (low bloat)
st.subheader("Pages")
st.markdown(
    """
- Snapshot
- Features
- Labels
- ML
- Predict
"""
)

st.subheader("Runs root")
st.code(str(runs_root.resolve()), language="text")

# Small sanity check
snapshots = sorted([p.name for p in runs_root.iterdir() if p.is_dir()])
st.write("Snapshots found:", len(snapshots))
if snapshots:
    st.write("Latest snapshot:", snapshots[-1])
