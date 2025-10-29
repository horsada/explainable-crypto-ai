from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def price_series(panel_path: Path, out_dir: Path) -> Path:
    """panel parquet with columns: timestamp,symbol,close (optionally volume)."""
    _ensure_dir(out_dir)
    df = pd.read_parquet(panel_path)
    fig, ax = plt.subplots(figsize=(10,3))
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("timestamp")
        ax.plot(pd.to_datetime(g["timestamp"]), g["close"], label=sym)
    ax.set_title("Price over time"); ax.set_ylabel("Close"); ax.legend(loc="upper left", fontsize=8)
    p = out_dir / "raw_price.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p

def volume_series(panel_path: Path, out_dir: Path, volume_col: str = "volume") -> Path | None:
    _ensure_dir(out_dir)
    df = pd.read_parquet(panel_path)
    if volume_col not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(10,3))
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("timestamp")
        ax.plot(pd.to_datetime(g["timestamp"]), g[volume_col], label=sym)
    ax.set_title("Volume over time"); ax.set_ylabel("Volume"); ax.legend(loc="upper left", fontsize=8)
    p = out_dir / "raw_volume.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p

def returns_hist(panel_path: Path, out_dir: Path) -> Path:
    _ensure_dir(out_dir)
    df = pd.read_parquet(panel_path).sort_values("timestamp")
    df["ret_log"] = np.log(df.groupby("symbol")["close"].pct_change().add(1)).replace([np.inf,-np.inf], np.nan)
    fig, ax = plt.subplots(figsize=(6,4))
    df["ret_log"].dropna().hist(ax=ax, bins=100)
    ax.set_title("Log-returns distribution"); ax.set_xlabel("ret_log")
    p = out_dir / "raw_returns_hist.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p

def rolling_vol(panel_path: Path, out_dir: Path, window: int = 24) -> Path:
    _ensure_dir(out_dir)
    df = pd.read_parquet(panel_path).sort_values("timestamp")
    df["ret"] = df.groupby("symbol")["close"].pct_change()
    vol = df.groupby("symbol")["ret"].rolling(window, min_periods=max(2, window//5)).std().reset_index(level=0, drop=True)
    df["vol"] = vol
    fig, ax = plt.subplots(figsize=(10,3))
    for sym, g in df.groupby("symbol"):
        ax.plot(pd.to_datetime(g["timestamp"]), g["vol"], label=sym)
    ax.set_title(f"Rolling volatility (window={window})"); ax.set_ylabel("σ")
    ax.legend(loc="upper left", fontsize=8)
    p = out_dir / "raw_rolling_vol.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p

def missing_heatmap(panel_path: Path, out_dir: Path) -> Path:
    """Heatmap of missing close by time vs symbol."""
    _ensure_dir(out_dir)
    df = pd.read_parquet(panel_path)[["timestamp","symbol","close"]]
    piv = df.pivot_table(index="timestamp", columns="symbol", values="close")
    miss = piv.isna().astype(int)
    fig, ax = plt.subplots(figsize=(8,4))
    im = ax.imshow(miss.values.T, aspect="auto", interpolation="nearest")
    ax.set_title("Missing data heatmap (1=missing)")
    ax.set_xlabel("time idx"); ax.set_ylabel("symbol")
    ax.set_yticks(range(len(miss.columns))); ax.set_yticklabels(miss.columns, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    p = out_dir / "raw_missing_heatmap.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p
