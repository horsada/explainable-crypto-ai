from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def histograms(features_path: Path, out_dir: Path, max_cols: int = 24) -> Path:
    """Histograms for first N feature columns (exclude timestamp,symbol)."""
    _ensure_dir(out_dir)
    X = pd.read_parquet(features_path)
    feats = [c for c in X.columns if c not in ("timestamp","symbol")][:max_cols]
    n = len(feats)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, feats):
        X[col].dropna().hist(ax=ax, bins=50)
        ax.set_title(col, fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    p = out_dir / "features_hist.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p

def rolling_feature(features_path: Path, out_dir: Path, feature: str, window: int = 48) -> Path:
    """Rolling mean z-score for one feature to visualize drift."""
    _ensure_dir(out_dir)
    X = pd.read_parquet(features_path)
    if feature not in X.columns:
        raise ValueError(f"feature '{feature}' not found")
    # use a single symbol series if multi-symbol present
    if "symbol" in X.columns:
        sym = X["symbol"].iloc[0]
        X = X[X["symbol"] == sym]
    s = X[feature].astype(float)
    z = (s - s.rolling(window, min_periods=max(2, window//5)).mean()) / s.rolling(window, min_periods=max(2, window//5)).std()
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(pd.to_datetime(X["timestamp"]), z)
    ax.set_title(f"{feature}: rolling z-score (window={window})"); ax.set_ylabel("z")
    p = out_dir / f"feature_{feature}_rolling.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p

def corr_heatmap(features_path: Path, out_dir: Path) -> Path:
    _ensure_dir(out_dir)
    X = pd.read_parquet(features_path)
    X = X.drop(columns=[c for c in ("timestamp","symbol") if c in X])
    corr = X.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr.index)));  ax.set_yticklabels(corr.index, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    p = out_dir / "features_corr.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p

def feature_target_corr(features_path: Path, labels_path: Path, label_col: str, out_dir: Path) -> Path:
    """Spearman correlation between features and the target label."""
    _ensure_dir(out_dir)
    X = pd.read_parquet(features_path)
    y = pd.read_parquet(labels_path)[["timestamp","symbol", label_col]]
    df = X.merge(y, on=["timestamp","symbol"], how="inner").dropna()
    feats = [c for c in df.columns if c not in ("timestamp","symbol", label_col)]
    corrs = df[feats + [label_col]].corr(method="spearman")[label_col].drop(label_col)
    corrs = corrs.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, max(3, 0.25*len(corrs))))
    ax.barh(corrs.index, corrs.values)
    ax.invert_yaxis()
    ax.set_title(f"Feature ↔ {label_col} (Spearman)")
    p = out_dir / "features_target_corr.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p
