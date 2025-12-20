# src/excrypto/viz/api.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score, accuracy_score
from excrypto.utils.paths import RunPaths
from .raw import price_series, volume_series, returns_hist, rolling_vol, missing_heatmap
from .features import histograms, rolling_feature, corr_heatmap, feature_target_corr

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_report_links(out_dir: Path, items: dict[str, Path]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "viz.md"
    lines = ["# Visualization", ""]
    for title, p in items.items():
        rel = p.name if p.parent == out_dir else p.as_posix()
        lines.append(f"- **{title}**: {rel}")
    out.write_text("\n".join(lines))
    return out


def plot_feature_correlation(features_path: Path, out_dir: Path) -> Path:
    _ensure_dir(out_dir)
    X = pd.read_parquet(features_path)
    X = X.drop(columns=[c for c in ["timestamp","symbol"] if c in X])
    corr = X.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr.index)));  ax.set_yticklabels(corr.index, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    p = out_dir / "features_corr_spearman.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p

def plot_label_balance(labels_path: Path, label_col: str, out_dir: Path) -> Path:
    _ensure_dir(out_dir)
    y = pd.read_parquet(labels_path)[["timestamp", label_col]].dropna()
    # class balance over time (per month)
    y["month"] = pd.to_datetime(y["timestamp"]).dt.to_period("M").dt.to_timestamp()
    grp = y.groupby("month")[label_col].mean()
    fig, ax = plt.subplots(figsize=(8,3))
    grp.plot(ax=ax)
    ax.set_title(f"Label balance over time: {label_col} (mean)")
    ax.set_ylabel("mean label"); ax.set_xlabel("month")
    p = out_dir / "labels_balance_over_time.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p

def plot_roc_pr(y_true: pd.Series, score: np.ndarray, out_dir: Path) -> tuple[Path, Path]:
    _ensure_dir(out_dir)
    # ROC
    fpr, tpr, _ = roc_curve(y_true, score)
    auc = roc_auc_score(y_true, score)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr); ax.plot([0,1],[0,1], linestyle="--")
    ax.set_title(f"ROC (AUC={auc:.3f})"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    p1 = out_dir / "ml_roc.png"; fig.tight_layout(); fig.savefig(p1, dpi=150); plt.close(fig)
    # PR
    prec, rec, _ = precision_recall_curve(y_true, score)
    ap = average_precision_score(y_true, score)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(rec, prec)
    ax.set_title(f"PR (AP={ap:.3f})"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    p2 = out_dir / "ml_pr.png"; fig.tight_layout(); fig.savefig(p2, dpi=150); plt.close(fig)
    return p1, p2

def plot_threshold_sweep(y_true: pd.Series, score: np.ndarray, out_dir: Path) -> Path:
    _ensure_dir(out_dir)
    thr = np.linspace(0.0, 1.0, 101)
    f1 = []; acc = []
    for t in thr:
        yhat = (score >= t).astype(int)
        f1.append(f1_score(y_true, yhat, zero_division=0))
        acc.append(accuracy_score(y_true, yhat))
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(thr, f1, label="F1"); ax.plot(thr, acc, label="Accuracy")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score"); ax.legend()
    p = out_dir / "ml_threshold_sweep.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p

def plot_equity_curve(panel_path: Path, signals_path: Path, out_dir: Path) -> Path:
    """Simple PnL (close-to-close, long when signal>=0.5). For demo; replace with your backtester’s PnL if present."""
    _ensure_dir(out_dir)
    pan = pd.read_parquet(panel_path)  # needs timestamp,symbol,close
    sig = pd.read_parquet(signals_path)  # timestamp,symbol,signal
    df = pan.merge(sig, on=["timestamp","symbol"], how="inner").sort_values("timestamp")
    df["ret"] = df.groupby("symbol")["close"].pct_change()
    df["pos"] = (df["signal"] >= 0.5).astype(float)
    df["pnl"] = df["pos"] * df["ret"]
    eq = (1.0 + df.groupby("timestamp")["pnl"].mean()).cumprod()
    fig, ax = plt.subplots(figsize=(8,3))
    eq.plot(ax=ax)
    ax.set_title("Equity curve (toy)"); ax.set_ylabel("Equity"); ax.set_xlabel("Time")
    p = out_dir / "equity_curve.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p

def from_train_manifest(manifest_path: Path) -> dict:
    man = json.loads(Path(manifest_path).read_text())
    # Find run base dir regardless of manifest location
    run_dir = manifest_path.parent if manifest_path.parent.name != "report" else manifest_path.parent.parent
    return {
        "snapshot": man["snapshot"],
        "timeframe": man["timeframe"],
        "symbols": tuple(man.get("symbols") or []),
        "features_path": Path(man["features_path"]),
        "labels_path": Path(man["labels_path"]),
        "label_col": man["params"]["label_col"],
        "model": man.get("model") or man.get("model_kind"),
        "run_dir": run_dir,
        "report_dir": run_dir / "report",
    }
