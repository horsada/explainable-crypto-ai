# src/excrypto/viz/cli.py
from __future__ import annotations
import typer
from pathlib import Path
import pandas as pd
from excrypto.viz.api import (
    plot_feature_correlation, plot_label_balance, plot_roc_pr,
    plot_threshold_sweep, plot_equity_curve, write_report_links, from_train_manifest
)
from excrypto.viz.raw import price_series, volume_series, returns_hist, rolling_vol, missing_heatmap
from excrypto.viz.features import histograms, rolling_feature, corr_heatmap, feature_target_corr
from excrypto.utils.paths import RunPaths

app = typer.Typer(help="Visualization helpers")

@app.command("raw")
def raw(
    panel: Path = typer.Argument(..., exists=True, dir_okay=False),
    out_dir: Path = typer.Option(...),
    window: int = 24
):
    out_dir.mkdir(parents=True, exist_ok=True)
    price_series(panel, out_dir)
    volume_series(panel, out_dir)
    returns_hist(panel, out_dir)
    rolling_vol(panel, out_dir, window=window)
    missing_heatmap(panel, out_dir)
    typer.echo(f"✅ Wrote raw data plots to {out_dir}")

@app.command("features")
def features(
    features_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    out_dir: Path = typer.Option(...),
    labels_path: Path | None = typer.Option(None, exists=True, dir_okay=False),
    label_col: str | None = typer.Option(None),
    rolling_feat: str | None = typer.Option(None),
    rolling_window: int = 48
):
    out_dir.mkdir(parents=True, exist_ok=True)
    histograms(features_path, out_dir)
    corr_heatmap(features_path, out_dir)
    if rolling_feat:
        rolling_feature(features_path, out_dir, rolling_feat, rolling_window)
    if labels_path and label_col:
        feature_target_corr(features_path, labels_path, label_col, out_dir)
    typer.echo(f"✅ Wrote feature plots to {out_dir}")

@app.command("from-train")
def from_train(manifest: Path = typer.Argument(..., exists=True, dir_okay=False)):
    ctx = from_train_manifest(manifest)
    rdir = ctx["report_dir"]; rdir.mkdir(parents=True, exist_ok=True)
    # feature corr
    p1 = plot_feature_correlation(ctx["features_path"], rdir)
    # label balance
    p2 = plot_label_balance(ctx["labels_path"], ctx["label_col"], rdir)
    # ROC/PR + threshold sweep (if predict scores present)
    score_path = (ctx["run_dir"] / "scores.parquet")
    p_roc = p_pr = p_thr = None
    if score_path.exists():
        df = pd.read_parquet(score_path)  # expected: timestamp,symbol,score,label
        p_roc, p_pr = plot_roc_pr(df["label"], df["score"].values, rdir)
        p_thr = plot_threshold_sweep(df["label"], df["score"].values, rdir)
    # equity curve (if signals + panel exist)
    panel = ctx["run_dir"] / "panel.parquet"
    signals = ctx["run_dir"] / "signals.parquet"
    p_eq = None
    if panel.exists() and signals.exists():
        p_eq = plot_equity_curve(panel, signals, rdir)
    # mini report
    items = {"Features corr": p1, "Label balance": p2}
    if p_roc: 
        items["ROC"] = p_roc
    if p_pr:  
        items["PR"] = p_pr
    if p_thr: 
        items["Threshold sweep"] = p_thr
    if p_eq:  
        items["Equity curve"] = p_eq

    # use the run’s actual report dir from ctx
    write_report_links(ctx["report_dir"], items)
    typer.echo(f"✔ wrote plots to {rdir}")
