# src/excrypto/risk/cli.py
from __future__ import annotations
import typer
import pandas as pd
from excrypto.utils.paths import RunPaths
from excrypto.risk.report import write_risk_report_md

app = typer.Typer(help="Risk reports from PnL series (RunPaths-native)")

def _parse_symbols(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def _parse_params(s: str | None) -> dict | None:
    if not s:
        return None
    out = {}
    for kv in (p.strip() for p in s.split(",") if p.strip()):
        k, v = kv.split("=", 1) if "=" in kv else (kv, "")
        out[k.strip()] = v.strip()
    return out or None

@app.command("report")
def report(
    snapshot: str = typer.Argument(..., help="Snapshot id (e.g. 2025-10-22 or COMBINED_...)"),
    strategy: str = typer.Argument(..., help="Strategy name (e.g. momentum, hodl)"),
    symbols: str = typer.Argument(..., help="CSV symbols (e.g. BTC/USDT,ETH/USDT)"),
    timeframe: str = typer.Option("1h", help="Timesteps"),
    params: str = typer.Option("", help="Strategy params key=val,key=val to select run folder"),
    pnl_col: str = typer.Option("pnl_net", help="P&L column to evaluate (falls back to equity pct-change)"),
    title: str = typer.Option("", help="Report title (defaults to '<strategy> | <snapshot>')"),
):
    syms = _parse_symbols(symbols)
    param_dict = _parse_params(params)

    paths = RunPaths(snapshot=snapshot, strategy=strategy, symbols=tuple(syms), timeframe=timeframe, params=param_dict)
    paths.ensure()

    bt = pd.read_parquet(paths.backtest)

    # Ensure DatetimeIndex if timestamp column present
    if "timestamp" in bt.columns and not isinstance(bt.index, pd.DatetimeIndex):
        bt["timestamp"] = pd.to_datetime(bt["timestamp"], utc=True, errors="coerce")
        if bt["timestamp"].notna().any():
            bt = bt.set_index("timestamp").sort_index()

    # PnL column or derive from equity
    if pnl_col not in bt.columns:
        if "equity" in bt.columns:
            bt[pnl_col] = bt["equity"].pct_change().fillna(0.0)
        else:
            raise typer.BadParameter(f"Need '{pnl_col}' or 'equity' in {paths.backtest}")

    report_title = title or f"{strategy} | {snapshot}"
    md = write_risk_report_md(
        returns=bt[pnl_col],
        weights=None,
        title=report_title,
        out_dir=str(paths.report_dir),
    )
    typer.echo(f"Wrote {md}")
