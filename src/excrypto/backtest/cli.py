# src/excrypto/backtest/cli.py
from __future__ import annotations
import typer
import pandas as pd
from pathlib import Path
from excrypto.backtest.engine import BacktestConfig, backtest_single, backtest_multi

app = typer.Typer(help="Backtest engine CLI (prices+signals → PnL)")

def _cfg(
    fee_bps: float,
    slippage_bps: float,
    latency_bars: int,
    target_vol_ann: float,
    max_leverage: float,
    vol_lookback: int,
) -> BacktestConfig:
    return BacktestConfig(
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        latency_bars=latency_bars,
        target_vol_ann=target_vol_ann,
        max_leverage=max_leverage,
        vol_lookback=vol_lookback,
    )

@app.command("single")
def run_single(
    data_path: str = typer.Argument(..., help="Parquet/CSV with DatetimeIndex or 'timestamp' col"),
    price_col: str = "close",
    signal_col: str = "signal",
    out_path: str = typer.Option("backtest_single.parquet", help="Output Parquet"),
    fee_bps: float = 1.0,
    slippage_bps: float = 1.0,
    latency_bars: int = 1,
    target_vol_ann: float = 0.20,
    max_leverage: float = 3.0,
    vol_lookback: int = 60,
):
    """Backtest a **single** asset time series."""
    df = _read_any(data_path)
    bt = backtest_single(
        df[[price_col, signal_col]],
        _cfg(fee_bps, slippage_bps, latency_bars, target_vol_ann, max_leverage, vol_lookback),
        price_col=price_col,
        signal_col=signal_col,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    bt.to_parquet(out_path)
    typer.echo(f"Wrote {out_path}")

@app.command("multi")
def run_multi(
    data_path: str = typer.Argument(..., help="Parquet/CSV long panel with 'symbol' column"),
    price_col: str = "close",
    signal_col: str = "signal",
    out_path: str = typer.Option("backtest_multi.parquet", help="Output Parquet"),
    fee_bps: float = 1.0,
    slippage_bps: float = 1.0,
    latency_bars: int = 1,
    target_vol_ann: float = 0.20,
    max_leverage: float = 3.0,
    vol_lookback: int = 60,
):
    """Backtest a **multi-asset** panel (index=time, columns include 'symbol')."""
    panel = _read_any(data_path)
    if "symbol" not in panel.columns:
        raise typer.BadParameter("Panel must include a 'symbol' column for multi-asset backtest.")
    bt = backtest_multi(
        panel[[price_col, signal_col, "symbol"]],
        _cfg(fee_bps, slippage_bps, latency_bars, target_vol_ann, max_leverage, vol_lookback),
        price_col=price_col,
        signal_col=signal_col,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    bt.to_parquet(out_path)
    typer.echo(f"Wrote {out_path}")

def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise typer.BadParameter(f"File not found: {path}")
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        raise typer.BadParameter("Only .parquet or .csv supported.")
    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        ts_col = "timestamp" if "timestamp" in df.columns else None
        if ts_col is None:
            raise typer.BadParameter("Provide a DatetimeIndex or a 'timestamp' column.")
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        df = df.set_index(ts_col).sort_index()
    if df.index.tz is None:
        # assume UTC if naive
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()

# Optional: expose this sub-CLI from the root app:
# from excrypto.backtest.cli import app as backtest_app
# app.add_typer(backtest_app, name="backtest")
