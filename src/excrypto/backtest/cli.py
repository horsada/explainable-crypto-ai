# src/excrypto/backtest/cli.py
from __future__ import annotations
import typer, yaml
import pandas as pd
from pathlib import Path
from typing import Dict
from excrypto.backtest.engine import BacktestConfig, backtest_single, backtest_multi
from excrypto.utils.paths import RunPaths

app = typer.Typer(help="Backtest engine (panel → PnL) using RunPaths and YAML config")

def _read_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Panel not found at {path}. Run your baseline first.")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df

def _parse_params(s: str | None) -> Dict[str, str]:
    if not s: return {}
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: Dict[str, str] = {}
    for kv in parts:
        if "=" not in kv:
            raise typer.BadParameter(f"Bad params item '{kv}', expected key=value")
        k, v = kv.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def _load_engine(config_path: str) -> BacktestConfig:
    p = Path(config_path)
    if not p.exists():
        raise typer.BadParameter(f"Config not found: {config_path}")
    cfg = yaml.safe_load(p.read_text()) or {}
    eng = cfg.get("engine") or {}
    # only pass known fields
    return BacktestConfig(
        fee_bps        = float(eng.get("fee_bps", 1.0)),
        slippage_bps   = float(eng.get("slippage_bps", 1.0)),
        latency_bars   = int(eng.get("latency_bars", 1)),
        target_vol_ann = float(eng.get("target_vol_ann", 0.20)),
        max_leverage   = float(eng.get("max_leverage", 3.0)),
        vol_lookback   = int(eng.get("vol_lookback", 60)),
    )

@app.command("run")
def run(
    snapshot: str = typer.Argument(..., help="Registry snapshot_id (e.g. 2025-10-22 or COMBINED_...)"),
    strategy: str = typer.Argument(..., help="Strategy name used by baselines (e.g. 'momentum','hodl')"),
    symbols: str = typer.Argument(..., help="CSV symbols, e.g. 'BTC/USDT,ETH/USDT'"),
    params: str = typer.Option("", help="Optional strategy params key=val,key=val to select the right run folder"),
    config: str = typer.Option("config/backtest.yaml", help="YAML with engine: {fee_bps,...}"),
    out_path: str | None = typer.Option(None, help="Override output path (defaults to RunPaths.backtest)"),
    price_col: str = "close",
    signal_col: str = "signal",
):
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    param_dict = _parse_params(params) or None

    # Locate artifacts via RunPaths
    paths = RunPaths(snapshot=snapshot, strategy=strategy, symbols=tuple(syms), params=param_dict)
    panel_path = paths.panel
    df = _read_panel(panel_path)

    # Sanity
    for c in [price_col, signal_col]:
        if c not in df.columns:
            raise typer.BadParameter(f"Missing column '{c}' in {panel_path}")

    # Engine from YAML
    engine = _load_engine(config)

    # Decide single vs multi
    is_multi = "symbol" in df.columns and df["symbol"].nunique() > 1

    # Output path
    out_file = Path(out_path) if out_path else paths.backtest
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Run
    if is_multi:
        bt = backtest_multi(df[[price_col, signal_col, "symbol"]], engine,
                            price_col=price_col, signal_col=signal_col)
    else:
        if "symbol" in df.columns:
            df = df.drop(columns=["symbol"])
        bt = backtest_single(df[[price_col, signal_col]], engine,
                             price_col=price_col, signal_col=signal_col)

    bt.to_parquet(out_file)
    typer.echo(f"Wrote {out_file}")
