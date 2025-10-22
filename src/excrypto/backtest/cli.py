# src/excrypto/backtest/cli.py
from __future__ import annotations
import typer, yaml
import pandas as pd
from pathlib import Path
from excrypto.backtest.engine import BacktestConfig, backtest_single, backtest_multi
#from excrypto.runner.backtest_cli import run_from_config as run_full_config  # full pipeline runner

app = typer.Typer(help="Backtest engine CLI (prices+signals → PnL)")

def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise typer.BadParameter(f"File not found: {path}")
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p, parse_dates=["timestamp"], infer_datetime_format=True)
    else:
        raise typer.BadParameter("Only .parquet or .csv supported.")
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" not in df.columns:
            raise typer.BadParameter("Provide a DatetimeIndex or a 'timestamp' column.")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()

def _deep_update(base: dict, override: dict) -> dict:
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        elif v is not None:
            out[k] = v
    return out

def _merge_cfg(config_path: str | None, overrides: dict) -> dict:
    base = {}
    if config_path:
        with open(config_path) as f:
            base = yaml.safe_load(f) or {}
    return _deep_update(base, overrides)

def _mk_engine(cfg: dict) -> BacktestConfig:
    return BacktestConfig(**(cfg.get("engine") or {}))

@app.command("single")
def run_single(
    # params (can come from YAML or flags)
    data_path: str | None = typer.Option(None, help="Parquet/CSV with DatetimeIndex or 'timestamp'"),
    price_col: str = "close",
    signal_col: str = "signal",
    out_path: str = "backtest_single.parquet",
    # engine
    fee_bps: float = 1.0,
    slippage_bps: float = 1.0,
    latency_bars: int = 1,
    target_vol_ann: float = 0.20,
    max_leverage: float = 3.0,
    vol_lookback: int = 60,
    # yaml preload
    config: str | None = typer.Option(None, help="YAML to preload params"),
):
    cfg = _merge_cfg(config, dict(
        data_path=data_path, price_col=price_col, signal_col=signal_col, out_path=out_path,
        engine=dict(fee_bps=fee_bps, slippage_bps=slippage_bps, latency_bars=latency_bars,
                    target_vol_ann=target_vol_ann, max_leverage=max_leverage, vol_lookback=vol_lookback)
    ))
    if not cfg.get("data_path"):
        raise typer.BadParameter("data_path is required (flag or YAML).")
    df = _read_any(cfg["data_path"])
    for c in [cfg.get("price_col","close"), cfg.get("signal_col","signal")]:
        if c not in df.columns: raise typer.BadParameter(f"Missing column: {c}")
    bt = backtest_single(
        df[[cfg["price_col"], cfg["signal_col"]]],
        _mk_engine(cfg),
        price_col=cfg["price_col"],
        signal_col=cfg["signal_col"],
    )
    Path(cfg["out_path"]).parent.mkdir(parents=True, exist_ok=True)
    bt.to_parquet(cfg["out_path"])
    typer.echo(f"Wrote {cfg['out_path']}")

@app.command("multi")
def run_multi(
    data_path: str | None = typer.Option(None, help="Parquet/CSV long panel with 'symbol'"),
    price_col: str = "close",
    signal_col: str = "signal",
    out_path: str = "backtest_multi.parquet",
    fee_bps: float = 1.0,
    slippage_bps: float = 1.0,
    latency_bars: int = 1,
    target_vol_ann: float = 0.20,
    max_leverage: float = 3.0,
    vol_lookback: int = 60,
    config: str | None = typer.Option(None, help="YAML to preload params"),
):
    cfg = _merge_cfg(config, dict(
        data_path=data_path, price_col=price_col, signal_col=signal_col, out_path=out_path,
        engine=dict(fee_bps=fee_bps, slippage_bps=slippage_bps, latency_bars=latency_bars,
                    target_vol_ann=target_vol_ann, max_leverage=max_leverage, vol_lookback=vol_lookback)
    ))
    if not cfg.get("data_path"):
        raise typer.BadParameter("data_path is required (flag or YAML).")
    panel = _read_any(cfg["data_path"])
    for c in [cfg.get("price_col","close"), cfg.get("signal_col","signal"), "symbol"]:
        if c not in panel.columns: raise typer.BadParameter(f"Missing column: {c}")
    bt = backtest_multi(
        panel[[cfg["price_col"], cfg["signal_col"], "symbol"]],
        _mk_engine(cfg),
        price_col=cfg["price_col"],
        signal_col=cfg["signal_col"],
    )
    Path(cfg["out_path"]).parent.mkdir(parents=True, exist_ok=True)
    bt.to_parquet(cfg["out_path"])
    typer.echo(f"Wrote {cfg['out_path']}")

"""
@app.command("from-config")
def from_config(config: str = typer.Argument(..., help="conf/backtest.yaml")):
    out = run_full_config(config)
    typer.echo(f"Artifacts → {out}")
"""