# src/excrypto/backtest/cli.py
from __future__ import annotations

from pathlib import Path
from typing import Dict

import typer
import yaml

from excrypto.backtest.engine import BacktestConfig, backtest_single, backtest_multi
from excrypto.backtest.io import load_trade_frame, resolve_inputs
from excrypto.backtest.metrics import summarize
from excrypto.backtest.writer import write_backtest_artifact
from excrypto.utils.paths import RunPaths

app = typer.Typer(help="Backtest engine (prices + signals → PnL) using RunPaths and YAML config")


def _parse_params(s: str | None) -> Dict[str, str]:
    if not s:
        return {}
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out: Dict[str, str] = {}
    for kv in parts:
        if "=" not in kv:
            raise typer.BadParameter(f"Bad params item '{kv}', expected key=value")
        k, v = kv.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _load_engine_dict(config_path: str) -> dict:
    p = Path(config_path)
    if not p.exists():
        raise typer.BadParameter(f"Config not found: {config_path}")
    cfg = yaml.safe_load(p.read_text()) or {}

    # support both:
    # - config/backtest.yaml -> engine: {...}
    # - agents/full.yaml     -> backtest: { engine: {...} }
    eng = (cfg.get("backtest") or {}).get("engine") or cfg.get("engine") or {}
    if not isinstance(eng, dict):
        raise typer.BadParameter("Engine config must be a dict.")
    return eng


def _load_engine(config_path: str) -> BacktestConfig:
    eng = _load_engine_dict(config_path)
    return BacktestConfig(
        fee_bps=float(eng.get("fee_bps", 1.0)),
        slippage_bps=float(eng.get("slippage_bps", 1.0)),
        latency_bars=int(eng.get("latency_bars", 1)),
        target_vol_ann=float(eng.get("target_vol_ann", 0.20)),
        max_leverage=float(eng.get("max_leverage", 3.0)),
        vol_lookback=int(eng.get("vol_lookback", 60)),
    )


@app.command("run")
def run(
    snapshot: str = typer.Argument(..., help="Snapshot id (e.g. 2018-01-01_to_2018-12-31)"),
    symbols: str = typer.Argument(..., help="CSV symbols, e.g. 'BTC/USDT,ETH/USDT'"),
    exchange: str = typer.Option("binance", help="Exchange label"),
    timeframe: str = typer.Option("1h", help="Timesteps"),
    runs_root: str = typer.Option("runs", help="Runs root folder"),

    # where to read from
    signals_strategy: str = typer.Option("predict", help="RunPaths strategy for signals"),
    prices_strategy: str = typer.Option("snapshot", help="RunPaths strategy for prices panel"),
    signals_params: str = typer.Option("", help="Extra params for signals run (key=val,...)"),
    prices_params: str = typer.Option("", help="Extra params for prices run (key=val,...)"),
    signals_path: str | None = typer.Option(None, help="Override signals.parquet path"),
    prices_path: str | None = typer.Option(None, help="Override prices panel.parquet path"),

    # engine / columns
    config: str = typer.Option("config/backtest.yaml", help="YAML with engine params"),
    price_col: str = typer.Option("close"),
    signal_col: str = typer.Option("signal"),
):
    syms = tuple(s.strip() for s in symbols.split(",") if s.strip())
    if not syms:
        raise typer.BadParameter("No symbols provided.")

    runs_root_p = Path(runs_root)

    engine = _load_engine(config)
    engine_dict = _load_engine_dict(config)

    inp = resolve_inputs(
        snapshot=snapshot,
        symbols=syms,
        timeframe=timeframe,
        runs_root=runs_root_p,
        exchange=exchange,
        signals_strategy=signals_strategy,
        prices_strategy=prices_strategy,
        signals_path=Path(signals_path) if signals_path else None,
        prices_path=Path(prices_path) if prices_path else None,
    )

    trade = load_trade_frame(
        prices_path=inp.prices_path,
        signals_path=inp.signals_path,
        price_col=price_col,
        signal_col=signal_col,
    )

    is_multi = "symbol" in trade.columns and trade["symbol"].nunique() > 1

    if is_multi:
        bt = backtest_multi(
            trade[[price_col, signal_col, "symbol"]],
            engine,
            price_col=price_col,
            signal_col=signal_col,
        )
    else:
        if "symbol" in trade.columns:
            trade = trade.drop(columns=["symbol"])
        bt = backtest_single(
            trade[[price_col, signal_col]],
            engine,
            price_col=price_col,
            signal_col=signal_col,
        )

    summary = summarize(bt)

    runpaths = RunPaths(
        snapshot=snapshot,
        strategy="backtest",
        symbols=syms,
        timeframe=timeframe,
        params={"exchange": exchange, "src": signals_strategy},
        runs_root=runs_root_p,
    )

    artifact = write_backtest_artifact(
        runpaths,
        bt,
        summary=summary,
        inputs={
            "prices_path": str(inp.prices_path),
            "signals_path": str(inp.signals_path),
            "signals_strategy": signals_strategy,
            "prices_strategy": prices_strategy,
            "price_col": price_col,
            "signal_col": signal_col,
        },
        engine=engine_dict,
    )

    typer.echo(f"Wrote {artifact.backtest_path}")
    typer.echo(f"Wrote {artifact.summary_path}")
    typer.echo(f"Wrote {artifact.manifest_path}")
