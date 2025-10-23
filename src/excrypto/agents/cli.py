# src/excrypto/agents/cli.py
import typer
from pathlib import Path
import pandas as pd
from excrypto.agents.runner import run_daily, run_range
from excrypto.agents.catalog import summarize

app = typer.Typer(help="Agentic workflows")

@app.command("daily")
def daily(
    snapshot: str = typer.Option("", help="YYYY-MM-DD UTC; empty = today"),
    symbols: str = "BTC/USDT,ETH/USDT",
    exchange: str = "binance",
):
    run_daily(snapshot or None, symbols, exchange)

@app.command("range")
def range(
    start: str = typer.Option("", help="YYYY-MM-DD UTC;"),
    end: str = typer.Option("", help="YYYY-MM-DD UTC (inclusive)"),
    timeframe: str = '1m',
    symbols: str = "BTC/USDT,ETH/USDT",
    exchange: str = "binance",
):
    run_range(start, end, timeframe, symbols, exchange)


@app.command("catalog")
def catalog(snapshot: str = "", out_dir: str = "runs/data_audit"):
    df = summarize(snapshot or None)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p_csv = Path(out_dir)/"catalog.csv"
    p_md  = Path(out_dir)/"catalog.md"
    df.to_csv(p_csv, index=False)
    p_md.write_text(df.to_markdown(index=False))
    typer.echo(f"Wrote {p_csv} and {p_md}")
