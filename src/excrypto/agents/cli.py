# src/excrypto/agents/cli.py
import typer
from excrypto.agents.runner import run_daily, run_range

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