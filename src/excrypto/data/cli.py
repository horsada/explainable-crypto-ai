# src/excrypto/data/cli.py
import typer
from pathlib import Path

from excrypto.data.snapshot import build_snapshot, SnapshotConfig
from excrypto.data.panel import build_and_write_panel

app = typer.Typer(help="Data pipeline: raw snapshots + helpers")


@app.command("snapshot")
def snapshot(
    start: str = typer.Option(..., help="YYYY-MM-DD (UTC)"),
    end: str = typer.Option(..., help="YYYY-MM-DD (UTC), inclusive"),
    exchange: str = typer.Option("binance"),
    symbols: str = typer.Option(..., help="CSV, e.g. BTC/USDT,ETH/USDT"),
    timeframe: str = typer.Option("1m"),
    ohlcv_limit: int = typer.Option(1000, help="Per API call; paging will be used"),
    funding_limit: int = typer.Option(1000),
    data_root: str = typer.Option("data/raw"),
):
    syms = tuple(s.strip() for s in symbols.split(",") if s.strip())
    if not syms:
        raise typer.BadParameter("--symbols must contain at least one symbol")

    cfg = SnapshotConfig(
        exchange=exchange,
        symbols=syms,
        timeframe=timeframe,
        ohlcv_limit=ohlcv_limit,
        funding_limit=funding_limit,
        root=typer.get_app_dir if False else __import__("pathlib").Path(data_root),  # avoid extra import noise
    )

    res = build_snapshot(cfg, start=start, end=end)
    typer.echo(str(res.root))


@app.command("panel")
def panel(
    snapshot: str = typer.Option(..., help="snapshot_id, e.g. 2018-01-01_to_2018-12-31"),
    symbols: str = typer.Option(..., help="CSV, e.g. BTC/USDT,ETH/USDT"),
    exchange: str = typer.Option("binance"),
    timeframe: str = typer.Option("1h"),
    runs_root: Path = typer.Option(Path("runs")),
):
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    if not syms:
        raise typer.BadParameter("--symbols must contain at least one symbol")

    art = build_and_write_panel(
        snapshot=snapshot,
        symbols=syms,
        exchange=exchange,
        timeframe=timeframe,
        runs_root=runs_root,
    )
    typer.echo(str(art.panel_path))
