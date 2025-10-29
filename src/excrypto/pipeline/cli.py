import typer
import pandas as pd, json, os
from excrypto.pipeline.snapshot import run_day, run_range, SnapArgs
from excrypto.pipeline.splits import build_rolling_splits

app = typer.Typer(help="Data pipeline: snapshot → features → splits")


@app.command("snapshot")
def snapshot(
    snapshot: str = typer.Option(None, help="YYYY-MM-DD (UTC)"),
    start: str = typer.Option(None, help="Range start YYYY-MM-DD (UTC)"),
    end: str = typer.Option(None, help="Range end YYYY-MM-DD (UTC)"),
    exchange: str = "binance",
    symbols: str = "BTC/USDT,ETH/USDT",
    timeframe: str = "1m",
    ohlcv_limit: int = 1000,
    data_root: str = "data/raw",
):
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    args = SnapArgs(exchange=exchange, symbols=syms, timeframe=timeframe, limit=ohlcv_limit, data_root=data_root)

    # mutually exclusive
    if snapshot and (start or end):
        raise typer.BadParameter("Use either --snapshot or (--start and --end), not both.")
    if start or end:
        if not (start and end):
            raise typer.BadParameter("Provide both --start and --end for ranges.")
        out = run_range(start, end, args=args)
        typer.echo(f"Wrote combined range → {out}/{timeframe}/{syms}")
    elif snapshot:
        out = run_day(snapshot=snapshot, exchange=exchange, symbols=syms)  # your existing function
        typer.echo(f"Wrote daily snapshot → {out}")
    else:
        raise typer.BadParameter("Provide --snapshot or a --start/--end range.")


@app.command()
def splits(
    snapshot: str = typer.Option(..., help="For locating processed data if needed"),
    train: str = "60D",
    valid: str = "30D",
    step: str = "30D",
    embargo: str = "1D",
):
    # Example: build splits over any df that has a DatetimeIndex
    # Replace with your actual feature load:
    df = pd.DataFrame(index=pd.date_range("2025-01-01", periods=1000, freq="T", tz="UTC"))
    folds = build_rolling_splits(df, train=train, valid=valid, step=step, embargo=embargo)
    os.makedirs("splits", exist_ok=True)
    with open(f"splits/{snapshot}_rolling.json","w") as f:
        json.dump([{"train": [int(i) for i in s.train_idx],
                    "valid": [int(i) for i in s.valid_idx]} for s in folds], f, indent=2)
    typer.echo(f"Wrote splits/{snapshot}_rolling.json")
