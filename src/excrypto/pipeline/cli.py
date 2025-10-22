import typer
import pandas as pd, json, os
from excrypto.pipeline.snapshot import run as snapshot_run
from excrypto.pipeline.features import run as features_run
from excrypto.pipeline.splits import build_rolling_splits

app = typer.Typer(help="Data pipeline: snapshot → features → splits")

@app.command()
def snapshot(
    snapshot: str = typer.Option(..., help="UTC date label, e.g. 2025-10-21"),
    exchange: str = "binance",
    symbols: str = typer.Option("BTC/USDT,ETH/USDT", help="CSV list"),
):
    snapshot_run(snapshot=snapshot, exchange=exchange, symbols=symbols)

@app.command()
def features(snapshot: str = typer.Option(..., help="UTC date label"), exchange: str = "binance", symbols: str = "", labels: bool = False):
    syms = [s.strip() for s in symbols.split(",")] if symbols else None
    out = features_run(snapshot=snapshot, exchange=exchange, symbols=syms, compute_label=labels)
    typer.echo(f"Wrote {out}")


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
