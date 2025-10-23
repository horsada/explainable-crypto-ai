import typer
import pandas as pd
from pathlib import Path
from excrypto.baseline import momentum, hodl, vt_hodl  # your momentum.py
from excrypto.utils import loader
app = typer.Typer(help="Baselines: generate signals/panels")


## deprecated
def _load_panel(snapshot:str, exchange:str, symbols:list[str], data_root="data/raw")->pd.DataFrame:
    rows=[]
    base = Path(data_root)/snapshot/exchange
    for sym in symbols:
        p = base/sym.replace("/","_")/"ohlcv.parquet"
        df = pd.read_parquet(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["symbol"] = sym
        rows.append(df[["timestamp","symbol","close"]])
    return pd.concat(rows).sort_values("timestamp").set_index("timestamp")

@app.command("momentum")
def momentum_signals(
    snapshot: str = typer.Option(...),
    symbols: str = typer.Option("BTC/USDT,ETH/USDT", help="CSV"),
    fast: int = 20,
    slow: int = 60,
    out_dir: str = "runs/momentum",
    write_panel: bool = True,
):
    syms = [s.strip() for s in symbols.split(",")]
    panel = loader.load_snapshot(snapshot, syms)
    # build signals per symbol (PIT-safe inside momentum)
    sigs = []
    for sym, g in panel.groupby("symbol"):
        s = momentum._sma_sig(g["close"], fast=fast, slow=slow).rename("signal")
        sigs.append(pd.DataFrame({"timestamp": s.index, "symbol": sym, "signal": s.values}))
    signals = pd.concat(sigs).sort_values(["timestamp","symbol"])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    signals.to_parquet(Path(out_dir)/"signals.parquet", index=False)
    if write_panel:
        merged = (panel.reset_index()[["timestamp","symbol","close"]]
                  .merge(signals, on=["timestamp","symbol"], how="inner"))
        merged.to_parquet(Path(out_dir)/"panel.parquet", index=False)
    typer.echo(f"Wrote {out_dir}/signals.parquet" + (" and panel.parquet" if write_panel else ""))

@app.command("hodl")
def hodl_signals(
    snapshot: str = typer.Option(...),
    symbols: str = typer.Option("BTC/USDT,ETH/USDT"),
    out_dir: str = "runs/hodl",
    write_panel: bool = True,
):
    syms = [s.strip() for s in symbols.split(",")]
    panel = loader.load_snapshot(snapshot, syms)
    # constant +1
    sig = pd.DataFrame({"timestamp": panel.index, "symbol": panel["symbol"].values, "signal": 1.0})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sig.to_parquet(Path(out_dir)/"signals.parquet", index=False)
    if write_panel:
        merged = (panel.reset_index()[["timestamp","symbol","close"]]
                  .merge(sig, on=["timestamp","symbol"], how="inner"))
        merged.to_parquet(Path(out_dir)/"panel.parquet", index=False)
    typer.echo(f"Wrote {out_dir}/signals.parquet" + (" and panel.parquet" if write_panel else ""))
