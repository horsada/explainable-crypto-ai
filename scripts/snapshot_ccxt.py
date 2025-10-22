# file: scripts/snapshot_ccxt.py
import argparse, os, json, ccxt, pandas as pd
from datetime import datetime, timezone

def fetch_ohlcv(ex, sym, tf="1m", limit=10_000):
    cols = ["timestamp","open","high","low","close","volume"]
    df = pd.DataFrame(ex.fetch_ohlcv(sym, timeframe=tf, limit=limit), columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

def fetch_funding(ex, sym, limit=1000):
    try: raw = ex.fetch_funding_rate_history(sym, limit=limit)
    except Exception: return pd.DataFrame()
    if not raw: return pd.DataFrame()
    df = pd.DataFrame(raw)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.rename(columns={"fundingRate":"funding"})[["timestamp","funding"]]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)           # e.g. 2025-10-21 (UTC)
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--symbols", required=True)            # CSV: BTC/USDT,ETH/USDT
    ap.add_argument("--timeframe", default="1m")
    args = ap.parse_args()

    root = f"data/raw/{args.snapshot}/{args.exchange}"
    os.makedirs(root, exist_ok=True)

    ex = getattr(ccxt, args.exchange)({"enableRateLimit": True})
    universe = []
    for sym in [s.strip() for s in args.symbols.split(",")]:
        pair_dir = f"{root}/{sym.replace('/','_')}"
        os.makedirs(pair_dir, exist_ok=True)
        fetch_ohlcv(ex, sym, args.timeframe).to_parquet(f"{pair_dir}/ohlcv.parquet")
        fr = fetch_funding(ex, sym)
        if not fr.empty: fr.to_parquet(f"{pair_dir}/funding.parquet")
        universe.append(sym)

    meta = {
        "snapshot_utc": args.snapshot,
        "time_captured_utc": datetime.now(timezone.utc).isoformat(),
        "exchange": args.exchange,
        "symbols": universe,
        "ccxt_version": ccxt.__version__,
    }
    with open(f"{root}/_snapshot_meta.json","w") as f: json.dump(meta, f, indent=2)
    with open(f"{root}/_universe.json","w") as f: json.dump(universe, f, indent=2)
