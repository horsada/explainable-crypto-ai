# src/excrypto/utils/registry.py
import pandas as pd, json
from pathlib import Path

REG_PATH = Path("data/registry/ohlcv.parquet")

def upsert_record(rec: dict):
    # assert single-exchange per snapshot
    if REG_PATH.exists():
        df = pd.read_parquet(REG_PATH)
        exs = df.loc[df.snapshot_id == rec["snapshot_id"], "exchange"].dropna().unique()
        if len(exs) and rec["exchange"] not in exs:
            raise ValueError(f"Snapshot {rec['snapshot_id']} already registered for exchange {exs[0]}; "
                             f"got {rec['exchange']}. Per-snapshot must be single exchange.")
        # upsert by (snapshot_id, symbol)
        m = (df["snapshot_id"]==rec["snapshot_id"]) & (df["symbol"]==rec["symbol"])
        df = df.loc[~m]
        df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    else:
        df = pd.DataFrame([rec])
    df.to_parquet(REG_PATH, index=False)


def find(snapshot_id=None, exchange=None, symbol=None, timeframe=None) -> pd.DataFrame:
    if not REG_PATH.exists(): 
        return pd.DataFrame()
    df = pd.read_parquet(REG_PATH)
    for k,v in {"snapshot_id":snapshot_id,"exchange":exchange,"symbol":symbol,"timeframe":timeframe}.items():
        if v: df = df[df[k]==v]
    return df.sort_values(["snapshot_id","symbol"])
