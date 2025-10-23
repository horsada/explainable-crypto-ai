# src/excrypto/utils/registry.py
import pandas as pd, json
from pathlib import Path

REG_PATH = Path("data/registry/ohlcv.parquet")

def upsert_record(rec: dict):
    REG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if REG_PATH.exists():
        df = pd.read_parquet(REG_PATH)
        df = df[~((df.snapshot_id==rec["snapshot_id"]) &
                  (df.exchange==rec["exchange"]) &
                  (df.symbol==rec["symbol"]))]
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
