# src/excrypto/utils/loaders.py
from pathlib import Path
import pandas as pd
from excrypto.utils.registry import find  # the registry you built

def load_snapshot(snapshot_id: str, symbols: list[str] | None = None) -> pd.DataFrame:
    recs = find(snapshot_id=snapshot_id)
    if recs.empty:
        raise FileNotFoundError(f"No registry entries for {snapshot_id}")
    if symbols:
        recs = recs[recs["symbol"].isin(symbols)]
        if recs.empty:
            raise FileNotFoundError(f"No symbols {symbols} for {snapshot_id}")

    parts = []
    for _, r in recs.iterrows():
        df = pd.read_parquet(Path(r["path"]))
        # ensure standard cols
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["symbol"] = r["symbol"]
        parts.append(df[["timestamp","symbol","close","open","high","low","volume"]])

    panel = (pd.concat(parts, ignore_index=True)
             .drop_duplicates(subset=["timestamp","symbol"])
             .sort_values(["timestamp","symbol"]))
    return panel.set_index("timestamp")
