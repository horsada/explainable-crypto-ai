# src/excrypto/utils/registry.py
from __future__ import annotations
import pandas as pd, json
from pathlib import Path

REG_PATH = Path("data/registry/ohlcv.parquet")

_UNIQUE_KEYS = ["snapshot_id", "exchange", "symbol", "timeframe"]

def _load_registry() -> pd.DataFrame:
    if not REG_PATH.exists():
        return pd.DataFrame(columns=_UNIQUE_KEYS + ["path","rows","first_ts","last_ts","created_utc"])
    df = pd.read_parquet(REG_PATH)
    # migrate if timeframe missing
    if "timeframe" not in df.columns:
        df["timeframe"] = "unknown"
    # ensure all expected columns exist
    for c in ["path","rows","first_ts","last_ts","created_utc"]:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def upsert_record(rec: dict):
    """
    rec must include: snapshot_id, exchange, symbol, timeframe, path, rows, first_ts, last_ts, created_utc
    - Enforces single-exchange per snapshot_id.
    - Upserts by (snapshot_id, exchange, symbol, timeframe).
    """
    df = _load_registry()

    # single-exchange per snapshot
    exs = df.loc[df.snapshot_id == rec["snapshot_id"], "exchange"].dropna().unique()
    if len(exs) and rec["exchange"] not in exs:
        raise ValueError(
            f"Snapshot {rec['snapshot_id']} already registered for exchange {exs[0]}; "
            f"got {rec['exchange']}. Per-snapshot must be single exchange."
        )

    # upsert by full key (includes timeframe)
    m = pd.Series(True, index=df.index)
    for k in _UNIQUE_KEYS:
        if k not in rec:
            raise ValueError(f"upsert_record: missing required field '{k}'")
        m &= (df[k] == rec[k])

    df = df.loc[~m]  # drop existing match
    df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)

    REG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(REG_PATH, index=False)

def find(snapshot_id=None, exchange=None, symbol=None, timeframe=None) -> pd.DataFrame:
    df = _load_registry()
    filters = {
        "snapshot_id": snapshot_id,
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
    }
    for k, v in filters.items():
        if v is not None:
            df = df[df[k] == v]
    return df.sort_values(_UNIQUE_KEYS)
