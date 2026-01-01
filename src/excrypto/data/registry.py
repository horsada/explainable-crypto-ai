# src/excrypto/data/registry.py
from __future__ import annotations

import pandas as pd
from pathlib import Path

REG_PATH = Path("data/registry/raw_market.parquet")

_UNIQUE_KEYS = ["kind", "snapshot_id", "exchange", "symbol", "timeframe"]
_META_COLS = ["rows", "first_ts", "last_ts", "created_utc"]


def _load_registry() -> pd.DataFrame:
    if not REG_PATH.exists():
        return pd.DataFrame(columns=_UNIQUE_KEYS + _META_COLS)

    df = pd.read_parquet(REG_PATH)

    # Back-compat migrations (old registries may have 'path' etc.)
    if "kind" not in df.columns:
        df["kind"] = "ohlcv"
    if "timeframe" not in df.columns:
        df["timeframe"] = "unknown"

    # ensure expected columns exist
    for c in _UNIQUE_KEYS + _META_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    # drop deprecated columns to keep it clean (optional, but recommended)
    keep = _UNIQUE_KEYS + _META_COLS
    df = df[keep].copy()

    return df


def upsert_record(rec: dict) -> None:
    """
    rec must include:
      kind, snapshot_id, exchange, symbol, timeframe,
      rows, first_ts, last_ts, created_utc
    """
    df = _load_registry()
    rec = dict(rec)
    rec.setdefault("kind", "ohlcv")

    for k in _UNIQUE_KEYS:
        if k not in rec:
            raise ValueError(f"upsert_record: missing required field '{k}'")

    # ensure meta cols exist
    for c in _META_COLS:
        rec.setdefault(c, pd.NA)

    m = pd.Series(True, index=df.index)
    for k in _UNIQUE_KEYS:
        m &= (df[k] == rec[k])

    df = df.loc[~m]
    df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)

    REG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(REG_PATH, index=False)


def find(kind=None, snapshot_id=None, exchange=None, symbol=None, timeframe=None) -> pd.DataFrame:
    df = _load_registry()
    filters = {
        "kind": kind,
        "snapshot_id": snapshot_id,
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
    }
    for k, v in filters.items():
        if v is not None:
            df = df[df[k] == v]
    return df.sort_values(_UNIQUE_KEYS)
