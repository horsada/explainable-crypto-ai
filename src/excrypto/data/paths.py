# src/excrypto/data/paths.py
from __future__ import annotations
from pathlib import Path

def slug_symbol(symbol: str) -> str:
    return symbol.replace("/", "_")

def raw_market_dir(raw_root: Path, snapshot_id: str, exchange: str, timeframe: str) -> Path:
    return raw_root / snapshot_id / exchange / timeframe

def raw_market_path(
    raw_root: Path,
    snapshot_id: str,
    exchange: str,
    timeframe: str,
    symbol: str,
    kind: str = "ohlcv",
) -> Path:
    """
    data/raw/<snapshot_id>/<exchange>/<timeframe>/<SYMBOL_>/<kind>.parquet
    """
    return raw_market_dir(raw_root, snapshot_id, exchange, timeframe) / slug_symbol(symbol) / f"{kind}.parquet"
