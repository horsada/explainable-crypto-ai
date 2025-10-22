from __future__ import annotations

"""
Pull immutable crypto snapshots (OHLCV + funding if available) and write to:
  data/raw/<snapshot>/<exchange>/<SYMBOL_>/[ohlcv|funding].parquet
Also writes:
  _snapshot_meta.json and _universe.json
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import ccxt
import pandas as pd


@dataclass
class SnapshotConfig:
    snapshot: str                    # e.g. "2025-10-21" (UTC date label)
    exchange: str = "binance"
    symbols: Iterable[str] = ("BTC/USDT",)
    timeframe: str = "1m"
    ohlcv_limit: int = 10_000        # per symbol
    funding_limit: int = 1_000       # per symbol
    root: str = "data/raw"           # base data directory


def _ex(exchange: str) -> ccxt.Exchange:
    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    ex.load_markets()
    return ex


def _path(cfg: SnapshotConfig, sym: str) -> str:
    return os.path.join(cfg.root, cfg.snapshot, cfg.exchange, sym.replace("/", "_"))


def _write_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def _fetch_ohlcv(ex: ccxt.Exchange, sym: str, timeframe: str, limit: int) -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    data = ex.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)  # ms timestamps
    df = pd.DataFrame(data, columns=cols)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.insert(0, "symbol", sym)
    return df


def _fetch_funding(ex: ccxt.Exchange, sym: str, limit: int) -> pd.DataFrame:
    # Works for perp symbols; skip gracefully otherwise.
    try:
        raw = ex.fetch_funding_rate_history(sym, limit=limit)
    except Exception:
        return pd.DataFrame()
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    # Standardize columns across exchanges
    # Expected: timestamp (ms), fundingRate or funding
    if "timestamp" not in df:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    # Try common field names
    if "fundingRate" in df:
        df["funding"] = df["fundingRate"]
    elif "funding" not in df:
        return pd.DataFrame()
    df = df[["timestamp", "funding"]].copy()
    df.insert(0, "symbol", sym)
    return df


def run(
    snapshot: str,
    exchange: str = "binance",
    symbols: Optional[str] = None,          # CSV string OR None to infer from exchange (top pairs)
    timeframe: str = "1m",
    ohlcv_limit: int = 10_000,
    funding_limit: int = 1_000,
    root: str = "data/raw",
) -> List[str]:
    """
    Execute a snapshot. Return the universe list actually written.
    """
    sym_list: List[str]
    ex = _ex(exchange)

    if symbols:
        sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    else:
        # Fallback: pick a small stable universe (can customize)
        candidates = ["BTC/USDT", "ETH/USDT"]
        sym_list = [s for s in candidates if s in ex.markets]

    cfg = SnapshotConfig(
        snapshot=snapshot,
        exchange=exchange,
        symbols=sym_list,
        timeframe=timeframe,
        ohlcv_limit=ohlcv_limit,
        funding_limit=funding_limit,
        root=root,
    )

    base = os.path.join(cfg.root, cfg.snapshot, cfg.exchange)
    os.makedirs(base, exist_ok=True)

    written: List[str] = []
    for sym in cfg.symbols:
        if sym not in ex.markets:
            continue
        pair_dir = _path(cfg, sym)
        os.makedirs(pair_dir, exist_ok=True)

        ohlcv = _fetch_ohlcv(ex, sym, cfg.timeframe, cfg.ohlcv_limit)
        if not ohlcv.empty:
            _write_parquet(ohlcv, os.path.join(pair_dir, "ohlcv.parquet"))
        else:
            # If no price data, skip funding too
            continue

        funding = _fetch_funding(ex, sym, cfg.funding_limit)
        if not funding.empty:
            _write_parquet(funding, os.path.join(pair_dir, "funding.parquet"))

        written.append(sym)

    meta = {
        "snapshot_utc": cfg.snapshot,
        "time_captured_utc": datetime.now(timezone.utc).isoformat(),
        "exchange": cfg.exchange,
        "symbols": written,
        "timeframe": cfg.timeframe,
        "ohlcv_limit": cfg.ohlcv_limit,
        "funding_limit": cfg.funding_limit,
        "ccxt_version": ccxt.__version__,
    }
    with open(os.path.join(base, "_snapshot_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(base, "_universe.json"), "w") as f:
        json.dump(written, f, indent=2)

    return written
