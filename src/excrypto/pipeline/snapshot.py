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
from datetime import datetime, timezone, timedelta
from typing import Iterable, List, Optional
import time
from pathlib import Path

import ccxt
import pandas as pd

from excrypto.utils import registry

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
    return os.path.join(cfg.root, cfg.snapshot, cfg.exchange, cfg.timeframe, sym.replace("/", "_"))



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


def run_day(
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

        # right after writing ohlcv.parquet (and optional funding)
        path = os.path.join(pair_dir, "ohlcv.parquet")
        rows = int(ohlcv.shape[0])
        first_ts = ohlcv["timestamp"].min().isoformat() if rows else None
        last_ts  = ohlcv["timestamp"].max().isoformat() if rows else None

        rec = {
            "snapshot_id": cfg.snapshot,
            "exchange": cfg.exchange,
            "symbol": sym,
            "timeframe": cfg.timeframe,
            "path": path.replace("\\","/"),
            "rows": rows,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        registry.upsert_record(rec)

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


@dataclass
class SnapArgs:
    exchange: str = "binance"
    symbols: list[str] = None
    timeframe: str = "1m"
    limit: int = 1000  # per-API call
    data_root: str = "data/raw"

def _ts_utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)

def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def _fetch_range(ex, symbol: str, timeframe: str, since_ms: int, 
                 until_ms: int, limit: int) -> pd.DataFrame:
    rows = []
    t = since_ms
    while t < until_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=t, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        # ccxt rows are [ts, open, high, low, close, volume]
        last_ts = batch[-1][0]
        # advance; +1 ms to avoid duplicates
        t = last_ts + 1
        # be gentle on rate limits
        time.sleep(getattr(ex, "rateLimit", 0) / 1000.0)
        # stop if we somehow didn't move forward
        if len(batch) < limit and last_ts >= until_ms:
            break
        if last_ts == batch[0][0] and len(batch) == 1:
            break
    if not rows:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[(df["timestamp"] >= pd.to_datetime(since_ms, unit="ms", utc=True)) &
            (df["timestamp"] <  pd.to_datetime(until_ms, unit="ms", utc=True))]
    return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

def run_range(start: str, end: str, *, args: SnapArgs) -> str:
    syms = args.symbols or ["BTC/USDT"]
    start_dt, end_dt = _ts_utc(start), _ts_utc(end) + timedelta(days=1)  # end is inclusive; add 1 day
    since_ms, until_ms = _ms(start_dt), _ms(end_dt)

    snap_id = f"{start}_to_{end}"
    out_root = Path(args.data_root) / snap_id / args.exchange / args.timeframe
    out_root.mkdir(parents=True, exist_ok=True)

    ex = getattr(ccxt, args.exchange)()
    ex.enableRateLimit = True
    ex.load_markets()

    for sym in syms:
        df = _fetch_range(ex, sym, args.timeframe, since_ms, until_ms, args.limit)
        out_dir = out_root / sym.replace("/","_")
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / "ohlcv.parquet", index=False)

        path = str(out_dir / "ohlcv.parquet")
        rows = int(df.shape[0])
        first_ts = df["timestamp"].min().isoformat() if rows else None
        last_ts  = df["timestamp"].max().isoformat() if rows else None

        rec = {
            "snapshot_id": snap_id,
            "exchange": args.exchange,
            "symbol": sym,
            "timeframe": args.timeframe,
            "path": path.replace("\\","/"),
            "rows": rows,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        registry.upsert_record(rec)


    meta = {
        "type": "combined_range",
        "exchange": args.exchange,
        "symbols": syms,
        "timeframe": args.timeframe,
        "start": start,
        "end": end,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_root / "_snapshot_meta.json").write_text(json.dumps(meta, indent=2))


    return str(out_root)