# src/excrypto/data/snapshot.py
from __future__ import annotations

"""
Fetch immutable raw market data (OHLCV + optional funding if available) and write to:

  data/raw/<snapshot_id>/<exchange>/<timeframe>/
      _snapshot_meta.json
      _universe.json
      <SYMBOL_>/ohlcv.parquet
      <SYMBOL_>/funding.parquet   (optional)

Also upserts a record into the registry for each symbol/timeframe dataset.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable

import ccxt
import pandas as pd

from excrypto.data import registry


@dataclass(frozen=True)
class SnapshotConfig:
    exchange: str = "binance"
    symbols: tuple[str, ...] = ("BTC/USDT",)
    timeframe: str = "1m"
    ohlcv_limit: int = 1000          # per API call (paging)
    funding_limit: int = 1000
    root: Path = Path("data/raw")


@dataclass(frozen=True)
class SnapshotResult:
    snapshot_id: str
    root: Path                    # dataset root: data/raw/<snapshot_id>/<exchange>/<timeframe>
    symbols_written: list[str]


def _ex(exchange: str) -> ccxt.Exchange:
    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    ex.load_markets()
    return ex


def _sym_dir(sym: str) -> str:
    return sym.replace("/", "_")


def _ts_utc_day(s: str) -> datetime:
    # s: YYYY-MM-DD
    # keep strict: treat as UTC date boundary
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _fetch_ohlcv_range(
    ex: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    limit: int,
    max_batches: int = 50_000,
) -> pd.DataFrame:
    """
    Paged OHLCV fetch in [since_ms, until_ms), with basic guards.
    """
    rows: list[list[float]] = []
    t = since_ms

    for _ in range(max_batches):
        if t >= until_ms:
            break

        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=t, limit=limit)
        if not batch:
            break

        rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts < t:
            break

        # advance; +1ms avoids duplicates
        t = last_ts + 1

        # rate limit friendly
        time.sleep(getattr(ex, "rateLimit", 0) / 1000.0)

        # if we got very little, likely done
        if len(batch) < limit and last_ts >= until_ms - 1:
            break

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # filter to requested half-open interval [since, until)
    since_dt = pd.to_datetime(since_ms, unit="ms", utc=True)
    until_dt = pd.to_datetime(until_ms, unit="ms", utc=True)
    df = df[(df["timestamp"] >= since_dt) & (df["timestamp"] < until_dt)]

    return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _fetch_funding(ex: ccxt.Exchange, sym: str, limit: int) -> pd.DataFrame:
    try:
        raw = ex.fetch_funding_rate_history(sym, limit=limit)
    except Exception:
        return pd.DataFrame()
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    if "timestamp" not in df.columns:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    if "fundingRate" in df.columns:
        df["funding"] = df["fundingRate"]
    elif "funding" not in df.columns:
        return pd.DataFrame()

    df = df[["timestamp", "funding"]].copy()
    df.insert(0, "symbol", sym)
    return df


def build_snapshot(cfg: SnapshotConfig, *, start: str, end: str) -> SnapshotResult:
    """
    Fetch OHLCV (and optional funding) for [start, end] inclusive as UTC dates (YYYY-MM-DD).

    Writes a single “snapshot dataset” rooted at:
      <cfg.root>/<start>_to_<end>/<exchange>/<timeframe>/

    For a single day, pass start=end (still produces ..._to_... id for consistency).
    """
    start_dt = _ts_utc_day(start)
    end_dt_inclusive = _ts_utc_day(end)
    until_dt = end_dt_inclusive + timedelta(days=1)  # half-open interval
    since_ms, until_ms = _ms(start_dt), _ms(until_dt)

    snap_id = f"{start}_to_{end}"
    out_root = cfg.root / snap_id / cfg.exchange / cfg.timeframe
    out_root.mkdir(parents=True, exist_ok=True)

    ex = _ex(cfg.exchange)

    written: list[str] = []
    for sym in cfg.symbols:
        if sym not in ex.markets:
            continue

        df = _fetch_ohlcv_range(ex, sym, cfg.timeframe, since_ms, until_ms, cfg.ohlcv_limit)
        if df.empty:
            continue

        df.insert(0, "symbol", sym)
        sym_out = out_root / _sym_dir(sym)
        _write_parquet(df, sym_out / "ohlcv.parquet")

        # optional funding (skip silently if unsupported)
        fund = _fetch_funding(ex, sym, cfg.funding_limit)
        if not fund.empty:
            _write_parquet(fund, sym_out / "funding.parquet")

        rows = int(df.shape[0])
        rec = {
            "kind": "ohlcv",
            "snapshot_id": snap_id,
            "exchange": cfg.exchange,
            "symbol": sym,
            "timeframe": cfg.timeframe,
            "rows": rows,
            "first_ts": df["timestamp"].min().isoformat() if rows else None,
            "last_ts": df["timestamp"].max().isoformat() if rows else None,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        registry.upsert_record(rec)
        written.append(sym)


    meta = {
        "snapshot_id": snap_id,
        "exchange": cfg.exchange,
        "timeframe": cfg.timeframe,
        "symbols": written,
        "start": start,
        "end": end,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ccxt_version": ccxt.__version__,
    }
    (out_root / "_snapshot_meta.json").write_text(json.dumps(meta, indent=2))
    (out_root / "_universe.json").write_text(json.dumps(written, indent=2))

    return SnapshotResult(snapshot_id=snap_id, root=out_root, symbols_written=written)
