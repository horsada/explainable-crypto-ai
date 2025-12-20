# src/excrypto/utils/loader.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from excrypto.data.registry import find as registry_find
from excrypto.data.paths import raw_market_path


def load_snapshot(
    snapshot_id: str,
    symbols: list[str] | None,
    *,
    exchange: str,                      # REQUIRED
    timeframe: str = "1m",
    raw_root: str | Path = "data/raw",  # MUST be the raw data root
    strict: bool = True,
) -> pd.DataFrame:
    """
    Load a panel with index=timestamp and columns: symbol, close.

    Registry-only, deterministic (exchange required).
    """
    raw_root = Path(raw_root)

    df_reg = registry_find(
        kind="ohlcv",
        snapshot_id=snapshot_id,
        exchange=exchange,
        timeframe=timeframe,
    )
    if df_reg.empty:
        raise FileNotFoundError(
            f"No raw OHLCV registered for snapshot={snapshot_id} exchange={exchange} timeframe={timeframe}"
        )

    if symbols:
        symbols_set = set(symbols)
        reg_syms = set(df_reg["symbol"].astype(str))
        missing = sorted(symbols_set - reg_syms)
        if missing and strict:
            raise FileNotFoundError(
                f"Missing symbols in registry for snapshot={snapshot_id} exchange={exchange} timeframe={timeframe}: {missing}"
            )
        df_reg = df_reg[df_reg["symbol"].isin(symbols)]

    rows: list[pd.DataFrame] = []
    for _, r in df_reg.iterrows():
        sym = str(r["symbol"])
        p = raw_market_path(raw_root, snapshot_id, exchange, timeframe, sym, kind="ohlcv")

        if not p.exists():
            if strict:
                raise FileNotFoundError(f"Expected raw file missing: {p}")
            continue

        df = pd.read_parquet(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["symbol"] = sym
        rows.append(df[["timestamp", "symbol", "close"]])

    if not rows:
        return pd.DataFrame(columns=["symbol", "close"]).set_index(pd.DatetimeIndex([], name="timestamp"))

    return pd.concat(rows).sort_values("timestamp").set_index("timestamp")