from __future__ import annotations

import pandas as pd

from excrypto.baseline import momentum


def _ensure_utc(ts: pd.Series | pd.DatetimeIndex) -> pd.Series:
    return pd.to_datetime(ts, utc=True)


def momentum_signals(panel: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """
    panel: indexed by timestamp, columns include ['symbol','close'].
    returns: DataFrame columns ['timestamp','symbol','signal'] sorted.
    """
    sigs: list[pd.DataFrame] = []
    for sym, g in panel.groupby("symbol"):
        s = momentum._sma_sig(g["close"], fast=fast, slow=slow).rename("signal")
        sigs.append(
            pd.DataFrame(
                {
                    "timestamp": _ensure_utc(s.index),
                    "symbol": sym,
                    "signal": s.astype(float).values,
                }
            )
        )

    return (
        pd.concat(sigs, ignore_index=True)
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )


def hodl_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """
    panel: indexed by timestamp, columns include ['symbol'].
    returns: DataFrame columns ['timestamp','symbol','signal'] sorted.
    """
    return (
        pd.DataFrame(
            {
                "timestamp": _ensure_utc(panel.index),
                "symbol": panel["symbol"].values,
                "signal": 1.0,
            }
        )
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )
