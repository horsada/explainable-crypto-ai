# src/excrypto/agents/catalog.py
from excrypto.utils.registry import find
import pandas as pd, math, os
from datetime import datetime, timezone

def _expected_per_year(freq_s: float): return 365.25*24*3600/freq_s

def summarize(snapshot_id: str | None = None) -> pd.DataFrame:
    recs = find(snapshot_id=snapshot_id)
    if recs.empty: return recs
    rows=[]
    for _, r in recs.iterrows():
        rows.append({
          "snapshot_id": r.snapshot_id,
          "symbol": r.symbol,
          "timeframe": r.timeframe,
          "rows": int(r.rows),
          "first_ts": r.first_ts, "last_ts": r.last_ts,
        })
    df = pd.DataFrame(rows)
    # crude completeness estimate (assumes fixed timeframe in seconds)
    tf_sec = {"1m":60,"5m":300,"1h":3600}.get(df["timeframe"].iloc[0], None)
    if tf_sec and df.first_ts.notna().all() and df.last_ts.notna().all():
        t0 = pd.to_datetime(df["first_ts"].min(), utc=True)
        t1 = pd.to_datetime(df["last_ts"].max(), utc=True)
        expected = math.floor((t1 - t0).total_seconds()/tf_sec) + 1
        df["pct_coverage_vs_union"] = (df["rows"]/expected).round(3)
    return df.sort_values(["snapshot_id","symbol"])
