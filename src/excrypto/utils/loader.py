# src/excrypto/utils/loaders.py
from pathlib import Path
import pandas as pd
from excrypto.utils.registry import find  # the registry you built

def _discover_symbols(tf_dir: Path) -> list[str]:
    """
    Given data/raw/<snapshot>/<exchange>/<timeframe>/,
    return symbols by reversing the folder slug (e.g., BTC_USDT -> BTC/USDT).
    """
    syms = []
    for p in tf_dir.iterdir():
        if p.is_dir():
            slug = p.name               # e.g. "BTC_USDT"
            syms.append(slug.replace("_", "/"))
    return sorted(syms)


# utils/loaders.py
# excrypto/utils/loader.py
def load_snapshot(snapshot_id: str, symbols: list[str] | None, timeframe: str = "1m", root="data/raw"):
    rows = []
    base = Path(root) / snapshot_id
    # allow multiple exchanges or pick one
    for ex in (p.name for p in (base).iterdir() if p.is_dir()):
        tf_dir = base / ex / timeframe
        if not tf_dir.exists():
            continue
        for sym in symbols or _discover_symbols(tf_dir):
            p = tf_dir / sym.replace("/","_") / "ohlcv.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["symbol"] = sym
            rows.append(df[["timestamp","symbol","close"]])
    panel = pd.concat(rows).sort_values("timestamp").set_index("timestamp")
    return panel


