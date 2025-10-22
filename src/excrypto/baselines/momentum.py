from __future__ import annotations
import pandas as pd
from excrypto.backtest.engine import BacktestConfig, backtest_single, backtest_multi

def _sma_sig(px: pd.Series, fast:int=20, slow:int=60) -> pd.Series:
    # leak-safe: use t-1 close
    c1 = px.shift(1)
    f = c1.rolling(fast).mean()
    s = c1.rolling(slow).mean()
    sig = (f > s).astype(float) - (f < s).astype(float)  # +1 / -1
    return sig.fillna(0.0)

def run_single(df: pd.DataFrame, cfg: BacktestConfig, fast:int=20, slow:int=60) -> pd.DataFrame:
    x = df.sort_index().copy()
    x["signal"] = _sma_sig(x["close"], fast, slow)
    return backtest_single(x[["close","signal"]], cfg)

def run_multi(panel: pd.DataFrame, cfg: BacktestConfig, fast:int=20, slow:int=60) -> pd.DataFrame:
    xs=[]
    for sym, g in panel.sort_index().groupby("symbol"):
        z = g.copy()
        z["signal"] = _sma_sig(z["close"], fast, slow)
        z["symbol"] = sym
        xs.append(z[["close","signal","symbol"]])
    return backtest_multi(pd.concat(xs), cfg)
