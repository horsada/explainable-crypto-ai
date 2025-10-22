from __future__ import annotations
import pandas as pd
from excrypto.backtest.engine import BacktestConfig, backtest_single, backtest_multi

def run_single(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    x = df.sort_index().copy()
    x["signal"] = 1.0   # engine will vol-target it
    return backtest_single(x[["close","signal"]], cfg)

def run_multi(panel: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    x = panel.sort_index().copy()
    x["signal"] = 1.0
    return backtest_multi(x[["close","signal","symbol"]], cfg)
