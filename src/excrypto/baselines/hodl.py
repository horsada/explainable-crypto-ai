from __future__ import annotations
import pandas as pd
from excrypto.backtest.engine import BacktestConfig, backtest_single, backtest_multi

def run_single(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Buy & hold on one asset."""
    x = df.sort_index().copy()
    x["signal"] = 1.0  # constant long
    # For pure HODL, disable vol-target/latency impact if desired:
    cfg2 = BacktestConfig(**{**cfg.__dict__, "target_vol_ann": cfg.target_vol_ann})  # keep as-is
    return backtest_single(x[["close","signal"]], cfg2)

def run_multi(panel: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Equal-weight buy & hold across symbols."""
    x = panel.sort_index().copy()
    x["signal"] = 1.0
    return backtest_multi(x[["close","signal","symbol"]], cfg)
