# tests/test_backtest_engine.py
import pandas as pd, numpy as np
from excrypto.backtest.engine import BacktestConfig, backtest_single

def test_backtest_basic_latency_and_shapes():
    idx = pd.date_range("2025-01-01", periods=120, freq="T", tz="UTC")
    px = 100 * (1 + 0.0005*np.arange(120)).astype(float)
    sig = pd.Series(1.0, index=idx)  # constant long
    df = pd.DataFrame({"close": px}, index=idx); df["signal"] = sig
    bt = backtest_single(df, BacktestConfig(latency_bars=1, vol_lookback=10))
    assert set(["ret","weight","pnl_gross","pnl_net","equity"]).issubset(bt.columns)
    # latency: first tradable pnl should be 0 (no position yet)
    assert bt["pnl_gross"].iloc[0] == 0.0
    # equity monotonic when fees=0 and positive drift
    bt2 = backtest_single(df, BacktestConfig(fee_bps=0, slippage_bps=0, vol_lookback=10))
    assert bt2["equity"].iloc[-1] >= 1.0
