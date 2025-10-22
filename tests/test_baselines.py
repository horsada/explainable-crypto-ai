import pandas as pd, numpy as np
from excrypto.backtest.engine import BacktestConfig
from excrypto.baselines import hodl, vt_hodl, momentum

def _df(n=300):
    idx = pd.date_range("2025-01-01", periods=n, freq="T", tz="UTC")
    px = pd.Series(100 + np.cumsum(np.random.default_rng(0).normal(0,0.05,n)), index=idx)
    return pd.DataFrame({"close": px}, index=idx)

def test_hodl_runs():
    df = _df()
    out = hodl.run_single(df, BacktestConfig(vol_lookback=30))
    assert {"pnl_net","equity"}.issubset(out.columns)

def test_vt_hodl_runs():
    df = _df()
    out = vt_hodl.run_single(df, BacktestConfig(vol_lookback=30))
    assert out["equity"].iloc[-1] > 0

def test_momentum_leak_safe():
    df = _df()
    out = momentum.run_single(df, BacktestConfig(vol_lookback=30), fast=5, slow=20)
    # first few bars should be zero signals due to shift/rolls
    assert out["weight"].iloc[:20].abs().sum() == 0 or out["weight"].iloc[:5].abs().sum() == 0
