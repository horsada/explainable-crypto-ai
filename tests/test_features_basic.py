import pandas as pd
import numpy as np
from src.excrypto.features import (
    SimpleReturns, LogReturns,
    RollingVolatility, RSI, FeaturePipeline
)

def _mk_df(n=200, seed=7):
    rng = np.random.default_rng(seed)
    price = 100 + np.cumsum(rng.normal(0, 1, n))
    vol = rng.uniform(10, 100, n)
    return pd.DataFrame({"close": price, "volume": vol})

def test_basic_returns_and_vol():
    df = _mk_df()
    r = LogReturns(["close"], "ret_log").transform(df)
    assert r.isna().sum() >= 1
    vol = RollingVolatility(["ret_log"], "vol_ann", window=30).transform(pd.concat([df, r], axis=1))
    assert vol.shape[0] == df.shape[0]

def test_rsi_pipeline():
    df = _mk_df()
    specs = [
        {"name": "log_returns", "input_cols": ["close"], "output_col": "ret_log"},
        {"name": "rolling_volatility", "input_cols": ["ret_log"], "output_col": "vol_30", "params": {"window": 30}},
        {"name": "rsi", "input_cols": ["close"], "output_col": "rsi_14", "params": {"window": 14}},
    ]
    pipe = FeaturePipeline(specs).fit(df)
    feats = pipe.transform(df)
    assert {"ret_log", "vol_30", "rsi_14"} <= set(feats.columns)
