import pandas as pd
from excrypto.data.pit import safe_final_bar, asof_join, assert_monotonic

def _df(ts, vals, symbol="BTC/USDT"):
    return pd.DataFrame({"symbol":symbol,"timestamp":pd.to_datetime(ts, utc=True),"x":vals})

def test_safe_final_bar_shift():
    prices = pd.DataFrame({
        "symbol":["s"]*3,
        "timestamp":pd.to_datetime(["2025-01-01T00:00Z","2025-01-01T00:01Z","2025-01-01T00:02Z"]),
        "close":[100,110,120],
    })
    out = safe_final_bar(prices)
    assert out.loc[1,"close_t1"] == 100
    assert pd.isna(out.loc[0,"close_t1"])

def test_asof_join_with_lag():
    left = _df(["2025-01-01T00:01Z","2025-01-01T00:02Z"], [0,0])
    right = _df(["2025-01-01T00:00Z","2025-01-01T00:02Z"], [1,2]).rename(columns={"x":"r"})
    # with 1m lag, at 00:02 left can only see right up to 00:01
    m = asof_join(left, right, pub_lag="1min")
    assert m.loc[m["timestamp"]=="2025-01-01 00:01:00+00:00","r"].iloc[0] == 1
    assert m.loc[m["timestamp"]=="2025-01-01 00:02:00+00:00","r"].iloc[0] == 1  # not 2

def test_assert_monotonic_raises():
    bad = pd.DataFrame({
        "symbol":["s","s"],
        "timestamp":pd.to_datetime(["2025-01-01T00:01Z","2025-01-01T00:00Z"]),
    })
    try:
        assert_monotonic(bad)
        assert False, "should raise"
    except ValueError:
        pass
