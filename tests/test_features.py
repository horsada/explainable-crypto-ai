import pandas as pd
from excrypto.data import FeaturePipeline, FeatureConfig

def test_features_pit_and_labels():
    ts = pd.date_range("2025-01-01", periods=10, freq="T", tz="UTC")
    df = pd.DataFrame({"open_time": ts, "close": range(10)})
    cfg = FeatureConfig(features=["ret_1","lag_close_1","roll_mean_close_5"], target_horizon=1)
    X, y = FeaturePipeline(cfg).transform(df, compute_label=True)

    assert not X.empty and y is not None
    assert isinstance(X.index, pd.DatetimeIndex)

    # leakage guard: features start after the first timestamp
    assert X.index.min() > df["open_time"].min()

    # label aligned & last horizon dropped
    assert (X.index == y.index).all()
    assert y.index.max() < df["open_time"].max()

