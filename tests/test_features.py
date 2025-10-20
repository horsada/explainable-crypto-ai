# tests/test_preprocessing_pipeline.py
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

# If you named it feature_pipeline.py, use:
# from excrypto.feature_pipeline import FeaturePipeline, FeatureConfig
from excrypto.preprocessing import FeaturePipeline, FeatureConfig


def _make_df(n=12, start="2024-01-01 00:00:00"):
    t0 = datetime.fromisoformat(start)
    times = [t0 + timedelta(minutes=i) for i in range(n)]
    # simple price series with a trend + noise
    rng = np.random.default_rng(0)
    close = 100 + np.arange(n) * 0.5 + rng.normal(0, 0.1, n)
    return pd.DataFrame({"open_time": times, "close": close})


def test_transform_generates_features_and_labels_alignment():
    df = _make_df(12)

    cfg = FeatureConfig(
        features=[
            # must match what the pipeline generates below
            "ret_1",
            "lag_close_1", "lag_close_2",
            "roll_mean_close_2", "roll_std_close_2",
        ],
        returns_windows=[1],
        lag_features={"close": [1, 2]},
        rolling_mean={"close": [2]},
        rolling_std={"close": [2]},
        fillna=None,     # drop NaNs created by lags/rolls
        dropna=True,
    )
    pipe = FeaturePipeline(cfg)
    X, y = pipe.transform(df, compute_label=True)

    # Columns as configured
    assert list(X.columns) == cfg.features

    # Due to lag 2 / rolling 2, at least first 2 rows dropped
    # (and possibly more from returns), so length must be n-2
    assert len(X) == len(df) - 2

    # No NaNs after cleaning
    assert not X.isna().any().any()

    # y exists, aligned to X index, and is binary {0,1}
    assert y is not None
    assert list(y.index) == list(X.index)
    assert set(map(int, y.dropna().unique())) <= {0, 1}

    # y equals (close.shift(-1) > close), restricted to X index
    df_sorted = df.sort_values("open_time").set_index("open_time", drop=False)
    expected_y = (df_sorted["close"].shift(-1) > df_sorted["close"]).astype(int)
    expected_y = expected_y.reindex(X.index)  # align to X
    pd.testing.assert_series_equal(y.astype(int), expected_y.astype(int), check_names=False)


def test_missing_required_columns_raises():
    df = _make_df(10).drop(columns=["open_time"])  # remove a required column
    cfg = FeatureConfig(features=["ret_1"], returns_windows=[1])
    pipe = FeaturePipeline(cfg)
    with pytest.raises(ValueError):
        pipe.transform(df, compute_label=False)


def test_no_dropna_with_fillna_zero():
    df = _make_df(8)
    cfg = FeatureConfig(
        features=["ret_1", "lag_close_1", "roll_mean_close_2"],
        returns_windows=[1],
        lag_features={"close": [1]},
        rolling_mean={"close": [2]},
        rolling_std={},           # none
        dropna=False,
        fillna=0.0,
    )
    pipe = FeaturePipeline(cfg)
    X, y = pipe.transform(df, compute_label=True)

    # Keep length of original (since we filled NaNs instead of dropping)
    assert len(X) == len(df)
    assert not X.isna().any().any()
    assert y is not None and len(y) == len(X)


def test_time_index_sorted_and_past_only():
    # Shuffle rows to ensure pipeline sorts by time
    df = _make_df(10).sample(frac=1.0, random_state=42).reset_index(drop=True)

    cfg = FeatureConfig(
        features=["lag_close_2", "roll_mean_close_2"],
        returns_windows=[],                 # avoid extra drops
        lag_features={"close": [2]},
        rolling_mean={"close": [2]},
        rolling_std={},
        dropna=True,
    )
    pipe = FeaturePipeline(cfg)
    X, _ = pipe.transform(df, compute_label=False)

    # After alignment/sort, earliest usable row should be index position 2 of the sorted series
    df_sorted = df.sort_values("open_time").set_index("open_time", drop=False)
    expected_first_idx = df_sorted.index[2]
    assert X.index[0] == expected_first_idx
