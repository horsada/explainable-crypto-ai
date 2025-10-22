# tests/test_splits.py
import pandas as pd
import numpy as np
import pytest
from excrypto.pipeline.splits import (
    build_rolling_splits, make_purged_kfold_indices,
    assert_no_overlap, assert_increasing_windows
)

def _df(n=300, start="2025-01-01", freq="T"):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(index=idx, data={"x": np.arange(n)})

def test_build_rolling_basic():
    df = _df(180)  # 3 hours of 1-min data
    folds = build_rolling_splits(df, train="90min", valid="30min", step="30min", embargo="5min")
    assert len(folds) > 0
    assert_no_overlap(folds)
    assert_increasing_windows(folds)
    # spot check: each fold has data
    for f in folds:
        assert f.train_idx.size > 0 and f.valid_idx.size > 0
        # embargo: last train ts < first valid ts
        assert df.index[f.train_idx].max() < df.index[f.valid_idx].min()

def test_build_rolling_strict_order_and_sizes():
    df = _df(200)
    folds = build_rolling_splits(df, train="60min", valid="20min", step="20min", embargo="10min")
    for f in folds:
        # windows are increasing
        assert f.train.start < f.train.end < f.valid.start < f.valid.end

def test_purged_kfold_no_overlap_with_embargo():
    df = _df(500)
    folds = make_purged_kfold_indices(df, n_splits=5, embargo="10min")
    assert len(folds) == 5
    assert_no_overlap(folds)

    emb = pd.Timedelta("10min")
    for f in folds:
        v_start = df.index[f.valid_idx].min()
        v_end   = df.index[f.valid_idx].max()
        train_times = df.index[f.train_idx]
        # No training times in (v_start - emb, v_end + emb)
        inside = (train_times > (v_start - emb)) & (train_times < (v_end + emb))
        assert not inside.any()


def test_requires_tzaware_index():
    # naive index should raise
    idx = pd.date_range("2025-01-01", periods=10, freq="H")  # no tz
    df = pd.DataFrame(index=idx, data={"x": range(10)})
    with pytest.raises(ValueError):
        build_rolling_splits(df, train="3H", valid="1H")
