from __future__ import annotations
import pandas as pd

def load_xy(features_path: str, labels_path: str, label_col: str | None = None):
    X = pd.read_parquet(features_path)
    y = pd.read_parquet(labels_path)
    df = X.merge(y, on=["timestamp","symbol"], how="inner").dropna()
    if label_col is None:
        # take the last non-key column from labels file
        label_cols = [c for c in y.columns if c not in ("timestamp","symbol")]
        if not label_cols:
            raise ValueError("No label columns found in labels parquet.")
        label_col = label_cols[-1]
    y = df[label_col]
    X = df.drop(columns=["timestamp","symbol", label_col])
    keys = df[["timestamp","symbol"]]  # for later join to signals
    return X, y, keys, label_col
