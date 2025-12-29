# src/excrypto/ml/datasets.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class XYData:
    X: pd.DataFrame
    y: pd.Series
    keys: pd.DataFrame
    label_col: str


def load_xy(
    features_path: str,
    labels_path: str,
    *,
    label_col: str | None = None,
    key_cols: Sequence[str] = ("timestamp", "symbol"),
    how: str = "inner",
    dropna: bool = True,
) -> XYData:
    Xdf = pd.read_parquet(features_path)
    ydf = pd.read_parquet(labels_path)

    for c in key_cols:
        if c not in Xdf.columns:
            raise ValueError(f"Features missing key col '{c}': {features_path}")
        if c not in ydf.columns:
            raise ValueError(f"Labels missing key col '{c}': {labels_path}")

    # choose label column
    if label_col is None:
        label_cols = [c for c in ydf.columns if c not in key_cols]
        if not label_cols:
            raise ValueError("No label columns found in labels parquet.")
        label_col = label_cols[-1]
    if label_col not in ydf.columns:
        raise ValueError(f"Label column '{label_col}' not found in labels parquet.")

    df = Xdf.merge(ydf[list(key_cols) + [label_col]], on=list(key_cols), how=how)

    # build outputs
    keys = df[list(key_cols)].copy()
    y = df[label_col]
    X = df.drop(columns=list(key_cols) + [label_col])

    if dropna:
        keep = ~(X.isna().any(axis=1) | y.isna())
        X = X.loc[keep].reset_index(drop=True)
        y = y.loc[keep].reset_index(drop=True)
        keys = keys.loc[keep].reset_index(drop=True)

    return XYData(X=X, y=y, keys=keys, label_col=label_col)
