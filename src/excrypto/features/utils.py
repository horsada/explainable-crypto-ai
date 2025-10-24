from __future__ import annotations
import pandas as pd
import numpy as np

def _as_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")
    s = df[col]
    if not isinstance(s, pd.Series):
        s = pd.Series(s, index=df.index)
    return s

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
        out = out.replace([np.inf, -np.inf], np.nan)
    return out
