from __future__ import annotations
import pandas as pd
import numpy as np
from .base import StatelessFeature
from .registry import register_feature
from .utils import _as_series, _safe_div

@register_feature("simple_returns")
class SimpleReturns(StatelessFeature):
    """
    Simple returns: r_t = (P_t / P_{t-1}) - 1
    """
    def transform(self, df: pd.DataFrame) -> pd.Series:
        price = _as_series(df, list(self.input_cols)[0]).astype(float)
        ret = price.pct_change()
        return ret.rename(self.output_col)

@register_feature("log_returns")
class LogReturns(StatelessFeature):
    """
    Log returns: ln(P_t) - ln(P_{t-1})
    """
    def transform(self, df: pd.DataFrame) -> pd.Series:
        price = _as_series(df, list(self.input_cols)[0]).astype(float)
        logp = np.log(price.replace(0, pd.NA))
        out = logp.diff()
        return out.rename(self.output_col)
