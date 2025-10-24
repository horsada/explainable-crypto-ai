from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from .base import StatelessFeature
from .registry import register_feature
from .utils import _as_series

@dataclass
class _RollingBase(StatelessFeature):
    window: int = 20
    min_periods: int = 5

@register_feature("rolling_mean")
class RollingMean(_RollingBase):
    def transform(self, df: pd.DataFrame) -> pd.Series:
        s = _as_series(df, list(self.input_cols)[0]).astype(float)
        return s.rolling(self.window, min_periods=self.min_periods).mean().rename(self.output_col)

@register_feature("rolling_std")
class RollingStd(_RollingBase):
    def transform(self, df: pd.DataFrame) -> pd.Series:
        s = _as_series(df, list(self.input_cols)[0]).astype(float)
        return s.rolling(self.window, min_periods=self.min_periods).std(ddof=0).rename(self.output_col)

@register_feature("rolling_zscore")
class RollingZScore(_RollingBase):
    def transform(self, df: pd.DataFrame) -> pd.Series:
        s = _as_series(df, list(self.input_cols)[0]).astype(float)
        mean = s.rolling(self.window, min_periods=self.min_periods).mean()
        std = s.rolling(self.window, min_periods=self.min_periods).std(ddof=0)
        z = (s - mean) / std
        return z.rename(self.output_col)

@register_feature("rolling_volatility")
class RollingVolatility(_RollingBase):
    """
    Annualized volatility of (log) returns within the window.
    Assumes input is returns; set `trading_periods` to scale (e.g., 365 for daily crypto).
    """
    trading_periods: int = 365

    def transform(self, df: pd.DataFrame) -> pd.Series:
        r = _as_series(df, list(self.input_cols)[0]).astype(float)
        vol = r.rolling(self.window, min_periods=self.min_periods).std(ddof=0) * np.sqrt(self.trading_periods)
        return vol.rename(self.output_col)
