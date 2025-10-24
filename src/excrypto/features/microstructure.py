from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from .base import StatelessFeature
from .registry import register_feature
from .utils import _as_series

@register_feature("roll_measure")
@dataclass
class RollMeasure(StatelessFeature):
    """
    Roll (1984) effective spread estimator using midprice changes.
    Input: midprice column (e.g., (bid+ask)/2 or close for proxy)
    """
    window: int = 50
    min_periods: int = 20

    def transform(self, df: pd.DataFrame) -> pd.Series:
        m = _as_series(df, list(self.input_cols)[0]).astype(float)
        dm = m.diff()
        cov = (dm * dm.shift(1)).rolling(self.window, min_periods=self.min_periods).mean()
        # Negative serial covariance expected; spread ≈ 2 * sqrt(-cov)
        spread = (2 * (-cov).clip(lower=0.0) ** 0.5).rename(self.output_col)
        return spread

@register_feature("vpin_approx")
@dataclass
class VPINApprox(StatelessFeature):
    """
    Simple VPIN-like imbalance using returns sign as buy/sell proxy.
    Input: returns column (signed), and volume column.
    """
    bucket: int = 50  # rolling window size

    def transform(self, df: pd.DataFrame) -> pd.Series:
        cols = list(self.input_cols)
        r = _as_series(df, cols[0]).astype(float)
        vol = _as_series(df, cols[1]).astype(float)
        buy_vol = (r >= 0).astype(float) * vol
        sell_vol = (r < 0).astype(float) * vol
        imbalance = (buy_vol - sell_vol).abs().rolling(self.bucket, min_periods=max(5, self.bucket // 5)).sum()
        total = vol.rolling(self.bucket, min_periods=max(5, self.bucket // 5)).sum()
        vpin = (imbalance / total).rename(self.output_col)
        return vpin
