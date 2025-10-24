from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from .base import StatelessFeature
from .registry import register_feature
from .utils import _as_series

@register_feature("rsi")
@dataclass
class RSI(StatelessFeature):
    window: int = 14
    min_periods: int = 5

    def transform(self, df: pd.DataFrame) -> pd.Series:
        price = _as_series(df, list(self.input_cols)[0]).astype(float)
        delta = price.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.rolling(self.window, min_periods=self.min_periods).mean()
        avg_loss = loss.rolling(self.window, min_periods=self.min_periods).mean()
        rs = avg_gain / (avg_loss.replace(0, pd.NA))
        rsi = 100 - (100 / (1 + rs))
        return rsi.rename(self.output_col)

@register_feature("macd")
@dataclass
class MACD(StatelessFeature):
    fast: int = 12
    slow: int = 26
    signal: int = 9

    def transform(self, df: pd.DataFrame) -> pd.Series:
        price = _as_series(df, list(self.input_cols)[0]).astype(float)
        ema_fast = price.ewm(span=self.fast, adjust=False).mean()
        ema_slow = price.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.signal, adjust=False).mean()
        hist = macd - signal
        # Convention: output the histogram; users can recompute lines if needed.
        return hist.rename(self.output_col)
