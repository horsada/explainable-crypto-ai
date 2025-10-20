from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

REQUIRED_COLS = ["open_time", "close"]

@dataclass
class FeatureConfig:
    features: List[str]
    returns_windows: List[int] = field(default_factory=lambda: [1, 5, 15])
    lag_features: Dict[str, List[int]] = field(default_factory=lambda: {"close": [1, 2, 3]})
    rolling_mean: Dict[str, List[int]] = field(default_factory=lambda: {"close": [5, 20]})
    rolling_std: Dict[str, List[int]]  = field(default_factory=lambda: {"close": [5, 20]})
    fillna: Optional[float] = None
    dropna: bool = True
    align_on_open_time: bool = True
    target_horizon: int = 1  # NEW

class FeaturePipeline:
    def __init__(self, config: FeatureConfig):
        self.cfg = config

    def transform(self, df: pd.DataFrame, *, compute_label: bool = False
                  ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        df = df.copy()
        self._validate_base(df)
        if self.cfg.align_on_open_time:
            df = self._align_time(df)

        self._gen_returns(df)
        self._gen_lags(df)
        self._gen_rolls(df)

        # Ensure requested features exist
        missing_feats = [c for c in self.cfg.features if c not in df.columns]
        if missing_feats:
            raise ValueError(f"Requested features not generated: {missing_feats}")

        X = df[self.cfg.features].copy()

        # Clean
        if self.cfg.dropna:
            X = X.dropna()
        elif self.cfg.fillna is not None:
            X = X.fillna(self.cfg.fillna)

        y = None
        if compute_label:
            h = self.cfg.target_horizon
            y = (df["close"].shift(-h) > df["close"]).astype(int)
            y = y.reindex(X.index)  # safer than .loc; aligns with X

        return X, y

    def _validate_base(self, df: pd.DataFrame):
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _align_time(self, df: pd.DataFrame) -> pd.DataFrame:
        if "open_time" in df.columns:
            df = df.set_index("open_time", drop=False)
        return df.sort_index()

    def _gen_returns(self, df: pd.DataFrame):
        for w in self.cfg.returns_windows:
            df[f"ret_{w}"] = df["close"].pct_change(w)

    def _gen_lags(self, df: pd.DataFrame):
        for col, lags in self.cfg.lag_features.items():
            for k in lags:
                df[f"lag_{col}_{k}"] = df[col].shift(k)

    def _gen_rolls(self, df: pd.DataFrame):
        for col, wins in self.cfg.rolling_mean.items():
            for w in wins:
                df[f"roll_mean_{col}_{w}"] = df[col].rolling(w).mean()
        for col, wins in self.cfg.rolling_std.items():
            for w in wins:
                df[f"roll_std_{col}_{w}"] = df[col].rolling(w).std()
