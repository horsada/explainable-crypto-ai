from __future__ import annotations
from typing import Iterable, List, Dict, Any
import pandas as pd
from .base import Feature
from .registry import get_feature_cls

class FeaturePipeline:
    """
    Lightweight pipeline to generate multiple features and concat to a DataFrame.
    specs example:
    [
      {"name": "log_returns", "input_cols": ["close"], "output_col": "ret_log"},
      {"name": "rolling_volatility", "input_cols": ["ret_log"], "output_col": "vol_30", "params": {"window": 30}},
      {"name": "rsi", "input_cols": ["close"], "output_col": "rsi_14", "params": {"window": 14}},
    ]
    """
    def __init__(self, specs: Iterable[Dict[str, Any]]) -> None:
        self.specs = list(specs)
        self.features: List[Feature] = []

    def build(self) -> "FeaturePipeline":
        self.features = []
        for spec in self.specs:
            cls = get_feature_cls(spec["name"])
            params = spec.get("params", {})
            feat = cls(input_cols=spec["input_cols"], output_col=spec["output_col"], **params)  # type: ignore[arg-type]
            self.features.append(feat)
        return self

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        if not self.features:
            self.build()
        for f in self.features:
            f.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.features:
            self.build()
        curr = df.copy()                  # <- cumulative working frame
        out = {}
        for f in self.features:
            s = f.transform(curr)         # read from curr (has prior outputs)
            out[f.output_col] = s
            curr[f.output_col] = s        # <- make new output available downstream
        return pd.DataFrame(out, index=df.index)


    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
