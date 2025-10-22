from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import json, os


REQUIRED_COLS = ["close"]
TS_CANDIDATES = ("open_time", "timestamp")

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

    def transform(self, df: pd.DataFrame, *, compute_label: bool=False):
        df = df.copy()
        self._validate_base(df)
        if self.cfg.align_on_open_time:
            df = self._align_time(df)

        # PIT: use previous closed bar
        df["close_t1"] = df["close"].shift(1)

        self._gen_returns(df)   # should use close_t1 inside
        self._gen_lags(df)
        self._gen_rolls(df)

        X = df[self.cfg.features].copy()

        y = None
        if compute_label:
            h = self.cfg.target_horizon
            # cast to pandas' nullable Int8 RIGHT AWAY
            lbl = (df["close"].shift(-h) > df["close"]).astype("Int8")
            if h > 0:
                lbl.iloc[-h:] = pd.NA
            y = lbl
            valid_idx = X.index.intersection(y.dropna().index)
            X = X.loc[valid_idx]
            y = y.loc[valid_idx].astype(int)


        # clean features AFTER alignment
        if self.cfg.dropna:
            X = X.dropna()
            if y is not None:  # keep y aligned if any rows dropped
                y = y.loc[X.index]
        elif self.cfg.fillna is not None:
            X = X.fillna(self.cfg.fillna)

        return X, y


    def _validate_base(self, df: pd.DataFrame):
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _align_time(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ts = next((c for c in TS_CANDIDATES if c in df.columns), None)
        if ts is None:
            raise ValueError(f"Need one of {TS_CANDIDATES}")
        df["timestamp"] = pd.to_datetime(df[ts], utc=True)
        if ts != "timestamp":
            df.drop(columns=[ts], inplace=True)
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.set_index("timestamp", inplace=True, drop=False)
        return df.sort_index()

    def _gen_lags(self, df: pd.DataFrame):
        for col, lags in self.cfg.lag_features.items():
            for k in lags:
                df[f"lag_{col}_{k}"] = df[col].shift(k)

    def _gen_returns(self, df: pd.DataFrame):
        for w in self.cfg.returns_windows:
            df[f"ret_{w}"] = df["close_t1"].pct_change(w)

    def _gen_rolls(self, df: pd.DataFrame):
        for col, wins in self.cfg.rolling_mean.items():
            base = "close_t1" if col == "close" else col
            for w in wins:
                df[f"roll_mean_{col}_{w}"] = df[base].rolling(w).mean()
        for col, wins in self.cfg.rolling_std.items():
            base = "close_t1" if col == "close" else col
            for w in wins:
                df[f"roll_std_{col}_{w}"] = df[base].rolling(w).std()


def run(
    snapshot: str,
    exchange: str = "binance",
    symbols: list[str] | None = None,
    out_dir: str = "data/processed",
    compute_label: bool = False,
    cfg: FeatureConfig | None = None,
) -> str:
    cfg = cfg or FeatureConfig(features=["ret_1","lag_close_1","roll_mean_close_5"])
    # load raw OHLCV for each symbol from snapshot
    base = f"data/raw/{snapshot}/{exchange}"
    if symbols is None:
        with open(f"{base}/_universe.json") as f: symbols = json.load(f)

    pipe = FeaturePipeline(cfg)
    rows = []
    for sym in symbols:
        df = pd.read_parquet(f"{base}/{sym.replace('/','_')}/ohlcv.parquet")
        df["symbol"] = sym
        X, y = pipe.transform(df, compute_label=compute_label)
        X["symbol"] = sym
        if y is not None:
            X["label"] = y
        rows.append(X.reset_index())  # keep timestamp as a column

    feats = pd.concat(rows, ignore_index=True)
    out_path = f"{out_dir}/{snapshot}/features.parquet"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    feats.to_parquet(out_path, index=False)

    # save the config used
    with open(f"{out_dir}/{snapshot}/features_config.json","w") as f:
        json.dump(asdict(cfg), f, indent=2)
    return out_path