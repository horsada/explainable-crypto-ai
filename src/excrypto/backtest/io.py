# src/excrypto/backtest/io.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json

import pandas as pd

from excrypto.utils.paths import RunPaths


@dataclass(frozen=True)
class TradeInputs:
    prices_path: Path
    signals_path: Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _to_utc_index(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        if ts_col not in df.columns:
            raise ValueError(f"Missing '{ts_col}' column and index is not DatetimeIndex.")
        idx = pd.DatetimeIndex(pd.to_datetime(df[ts_col], utc=True))
        df = df.drop(columns=[ts_col])

    # idx is now guaranteed DatetimeIndex with tz
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    out = df.copy()
    out.index = idx
    return out.sort_index()



# src/excrypto/backtest/io.py

def _resolve_via_latest(
    *,
    runs_root: Path,
    snapshot: str,
    stage: str,
    timeframe: str,
    universe: str,
    artifact_key: str,  # "signals" or "panel"
) -> Path:
    uni_dir = runs_root / snapshot / stage / timeframe / universe
    latest = uni_dir / "latest_manifest.json"
    if not latest.exists():
        raise FileNotFoundError(f"Missing latest pointer: {latest}")

    lm = _read_json(latest)

    manifest_str = (lm.get("paths") or {}).get("manifest")
    if not isinstance(manifest_str, str) or not manifest_str:
        raise ValueError(f"latest_manifest.json missing 'paths.manifest': {latest}")

    man_path = Path(manifest_str)
    if not man_path.is_absolute():
        # if manifest_str already starts with "runs/", don't prefix runs_root again
        if man_path.parts and man_path.parts[0] == runs_root.name:
            man_path = man_path.resolve()
        else:
            man_path = (runs_root / man_path).resolve()

    if not man_path.exists():
        raise FileNotFoundError(f"Manifest not found from latest pointer: {man_path}")

    m = _read_json(man_path)
    art_str = (m.get("paths") or {}).get(artifact_key)
    if not isinstance(art_str, str) or not art_str:
        raise ValueError(f"Manifest missing 'paths.{artifact_key}': {man_path}")

    art_path = Path(art_str)
    if not art_path.is_absolute():
        # if manifest_str already starts with "runs/", don't prefix runs_root again
        if art_path.parts and art_path.parts[0] == runs_root.name:
            art_path = art_path.resolve()
        else:
            art_path = (runs_root / art_path).resolve()
    if not art_path.exists():
        raise FileNotFoundError(f"Artifact not found: {art_path}")

    return art_path


def resolve_inputs(
    *,
    snapshot: str,
    symbols: tuple[str, ...],
    timeframe: str,
    runs_root: Path,
    exchange: str,
    signals_strategy: str = "predict",
    prices_strategy: str = "snapshot",
    signals_path: Optional[Path] = None,
    prices_path: Optional[Path] = None,
) -> TradeInputs:
    # Universe folder is independent of params hash (we want to find latest pointer)
    # inside resolve_inputs(...)

    universe = RunPaths(
        snapshot=snapshot,
        strategy=signals_strategy,
        symbols=symbols,
        timeframe=timeframe,
        params=None,
        runs_root=runs_root,
    ).universe

    if signals_path is None:
        signals_path = _resolve_via_latest(
            runs_root=runs_root,
            snapshot=snapshot,
            stage=signals_strategy,
            timeframe=timeframe,
            universe=universe,
            artifact_key="signals",
        )

    if prices_path is None:
        prices_path = _resolve_via_latest(
            runs_root=runs_root,
            snapshot=snapshot,
            stage=prices_strategy,
            timeframe=timeframe,
            universe=universe,
            artifact_key="panel",
        )

    return TradeInputs(prices_path=signals_path.__class__(prices_path), signals_path=signals_path.__class__(signals_path))



def load_trade_frame(
    *,
    prices_path: Path,
    signals_path: Path,
    price_col: str = "close",
    signal_col: str = "signal",
) -> pd.DataFrame:
    """
    Builds the tradeable frame with columns:
      - price_col (e.g. close)
      - signal_col (e.g. signal)
      - symbol
    Indexed by UTC timestamp.
    """
    if not prices_path.exists():
        raise FileNotFoundError(f"Prices not found: {prices_path}")
    if not signals_path.exists():
        raise FileNotFoundError(f"Signals not found: {signals_path}")

    prices = pd.read_parquet(prices_path)
    signals = pd.read_parquet(signals_path)

    prices = _to_utc_index(prices, ts_col="timestamp")
    signals = _to_utc_index(signals, ts_col="timestamp")

    # require schemas
    for c in ["symbol", price_col]:
        if c not in prices.columns:
            raise ValueError(f"Prices missing required column '{c}' in {prices_path}")
    for c in ["symbol", signal_col]:
        if c not in signals.columns:
            raise ValueError(f"Signals missing required column '{c}' in {signals_path}")

    prices = prices[[price_col, "symbol"]].reset_index().rename(columns={"index": "timestamp"})
    signals = signals[[signal_col, "symbol"]].reset_index().rename(columns={"index": "timestamp"})

    joined = (
        prices.merge(signals, on=["timestamp", "symbol"], how="inner", validate="one_to_one")
        .set_index("timestamp")
        .sort_index()
    )

    if joined.empty:
        raise ValueError(
            "Trade frame is empty after joining prices and signals. "
            "Check that timestamp+symbol keys match between panel and signals."
        )

    return joined
