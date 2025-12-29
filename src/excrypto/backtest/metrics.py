from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _infer_ann_fac(idx: pd.DatetimeIndex) -> float:
    dt = np.median(np.diff(idx.values).astype("timedelta64[s]").astype(int))
    if dt <= 0:
        return 0.0
    secs_per_year = 365.25 * 24 * 3600
    return secs_per_year / dt


def _drawdown(eq: pd.Series) -> pd.Series:
    peak = eq.cummax()
    return (eq / peak) - 1.0


def summarize(bt: pd.DataFrame, pnl_col: str = "pnl_net", eq_col: str = "equity") -> dict[str, Any]:
    if pnl_col not in bt.columns or eq_col not in bt.columns:
        raise ValueError(f"Backtest missing required cols: '{pnl_col}', '{eq_col}'")

    pnl = bt[pnl_col].fillna(0.0)
    eq = bt[eq_col].fillna(1.0)

    ann_fac = _infer_ann_fac(bt.index) if isinstance(bt.index, pd.DatetimeIndex) else 0.0

    mu = float(pnl.mean())
    sd = float(pnl.std(ddof=0))
    sharpe = float((mu / sd) * np.sqrt(ann_fac)) if (sd > 0 and ann_fac > 0) else 0.0

    dd = _drawdown(eq)
    out = {
        "bars": int(len(bt)),
        "ann_fac": float(ann_fac),
        "mean_ret_bar": mu,
        "vol_bar": sd,
        "sharpe": sharpe,
        "equity_end": float(eq.iloc[-1]) if len(eq) else 1.0,
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
    }

    if "turnover" in bt.columns:
        out["avg_turnover"] = float(bt["turnover"].fillna(0.0).mean())

    if "gross_leverage" in bt.columns:
        out["avg_gross_leverage"] = float(bt["gross_leverage"].fillna(0.0).mean())

    return out


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))
