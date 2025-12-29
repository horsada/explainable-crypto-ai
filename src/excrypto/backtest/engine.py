from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    fee_bps: float = 1.0
    slippage_bps: float = 1.0
    latency_bars: int = 1
    target_vol_ann: float = 0.20
    max_leverage: float = 3.0
    vol_lookback: int = 60
    tzaware: bool = True


def _infer_bar_annualization(idx: pd.DatetimeIndex) -> float:
    dt = np.median(np.diff(idx.values).astype("timedelta64[s]").astype(int))
    if dt <= 0:
        raise ValueError("Non-increasing timestamps")
    secs_per_year = 365.25 * 24 * 3600
    return secs_per_year / dt


def _shift_for_latency(sig: pd.Series, bars: int) -> pd.Series:
    return sig.shift(bars)


def _calc_simple_returns(px: pd.Series) -> pd.Series:
    return px.pct_change()


def _vol_target_position(signal: pd.Series, ret: pd.Series, cfg: BacktestConfig, ann_fac: float) -> pd.Series:
    rv = ret.rolling(cfg.vol_lookback).std().shift(1)
    ann_vol = rv * np.sqrt(ann_fac)
    raw = signal / ann_vol.replace(0, np.nan)
    pos = (raw * cfg.target_vol_ann).clip(-cfg.max_leverage, cfg.max_leverage)
    return pos.fillna(0.0)


def _apply_costs(pnl_gross: pd.Series, pos: pd.Series, cfg: BacktestConfig) -> tuple[pd.Series, pd.Series]:
    turnover = (pos - pos.shift(1)).abs().fillna(0.0)
    cost_bps = cfg.fee_bps + cfg.slippage_bps
    costs = turnover * (cost_bps * 1e-4)
    pnl_net = pnl_gross - costs
    return pnl_net, turnover


def backtest_single(df: pd.DataFrame, cfg: BacktestConfig, price_col="close", signal_col="signal") -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")
    if cfg.tzaware and df.index.tz is None:
        raise ValueError("Index must be timezone-aware (UTC recommended)")

    df = df.sort_index().copy()
    ann_fac = _infer_bar_annualization(df.index)

    sig_exec = _shift_for_latency(df[signal_col], cfg.latency_bars).fillna(0.0)

    ret = _calc_simple_returns(df[price_col])
    pos = _vol_target_position(sig_exec, ret, cfg, ann_fac)

    pnl_gross = (pos.shift(1) * ret).fillna(0.0)
    pnl_net, turnover = _apply_costs(pnl_gross, pos, cfg)

    equity = (1.0 + pnl_net).cumprod()
    dd = (equity / equity.cummax()) - 1.0

    return pd.DataFrame(
        {
            "ret": ret,
            "signal_exec": sig_exec,
            "position": pos,
            "turnover": turnover,
            "pnl_gross": pnl_gross,
            "pnl_net": pnl_net,
            "equity": equity,
            "drawdown": dd,
        }
    )


def backtest_multi(panel: pd.DataFrame, cfg: BacktestConfig, price_col="close", signal_col="signal") -> pd.DataFrame:
    if "symbol" not in panel.columns:
        raise ValueError("Require 'symbol' column for multi-asset backtest.")

    results = []
    for sym, df_sym in panel.groupby("symbol"):
        res = backtest_single(df_sym[[price_col, signal_col]].copy(), cfg, price_col, signal_col)
        res["symbol"] = sym
        results.append(res)

    joined = pd.concat(results).set_index("symbol", append=True).sort_index()  # (ts, symbol)

    pnl_mat = joined["pnl_net"].unstack("symbol").fillna(0.0)
    port_pnl = pnl_mat.mean(axis=1)  # keep your equal-weight aggregation for now
    equity = (1.0 + port_pnl).cumprod()
    dd = (equity / equity.cummax()) - 1.0

    return pd.DataFrame({"pnl_net": port_pnl, "equity": equity, "drawdown": dd})
