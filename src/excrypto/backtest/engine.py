from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class BacktestConfig:
    fee_bps: float = 1.0            # 1 bps per notional per trade side
    slippage_bps: float = 1.0       # simple bps impact per turnover
    latency_bars: int = 1           # execute next bar
    target_vol_ann: float = 0.20    # 20% annual vol-target
    max_leverage: float = 3.0
    vol_lookback: int = 60          # bars (rolling stdev on returns)
    tzaware: bool = True

def _infer_bar_annualization(idx: pd.DatetimeIndex) -> float:
    """Return bars per year (used for vol annualization)."""
    dt = np.median(np.diff(idx.values).astype('timedelta64[s]').astype(int))
    if dt <= 0: raise ValueError("Non-increasing timestamps")
    secs_per_year = 365.25 * 24 * 3600
    return secs_per_year / dt

def _shift_for_latency(sig: pd.Series, bars: int) -> pd.Series:
    return sig.shift(bars)

def _calc_simple_returns(px: pd.Series) -> pd.Series:
    # close-to-close simple returns; align to t with px_t / px_{t-1} - 1
    return px.pct_change()

def _vol_target_weights(signal: pd.Series, ret: pd.Series, cfg: BacktestConfig, ann_fac: float) -> pd.Series:
    # rolling vol on returns; avoid lookahead by using shift
    rv = ret.rolling(cfg.vol_lookback).std().shift(1)
    # convert to annualized
    ann_vol = rv * np.sqrt(ann_fac)
    raw_w = signal / ann_vol.replace(0, np.nan)
    w = raw_w * cfg.target_vol_ann
    w = w.clip(lower=-cfg.max_leverage, upper=cfg.max_leverage)
    return w.fillna(0.0)

def _apply_costs(pnl_gross: pd.Series, weight: pd.Series, cfg: BacktestConfig) -> pd.Series:
    # turnover = |w_t - w_{t-1}|; cost per side ~ (fee + slippage) bps
    turn = (weight - weight.shift(1)).abs().fillna(0.0)
    cost_bps = (cfg.fee_bps + cfg.slippage_bps)
    costs = turn * (cost_bps * 1e-4)
    return pnl_gross - costs

def backtest_single(df: pd.DataFrame, cfg: BacktestConfig, price_col="close", signal_col="signal") -> pd.DataFrame:
    """
    df (indexed by tz-aware timestamps) with columns: 'close', 'signal'
    Returns DataFrame with columns: ret, weight, pnl_gross, pnl_net, equity
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")
    if cfg.tzaware and df.index.tz is None:
        raise ValueError("Index must be timezone-aware (UTC recommended)")

    df = df.sort_index().copy()
    ann_fac = _infer_bar_annualization(df.index)

    # latency: execute after cfg.latency_bars
    sig_exec = _shift_for_latency(df[signal_col], cfg.latency_bars).fillna(0.0)

    # returns and vol-targeted weights
    ret = _calc_simple_returns(df[price_col])
    w = _vol_target_weights(sig_exec, ret, cfg, ann_fac)

    # PnL (gross): w_{t-1} * ret_t  (position set at open of bar t uses ret_t)
    pnl_gross = (w.shift(1) * ret).fillna(0.0)

    # Costs & equity
    pnl_net = _apply_costs(pnl_gross, w, cfg)
    equity = (1.0 + pnl_net).cumprod()

    out = pd.DataFrame({
        "ret": ret,
        "weight": w,
        "pnl_gross": pnl_gross,
        "pnl_net": pnl_net,
        "equity": equity,
    })
    return out

def backtest_multi(panel: pd.DataFrame, cfg: BacktestConfig,
                   price_col="close", signal_col="signal") -> pd.DataFrame:
    if "symbol" not in panel.columns:
        raise ValueError("Require 'symbol' column for multi-asset backtest.")

    results = []
    for sym, df_sym in panel.groupby("symbol"):
        res = backtest_single(df_sym[[price_col, signal_col]].copy(), cfg,
                              price_col, signal_col)
        res["symbol"] = sym
        results.append(res)

    # Build MultiIndex: (timestamp, symbol)
    joined = pd.concat(results)
    joined = joined.set_index("symbol", append=True).sort_index()  # index -> (timestamp, symbol)

    # Equal-weight portfolio across symbols each bar
    port = joined["pnl_net"].unstack("symbol").fillna(0.0).mean(axis=1)

    return pd.DataFrame({
        "pnl_net": port,
        "equity": (1.0 + port).cumprod()
    })

