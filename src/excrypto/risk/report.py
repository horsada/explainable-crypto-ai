from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from excrypto.risk.metrics import (
    drawdown_curve, max_drawdown, ann_vol,
    var_historic, es_historic, var_cornish_fisher,
    kupiec_pof_test, christoffersen_ind_test
)

def _infer_ann_fac(idx: pd.DatetimeIndex) -> float:
    dt_sec = np.median(np.diff(idx.values).astype("timedelta64[s]").astype(int))
    if dt_sec <= 0:
        return 252.0
    return (365.25 * 24 * 3600) / dt_sec

def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _plot_equity(returns: pd.Series, out_png: str):
    eq = (1 + returns.fillna(0)).cumprod()
    plt.figure()
    eq.plot()
    plt.title("Equity Curve")
    plt.xlabel("Time"); plt.ylabel("Equity")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def _plot_drawdown(returns: pd.Series, out_png: str):
    dd = drawdown_curve(returns)["drawdown"]
    plt.figure()
    dd.plot()
    plt.title("Drawdown")
    plt.xlabel("Time"); plt.ylabel("Drawdown")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def _var_breaches(returns: pd.Series, alpha: float):
    VaR = var_historic(returns, alpha)
    viol = (-returns > VaR)
    return VaR, viol

def write_risk_report_md(
    returns: pd.Series,
    weights: pd.Series | None,
    title: str,
    out_dir: str,
    alpha: float = 0.99,
    ann_fac: float = 252.0
) -> str:
    _ensure_dir(out_dir)
    # Plots
    eq_png = os.path.join(out_dir, "equity.png")
    dd_png = os.path.join(out_dir, "drawdown.png")
    _plot_equity(returns, eq_png)
    _plot_drawdown(returns, dd_png)

    # Metrics
    dd = max_drawdown(returns)

    if ann_fac is None:
        ann_fac = _infer_ann_fac(returns.index)

    vol = ann_vol(returns, ann_fac=ann_fac)
    var_h = var_historic(returns, alpha)
    es_h = es_historic(returns, alpha)
    var_cf = var_cornish_fisher(returns, alpha)
    VaR, viol = _var_breaches(returns, alpha)
    p_k = kupiec_pof_test(viol, alpha=1-alpha)
    p_c = christoffersen_ind_test(viol)

    # Markdown
    md = f"""# {title}

## Summary
- Observations: {returns.dropna().shape[0]}
- Ann. Vol: {vol:.4f}
- Max Drawdown: {dd:.4f}
- VaR (hist, {alpha:.2%}): {var_h:.4f}
- ES  (hist, {alpha:.2%}): {es_h:.4f}
- VaR (Cornish - Fisher, {alpha:.2%}): {var_cf:.4f}

## VaR Backtests
- Kupiec p-value: {p_k if p_k==p_k else 'NaN'}
- Christoffersen p-value: {p_c if p_c==p_c else 'NaN'}

## Plots
![Equity]({os.path.basename(eq_png)})
![Drawdown]({os.path.basename(dd_png)})
"""
    md_path = os.path.join(out_dir, "risk_report.md")
    with open(md_path, "w") as f:
        f.write(md)
    return md_path
