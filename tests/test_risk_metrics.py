# tests/test_risk_metrics.py
import pandas as pd, numpy as np
from excrypto.risk.metrics import (
    max_drawdown, ann_vol, var_historic, es_historic,
    kupiec_pof_test, christoffersen_ind_test, drawdown_curve
)

def test_drawdown_and_vol():
    r = pd.Series([0.01]*50 + [-0.02]*10 + [0.0]*40)
    dd = max_drawdown(r)
    assert dd <= 0  # drawdowns are negative
    dv = drawdown_curve(r)
    assert "drawdown" in dv and dv["drawdown"].min() == dd
    assert ann_vol(r, ann_fac=252) > 0

def test_var_es_monotonic():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 1000))
    var95 = var_historic(r, 0.95)
    var99 = var_historic(r, 0.99)
    es95  = es_historic(r, 0.95)
    assert var99 >= var95  # higher confidence => larger VaR
    assert es95 >= var95   # ES ≥ VaR

def test_var_backtests():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, 2000))
    # pretend VaR_99 is constant estimate from sample
    from excrypto.risk.metrics import var_historic
    VaR = var_historic(r.iloc[:1000], 0.99)
    viol = (-r.iloc[1000:] > VaR)
    p_k = kupiec_pof_test(viol, alpha=0.01)
    p_c = christoffersen_ind_test(viol)
    assert 0 <= p_k <= 1 and 0 <= p_c <= 1
