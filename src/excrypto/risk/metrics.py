from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------- basic risk stats ---------- #

def drawdown_curve(returns: pd.Series) -> pd.DataFrame:
    eq = (1 + returns.fillna(0)).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return pd.DataFrame({"equity": eq, "peak": peak, "drawdown": dd})

def max_drawdown(returns: pd.Series) -> float:
    return drawdown_curve(returns)["drawdown"].min()

def ann_vol(returns: pd.Series, ann_fac: float = 252.0) -> float:
    return returns.std(ddof=0) * np.sqrt(ann_fac)

def turnover(weights: pd.Series) -> pd.Series:
    return weights.diff().abs().fillna(0.0)

# ---------- VaR / ES ---------- #

def var_historic(returns: pd.Series, level: float = 0.99) -> float:
    """One-period (same horizon as returns) historical VaR (>0 = loss)."""
    q = np.nanpercentile(-returns.dropna().values, level * 100)
    return float(q)

def es_historic(returns: pd.Series, level: float = 0.99) -> float:
    losses = -returns.dropna().values
    thresh = np.nanpercentile(losses, level * 100)
    tail = losses[losses >= thresh]
    return float(np.mean(tail)) if tail.size else float("nan")

def var_cornish_fisher(returns: pd.Series, level: float = 0.99) -> float:
    """Cornish–Fisher adjusted VaR (one-period)."""
    x = returns.dropna().values
    mu, sigma = np.mean(x), np.std(x, ddof=0)
    if sigma == 0 or np.isnan(sigma): return float("nan")
    z = norm.ppf(level)
    s = (x - mu) / sigma
    skew = np.mean(s**3)
    kurt = np.mean(s**4) - 3.0
    z_cf = (z +
            (z**2 - 1)*skew/6 +
            (z**3 - 3*z)*kurt/24 -
            (2*z**3 - 5*z)*(skew**2)/36)
    return float(-(mu + z_cf * sigma))

# ---------- VaR backtests ---------- #

def kupiec_pof_test(violations: pd.Series, alpha: float) -> float:
    """
    Kupiec Proportion of Failures test p-value.
    violations: boolean Series (True if loss > VaR) aligned to returns index.
    """
    v = violations.dropna()
    n = len(v)
    x = int(v.sum())
    if n == 0: return np.nan
    # Likelihood ratio
    pi_hat = x / n if n else 0
    def _L(pi): return (pi**x) * ((1-pi)**(n-x))
    LR = -2 * np.log((_L(alpha) + 1e-15) / (_L(pi_hat) + 1e-15))
    # 1 df chi-square approx -> p-value
    from scipy.stats import chi2
    return float(1 - chi2.cdf(LR, df=1))

def christoffersen_ind_test(violations: pd.Series) -> float:
    """
    Christoffersen independence test p-value for violation clustering.
    """
    v = violations.dropna().astype(int).values
    if v.size < 2: return np.nan
    N00=N01=N10=N11=0
    for i in range(1, v.size):
        prev, cur = v[i-1], v[i]
        if prev==0 and cur==0: N00+=1
        elif prev==0 and cur==1: N01+=1
        elif prev==1 and cur==0: N10+=1
        else: N11+=1
    pi = (N01+N11) / max(N00+N01+N10+N11, 1)
    pi0 = N01 / max(N00+N01, 1)
    pi1 = N11 / max(N10+N11, 1)
    from scipy.stats import chi2
    L0 = ((1-pi)**(N00+N10)) * (pi**(N01+N11))
    L1 = ((1-pi0)**N00) * (pi0**N01) * ((1-pi1)**N10) * (pi1**N11)
    LR = -2*np.log((L0+1e-15)/(L1+1e-15))
    return float(1 - chi2.cdf(LR, df=1))
