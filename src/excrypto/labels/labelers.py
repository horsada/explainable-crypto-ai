# src/excrypto/labels/labelers.py
from __future__ import annotations
import numpy as np
import pandas as pd

def fixed_horizon_return(
    price: pd.Series,
    horizon: int = 24,
    as_class: bool = True,
    thr: float = 0.0,            # classification threshold on forward log-return
) -> pd.Series:
    """Forward log-return over `horizon` bars; classify by sign (>thr = 1, < -thr = -1, else 0)."""
    logp = np.log(price.astype(float))
    fwd = logp.shift(-horizon) - logp
    if not as_class:
        return fwd.rename(f"fh_ret_{horizon}")
    lab = pd.Series(np.where(fwd > thr, 1, np.where(fwd < -thr, -1, 0)), index=price.index)
    return lab.rename(f"fh_lbl_{horizon}")

def triple_barrier(
    price: pd.Series,
    horizon: int = 24,
    up_mult: float = 2.0,
    dn_mult: float = 2.0,
    vol_window: int = 50,
) -> pd.Series:
    """
    Lopez de Prado-style triple barrier (simplified):
    - Vertical barrier at `horizon`.
    - Dynamic horizontal barriers: entry * exp(±mult * rolling_vol), where vol = std(log-returns, window).
    - Label  1 if upper hit first, -1 if lower hit first, else 0 at horizon.
    """
    price = price.astype(float)
    logret = np.log(price).diff()
    vol = logret.rolling(vol_window, min_periods=max(5, vol_window//5)).std(ddof=0).fillna(method="bfill")
    idx = price.index.to_numpy()
    p = price.to_numpy()
    v = vol.to_numpy()

    out = np.zeros(len(p), dtype=int)
    for i in range(len(p) - horizon):
        p0 = p[i]
        # dynamic barriers
        up = p0 * np.exp(up_mult * v[i])
        dn = p0 * np.exp(-dn_mult * v[i])
        # scan window (exclusive of i, inclusive of i+horizon)
        sl = slice(i+1, i+horizon+1)
        # first touch (upper/lower)
        path = p[sl]
        hit_up = np.argmax(path >= up) if np.any(path >= up) else 0
        hit_dn = np.argmax(path <= dn) if np.any(path <= dn) else 0
        if hit_up and hit_dn:
            out[i] = 1 if hit_up < hit_dn else -1
        elif hit_up:
            out[i] = 1
        elif hit_dn:
            out[i] = -1
        else:
            # no touch: fall back to sign of horizon return
            out[i] = int(np.sign(np.log(p[i+horizon]) - np.log(p0)))
    return pd.Series(out, index=price.index, name=f"tb_lbl_h{horizon}_u{up_mult}_d{dn_mult}_w{vol_window}")
