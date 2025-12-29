# src/excrypto/labels/labelers.py
from __future__ import annotations

import numpy as np
import pandas as pd


def fixed_horizon_return(
    price: pd.Series,
    *,
    horizon: int = 24,
    as_class: bool = True,
    thr: float = 0.0,
    name: str | None = None,
) -> pd.Series:
    """
    Forward log-return over `horizon` bars.

    - Regression: returns forward log-return.
    - Classification:  1 if fwd > thr, -1 if fwd < -thr, else 0.

    Notes:
    - The last `horizon` rows will be NaN (regression) or 0 (classification) by default.
      (You can drop them downstream via your NaN policy.)
    """
    if horizon <= 0:
        raise ValueError("horizon must be > 0")

    p = price.astype(float)
    logp = np.log(p)
    fwd = logp.shift(-horizon) - logp

    if not as_class:
        out_name = name or f"fh_ret_{horizon}"
        return fwd.rename(out_name)

    # Classification: keep unlabeled tail as 0 (common for pipelines that later drop/ignore)
    arr = fwd.to_numpy()
    labels = np.where(arr > thr, 1, np.where(arr < -thr, -1, 0)).astype(int)
    out_name = name or f"fh_lbl_{horizon}"
    return pd.Series(labels, index=price.index, name=out_name)


def triple_barrier(
    price: pd.Series,
    *,
    horizon: int = 24,
    up_mult: float = 2.0,
    dn_mult: float = 2.0,
    vol_window: int = 50,
    min_periods: int | None = None,
    tail_value: int = 0,
    name: str | None = None,
) -> pd.Series:
    """
    Simplified triple barrier labeling.

    Structural / best-practice changes vs your current version:
    - Fixes a bug where "hit at first step" was treated as "no hit" (because argmax==0 is falsy).
    - Removes unused variables and makes edge cases explicit.
    - Avoids backfilling volatility (which can leak future info). Uses ffill then fills remaining with 0.
    - Clearly defines how the last `horizon` rows are handled (tail_value).
    """
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if vol_window <= 1:
        raise ValueError("vol_window must be > 1")

    p = price.astype(float)
    logret = np.log(p).diff()

    mp = min_periods if min_periods is not None else max(5, vol_window // 5)
    vol = (
        logret.rolling(vol_window, min_periods=mp)
        .std(ddof=0)
        .ffill()               # do NOT bfill (future leakage)
        .fillna(0.0)
    )

    p_arr = p.to_numpy()
    v_arr = vol.to_numpy()
    n = len(p_arr)

    out = np.full(n, tail_value, dtype=int)
    last_start = n - horizon
    if last_start <= 0:
        # Not enough data to label anything
        out_name = name or f"tb_lbl_h{horizon}_u{up_mult}_d{dn_mult}_w{vol_window}"
        return pd.Series(out, index=price.index, name=out_name)

    for i in range(last_start):
        p0 = p_arr[i]
        vi = v_arr[i]

        up = p0 * np.exp(up_mult * vi)
        dn = p0 * np.exp(-dn_mult * vi)

        path = p_arr[i + 1 : i + horizon + 1]

        hit_up_mask = path >= up
        hit_dn_mask = path <= dn

        hit_up_idx = int(np.argmax(hit_up_mask)) if hit_up_mask.any() else None
        hit_dn_idx = int(np.argmax(hit_dn_mask)) if hit_dn_mask.any() else None

        if hit_up_idx is not None and hit_dn_idx is not None:
            out[i] = 1 if hit_up_idx < hit_dn_idx else -1
        elif hit_up_idx is not None:
            out[i] = 1
        elif hit_dn_idx is not None:
            out[i] = -1
        else:
            # No barrier hit: fall back to sign of horizon return
            out[i] = int(np.sign(np.log(p_arr[i + horizon]) - np.log(p0)))

    out_name = name or f"tb_lbl_h{horizon}_u{up_mult}_d{dn_mult}_w{vol_window}"
    return pd.Series(out, index=price.index, name=out_name)
