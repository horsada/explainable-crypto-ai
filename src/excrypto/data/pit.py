from __future__ import annotations

"""
Point-in-time helpers:
- safe_final_bar: prevent using the live/incomplete bar for features (use t-1).
- asof_join: 'as-of' merge with optional publication lag.
- assert_monotonic: quick data sanity check for time series per symbol.
"""

import pandas as pd


def assert_monotonic(df: pd.DataFrame, symbol_col: str = "symbol", ts_col: str = "timestamp") -> None:
    """
    Raise if timestamps are not strictly increasing (no ties, no reversals)
    in the ORIGINAL row order for each symbol.
    """
    def _viol(s: pd.Series) -> bool:
        # Strict: every step must increase
        d = s.diff()
        # first diff is NaT; ignore it. Any <= 0 means violation.
        return (d.iloc[1:] <= pd.Timedelta(0)).any()

    viol = (
        df.groupby(symbol_col, sort=False, as_index=False)[ts_col]
          .apply(_viol)
    )

    if viol.any():
        bad_syms = list(viol[viol].index)
        raise ValueError(f"Non-monotonic {ts_col} order for symbols: {bad_syms}")


def safe_final_bar(
    prices: pd.DataFrame,
    symbol_col: str = "symbol",
    ts_col: str = "timestamp",
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Use only information available at t-1 when building features at timestamp t.
    Adds a new column '<price_col>_t1' which is shifted by one bar per symbol.
    """
    out = prices.sort_values([symbol_col, ts_col]).copy()
    out[f"{price_col}_t1"] = out.groupby(symbol_col)[price_col].shift(1)
    return out


def asof_join(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    by: list[str] = ["symbol"],
    ts_col: str = "timestamp",
    pub_lag: str = "0s",
    suffixes: tuple[str, str] = ("", "_r"),
) -> pd.DataFrame:
    """
    For each row in left_df at time t, attach the most recent row from right_df
    with timestamp <= (t - pub_lag). If pub_lag is "0s", it's a standard backward as-of join.

    Notes:
      - Both frames must be sorted by [by, ts_col].
      - If right_df is empty/None, left_df is returned unchanged.
    """
    if right_df is None or right_df.empty:
        return left_df

    left = left_df.sort_values(by + [ts_col]).copy()
    right = right_df.sort_values(by + [ts_col]).copy()

    if pub_lag and pub_lag != "0s":
        # Information in 'right' becomes available only after pub_lag
        right = right.copy()
        right[ts_col] = right[ts_col] + pd.to_timedelta(pub_lag)

    merged = pd.merge_asof(
        left,
        right,
        by=by,
        on=ts_col,
        direction="backward",
        allow_exact_matches=True,
        suffixes=suffixes,
    )
    return merged
