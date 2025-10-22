from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------- helpers ---------------------------- #

def _ensure_dt_index(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Ensure df is indexed by a tz-aware DatetimeIndex (UTC recommended)."""
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            raise ValueError("DatetimeIndex must be timezone-aware (e.g., UTC).")
        return df.sort_index()

    if ts_col not in df.columns:
        raise ValueError(f"DataFrame must have a DatetimeIndex or a '{ts_col}' column.")

    ts = pd.to_datetime(df[ts_col], utc=True)
    out = df.copy()
    out.index = ts
    return out.sort_index()


def _to_tdelta(x: Union[str, int, float, pd.Timedelta]) -> pd.Timedelta:
    """Accept '30min'/'1D'/Timedelta or minutes (int/float) and return Timedelta."""
    if isinstance(x, pd.Timedelta):
        return x
    if isinstance(x, (int, float)):
        return pd.to_timedelta(x, unit="m")
    return pd.to_timedelta(x)


def _time_slice_index(idx: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """Return integer positions inside [start, end] inclusive."""
    # idx is sorted DatetimeIndex
    left = idx.searchsorted(start, side="left")
    right = idx.searchsorted(end, side="right")  # exclusive
    return np.arange(left, right, dtype=int)


# ---------------------------- data classes ---------------------------- #

@dataclass(frozen=True)
class TimeWindow:
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class SplitFold:
    train: TimeWindow
    valid: TimeWindow
    train_idx: np.ndarray
    valid_idx: np.ndarray


# ---------------------------- core API ---------------------------- #

def build_rolling_splits(
    df: pd.DataFrame,
    *,
    train: Union[str, int, float, pd.Timedelta],
    valid: Union[str, int, float, pd.Timedelta],
    step: Optional[Union[str, int, float, pd.Timedelta]] = None,
    embargo: Union[str, int, float, pd.Timedelta] = "0s",
    ts_col: str = "timestamp",
    min_obs: int = 1,
) -> List[SplitFold]:
    """
    Create rolling **train→valid** splits with an optional **embargo**.

    Parameters
    ----------
    train : window length for training (e.g. "90D", "48h", 120 (minutes))
    valid : window length for validation
    step  : how far to move the window each fold; defaults to `valid` if None
    embargo : time gap removed from training around the validation window
    min_obs : minimum number of observations required in each subset

    Returns
    -------
    List[SplitFold]
    """
    df = _ensure_dt_index(df, ts_col=ts_col)
    idx = df.index

    train_td = _to_tdelta(train)
    valid_td = _to_tdelta(valid)
    step_td = _to_tdelta(step if step is not None else valid)
    embargo_td = _to_tdelta(embargo)

    folds: List[SplitFold] = []

    if len(idx) == 0:
        return folds

    t0 = idx.min()
    t_last = idx.max()

    cur_valid_start = t0 + train_td + embargo_td
    while True:
        cur_valid_end = cur_valid_start + valid_td
        # Stop if validation window exceeds data
        if cur_valid_end > t_last:
            break

        # Raw train window = [valid_start - embargo - train_len, valid_start - embargo]
        raw_train_end = cur_valid_start - embargo_td
        raw_train_start = raw_train_end - train_td
        # If embargo > 0, also remove tail after validation:
        post_valid_embargo_start = cur_valid_end
        post_valid_embargo_end = cur_valid_end + embargo_td

        # Index positions
        train_pos = _time_slice_index(idx, raw_train_start, raw_train_end)
        valid_pos = _time_slice_index(idx, cur_valid_start, cur_valid_end)

        # Remove post-valid embargo from the right side of train, if it overlaps (rare on time axis)
        if embargo_td > pd.Timedelta(0):
            post_embargo_pos = _time_slice_index(idx, post_valid_embargo_start, post_valid_embargo_end)
            # train set must exclude any overlap with post-valid embargo
            if post_embargo_pos.size:
                train_pos = np.setdiff1d(train_pos, post_embargo_pos, assume_unique=False)

        # sanity / min sizes
        if train_pos.size >= min_obs and valid_pos.size >= min_obs:
            fold = SplitFold(
                train=TimeWindow(train_pos.size and idx[train_pos[0]] or raw_train_start,
                                 train_pos.size and idx[train_pos[-1]] or raw_train_end),
                valid=TimeWindow(valid_pos.size and idx[valid_pos[0]] or cur_valid_start,
                                 valid_pos.size and idx[valid_pos[-1]] or cur_valid_end),
                train_idx=train_pos,
                valid_idx=valid_pos,
            )
            folds.append(fold)

        # advance
        cur_valid_start = cur_valid_start + step_td

    return folds


def make_purged_kfold_indices(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    embargo: Union[str, int, float, pd.Timedelta] = "0s",
    ts_col: str = "timestamp",
) -> List[SplitFold]:
    """
    Time-ordered **purged K-Fold** on contiguous blocks of the index.
    Each fold's validation block is embargoed on both sides from the training set.
    """
    df = _ensure_dt_index(df, ts_col=ts_col)
    idx = df.index

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    # Cut the time index into contiguous blocks
    cuts = np.linspace(0, len(idx), n_splits + 1, dtype=int)
    embargo_td = _to_tdelta(embargo)

    folds: List[SplitFold] = []
    for k in range(n_splits):
        v_start_pos, v_end_pos = cuts[k], cuts[k + 1]
        valid_idx = np.arange(v_start_pos, v_end_pos, dtype=int)

        v_start_ts = idx[valid_idx[0]]
        v_end_ts = idx[valid_idx[-1]]

        # Embargo zones around validation
        left_cut_ts = v_start_ts - embargo_td
        right_cut_ts = v_end_ts + embargo_td

        all_idx = np.arange(len(idx), dtype=int)
        left_train = all_idx[idx <= left_cut_ts]
        right_train = all_idx[idx >= right_cut_ts]
        train_idx = np.concatenate([left_train, right_train])

        folds.append(
            SplitFold(
                train=TimeWindow(idx[train_idx[0]], idx[train_idx[-1]]) if train_idx.size else
                      TimeWindow(idx[0], idx[0]),
                valid=TimeWindow(v_start_ts, v_end_ts),
                train_idx=train_idx,
                valid_idx=valid_idx,
            )
        )
    return folds


# ---------------------------- small validators ---------------------------- #

def assert_no_overlap(folds: Iterable[SplitFold]) -> None:
    """Raise if any fold has overlapping train/valid indices."""
    for i, f in enumerate(folds):
        inter = np.intersect1d(f.train_idx, f.valid_idx)
        if inter.size:
            raise ValueError(f"Fold {i} has overlap between train and valid indices.")


def assert_increasing_windows(folds):
    last_end = None
    for i, f in enumerate(folds):
        if last_end is not None and f.valid.start < last_end:   # allow equality
            raise ValueError(f"Fold {i} not increasing: {f.valid.start} < {last_end}")
        last_end = f.valid.end
