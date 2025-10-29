from __future__ import annotations
import numpy as np
from typing import Iterator, Tuple

class PurgedKFold:
    """
    Time-aware CV splitter with purge/embargo (Lopez de Prado).
    Assumes X is indexed in time order already.
    """
    def __init__(self, n_splits: int = 5, purge: int = 0, embargo: int = 0):
        if n_splits < 2:
            raise ValueError("n_splits >= 2")
        self.n_splits, self.purge, self.embargo = n_splits, purge, embargo

    def split(self, X) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        idx = np.arange(n)
        current = 0
        cuts = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            cuts.append((start, stop))
            current = stop

        for k, (start, stop) in enumerate(cuts):
            # test = [start:stop]
            test_idx = idx[start:stop]
            # validation purge/embargo around test
            left = max(0, start - self.purge)
            right = min(n, stop + self.embargo)
            train_mask = np.ones(n, dtype=bool)
            train_mask[left:right] = False
            train_idx = idx[train_mask]
            yield train_idx, test_idx
