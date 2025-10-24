from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Optional
import pandas as pd

@dataclass
class Feature(ABC):
    """Base interface for all features."""
    input_cols: Iterable[str]
    output_col: str
    requires_fit: bool = False
    fitted_: bool = field(default=False, init=False)

    def fit(self, df: pd.DataFrame) -> "Feature":
        if self.requires_fit:
            self._fit(df)
            self.fitted_ = True
        return self

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.Series:
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.Series:
        return self.fit(df).transform(df)

    def check_ready(self) -> None:
        if self.requires_fit and not self.fitted_:
            raise RuntimeError(f"{self.__class__.__name__} must be fit() before transform().")

    def _fit(self, df: pd.DataFrame) -> None:
        """Override only if requires_fit=True."""
        return

class StatelessFeature(Feature):
    requires_fit: bool = False

class StatefulFeature(Feature):
    requires_fit: bool = True
