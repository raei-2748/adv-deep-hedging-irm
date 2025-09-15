from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class RegimeDetectorBase(ABC):
    @abstractmethod
    def detect(self, market_df: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class InvariancePenalty(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):  # -> torch.Tensor
        raise NotImplementedError

