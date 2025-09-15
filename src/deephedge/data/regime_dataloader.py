"""
Extensions to DataManager to attach simple regime labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dataloader import DataManager
from firm.regime.detector import RegimeDetector


class RegimeDataManager(DataManager):
    def get_data_with_regimes(
        self,
        method: str = "volatility_percentile",
        low_pct: float = 0.3,
        high_pct: float = 0.7,
        synthetic: bool = False,
    ):
        market, options = self.get_data(synthetic=synthetic)
        detector = RegimeDetector(method=method)
        regimes = detector.detect(market, low_pct=low_pct, high_pct=high_pct)
        market = market.copy()
        market["Regime"] = regimes
        return market, options, regimes

