from __future__ import annotations

import numpy as np
import pandas as pd


class RegimeDetector:
    """
    Simple market regime detector.

    Methods:
      - volatility_percentile: classify by realized volatility percentiles
      - vix_threshold: classify by provided VIX series thresholds

    Returns an integer numpy array of regime labels {0,1} for low/high.
    """

    def __init__(self, method: str = "volatility_percentile"):
        self.method = method

    def detect(
        self,
        market_df: pd.DataFrame,
        vix: pd.Series | None = None,
        low_pct: float = 0.3,
        high_pct: float = 0.7,
        vix_low: float | None = None,
        vix_high: float | None = None,
    ) -> np.ndarray:
        if self.method == "vix_threshold":
            if vix is None:
                raise ValueError("vix series required for vix_threshold method")
            if vix_low is None or vix_high is None:
                # default to percentiles from the provided VIX
                vix_low = float(np.nanpercentile(vix.values, 100 * low_pct))
                vix_high = float(np.nanpercentile(vix.values, 100 * high_pct))
            vals = vix.reindex(market_df.index).fillna(method="ffill").values
            regimes = np.where(vals < vix_low, 0, np.where(vals > vix_high, 1, 0))
            return regimes.astype(int)

        # Default: realized volatility percentiles from market_df
        vol = market_df.get("Volatility")
        if vol is None:
            # compute from returns if not present
            ret = market_df.get("Returns")
            if ret is None:
                raise ValueError("market_df must have Volatility or Returns column")
            vol = ret.rolling(20).std() * np.sqrt(252)
        low_thr = float(np.nanpercentile(vol.values, 100 * low_pct))
        high_thr = float(np.nanpercentile(vol.values, 100 * high_pct))
        regimes = np.where(vol.values < low_thr, 0, np.where(vol.values > high_thr, 1, 0))
        return regimes.astype(int)

