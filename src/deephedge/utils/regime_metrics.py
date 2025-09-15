from __future__ import annotations

import numpy as np
import pandas as pd


def per_regime_mean(values: np.ndarray, regimes: np.ndarray) -> dict:
    out = {}
    for r in np.unique(regimes):
        m = float(np.mean(values[regimes == r])) if np.any(regimes == r) else float("nan")
        out[int(r)] = m
    return out


def per_regime_summary(market_df: pd.DataFrame, metric_col: str = "Returns") -> dict:
    regimes = market_df.get("Regime")
    if regimes is None:
        raise ValueError("market_df must contain 'Regime' column")
    vals = market_df[metric_col].values
    return per_regime_mean(vals, regimes.values)

