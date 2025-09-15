from __future__ import annotations

import numpy as np
import pandas as pd


def invariant_feature_score(features: pd.DataFrame, regimes: np.ndarray) -> pd.Series:
    """
    Compute a simple invariance score per feature as 1 - variance of
    per-regime means (normalized). Higher is more invariant.
    """
    scores = {}
    for col in features.columns:
        vals = features[col].values
        means = []
        for r in np.unique(regimes):
            mask = regimes == r
            if mask.sum() == 0:
                continue
            means.append(vals[mask].mean())
        if not means:
            scores[col] = 0.0
            continue
        means = np.array(means)
        var = float(np.var(means))
        denom = float(np.var(vals) + 1e-8)
        score = 1.0 - min(1.0, var / (denom + 1e-8))
        scores[col] = score
    return pd.Series(scores).sort_values(ascending=False)

