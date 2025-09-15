from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_regimes(market_df: pd.DataFrame, regimes: np.ndarray, ax=None, title: str = "Market Regimes"):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(market_df.index, market_df["Close"], label="Close", lw=1.0)
    # Shade high-vol regime
    mask_high = regimes == 1
    ax.fill_between(
        market_df.index,
        market_df["Close"].min(),
        market_df["Close"].max(),
        where=mask_high,
        color="red",
        alpha=0.1,
        label="High-vol regime",
    )
    ax.set_title(title)
    ax.legend(loc="best")
    return ax

