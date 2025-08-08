
import numpy as np

def get_rolling_volatility(prices, window=30):
    """Calculate rolling volatility"""
    returns = np.log(prices / np.roll(prices, 1))
    rolling_vol = np.std(returns[-window:])
    return rolling_vol
