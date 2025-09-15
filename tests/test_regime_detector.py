import numpy as np
import pandas as pd

from firm.regime.detector import RegimeDetector


def test_volatility_percentile_basic():
    # Construct simple series with rising volatility
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    close = np.cumsum(np.random.randn(100)) + 100
    returns = np.r_[0, np.diff(close) / close[:-1]]
    vol = pd.Series(returns).rolling(5).std().fillna(returns.std())
    df = pd.DataFrame({"Close": close, "Returns": returns, "Volatility": vol.values}, index=idx)

    det = RegimeDetector(method="volatility_percentile")
    regimes = det.detect(df, low_pct=0.2, high_pct=0.8)
    assert regimes.shape[0] == len(df)
    # Expect at least some high-vol labels
    assert np.any(regimes == 1)

