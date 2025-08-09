import numpy as np
from src.deephedge.utils.metrics import calculate_cvar


def test_cvar_definition_basic():
    data = np.array([-10, -9, -8, -1, 0, 1, 2, 3, 4, 5])
    cvar95 = calculate_cvar(data, confidence=0.95, assume_losses_negative=True)
    # 5% tail of 10 samples -> k=floor(0.5)=0 => enforce at least 1, pick worst element (-10)
    assert cvar95 == -10


def test_cvar_mixture_left_tail():
    rng = np.random.default_rng(123)
    n = 50000
    mix = np.where(rng.uniform(size=n) < 0.1, rng.normal(-5, 1, n), rng.normal(0, 1, n))
    cvar95 = calculate_cvar(mix, confidence=0.95, assume_losses_negative=True)
    assert cvar95 < -1.0


