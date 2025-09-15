

class SimpleDeltaHedger:
    """Simple delta-hedging strategy for comparison"""

    def __init__(self, pricer=None):
        self.pricer = pricer

    def get_hedge_position(self, option_delta: float) -> float:
        """Return hedge position based on option delta"""
        return -float(option_delta)
