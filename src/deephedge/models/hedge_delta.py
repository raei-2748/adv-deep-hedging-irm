
import torch
import torch.nn as nn

class SimpleDeltaHedger:
    """Simple delta-hedging strategy for comparison"""

    def __init__(self, pricer):
        self.pricer = pricer

    def get_hedge_position(self, option_delta):
        """Return hedge position based on option delta"""
        return -option_delta
