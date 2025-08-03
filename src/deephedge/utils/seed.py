"""
Random seed utilities for reproducibility
"""

import torch
import numpy as np
import random


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_seed():
    """Get current random state for debugging"""
    return {
        'torch_seed': torch.initial_seed(),
        'numpy_seed': np.random.get_state(),
        'python_seed': random.getstate()
    } 