"""
Random seed utilities for reproducibility
"""

import os
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): Seed value (default: 42)
        deterministic (bool): Whether to enforce deterministic behavior in CUDA ops
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False


def get_seed():
    """
    Get current random state for debugging.
    Warning: Python & NumPy states are complex objects.
    """
    return {
        "torch_seed": torch.initial_seed(),
        "numpy_seed": np.random.get_state(),
        "python_seed": random.getstate(),
    }


def verify_determinism(num_tests=5, tolerance=1e-6):
    """
    Verify deterministic behavior by running multiple tests.
    
    Args:
        num_tests (int): Number of tests to run
        tolerance (float): Tolerance for floating point comparisons
    
    Returns:
        bool: True if deterministic, False otherwise
    """
    print("🔍 Verifying deterministic behavior...")
    
    # Test 1: PyTorch tensor generation
    torch_results = []
    for i in range(num_tests):
        set_seed(42, deterministic=True)
        tensor = torch.randn(100, 100)
        torch_results.append(tensor.sum().item())
    
    # Test 2: NumPy array generation
    np_results = []
    for i in range(num_tests):
        set_seed(42, deterministic=True)
        array = np.random.randn(100, 100)
        np_results.append(array.sum())
    
    # Test 3: Python random
    py_results = []
    for i in range(num_tests):
        set_seed(42, deterministic=True)
        numbers = [random.random() for _ in range(100)]
        py_results.append(sum(numbers))
    
    # Check if all results are identical
    torch_consistent = all(abs(torch_results[0] - x) < tolerance for x in torch_results)
    np_consistent = all(abs(np_results[0] - x) < tolerance for x in np_results)
    py_consistent = all(abs(py_results[0] - x) < tolerance for x in py_results)
    
    print(f"🎲 PyTorch deterministic: {torch_consistent}")
    print(f"🎲 NumPy deterministic: {np_consistent}")
    print(f"🎲 Python random deterministic: {py_consistent}")
    
    is_deterministic = torch_consistent and np_consistent and py_consistent
    print(f"🔒 Overall deterministic: {is_deterministic}")
    
    return is_deterministic