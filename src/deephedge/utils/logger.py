"""
Logging utilities for deep hedging experiments
"""

import logging
import sys
from datetime import datetime


def setup_logger(name='deephedge', level=logging.INFO):
    """Setup logger for deep hedging experiments"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger


def get_experiment_logger(experiment_name, log_dir='runs'):
    """Get logger for specific experiment"""
    import os
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create experiment-specific logger
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger_name = f"{experiment_name}_{timestamp}"
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # File handler
    log_file = os.path.join(log_dir, f"{logger_name}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger 