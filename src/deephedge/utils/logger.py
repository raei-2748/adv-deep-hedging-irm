"""
Enhanced logging utilities for deep hedging experiments
"""

import logging
import os
import subprocess
import hashlib
import json
from datetime import datetime
from pathlib import Path
from loguru import logger
import torch
from omegaconf import OmegaConf


def get_git_commit_sha():
    """Get current Git commit SHA"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], 
            capture_output=True, 
            text=True, 
            cwd=os.getcwd()
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_docker_image_hash():
    """Get Docker image hash if running in container"""
    try:
        with open('/proc/self/cgroup', 'r') as f:
            for line in f:
                if 'docker' in line:
                    return line.strip().split('/')[-1][:12]
    except Exception:
        pass
    return "not-docker"


def calculate_config_hash(config_dict):
    """Calculate hash of configuration for reproducibility"""
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def setup_logger(log_dir="runs", experiment_name=None):
    """
    Setup comprehensive logging with loguru
    
    Args:
        log_dir: Directory to store logs
        experiment_name: Name of the experiment
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment-specific log file
    if experiment_name:
        log_file = log_path / f"{experiment_name}_{timestamp}.log"
    else:
        log_file = log_path / f"experiment_{timestamp}.log"
    
    # Remove default loguru handler
    logger.remove()
    
    # Add console handler with colors
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file handler
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="100 MB",
        retention="30 days"
    )
    
    return logger


def log_experiment_start(config, device_info=True):
    """
    Log comprehensive experiment start information
    
    Args:
        config: Experiment configuration
        device_info: Whether to log device information
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT START")
    logger.info("=" * 60)
    
    # Log basic experiment info
    logger.info(f"Experiment Name: {config.experiment.name}")
    logger.info(f"Seed: {config.experiment.seed}")
    logger.info(f"Device: {config.experiment.device}")
    
    # Log Git information
    git_sha = get_git_commit_sha()
    docker_hash = get_docker_image_hash()
    logger.info(f"Git Commit SHA: {git_sha}")
    logger.info(f"Docker Image Hash: {docker_hash}")
    
    # Log configuration hash
    try:
        config_dict = OmegaConf.to_container(config)
        config_hash = calculate_config_hash(config_dict)
        logger.info(f"Config Hash: {config_hash}")
    except Exception as e:
        logger.warning(f"Could not calculate config hash: {e}")
        config_hash = "unknown"
    
    # Log device information
    if device_info:
        if torch.cuda.is_available():
            logger.info(f"CUDA Device: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.info("CUDA not available, using CPU")
    
    # Log deterministic settings
    logger.info(f"Deterministic Mode: {config.deterministic.deterministic_mode}")
    logger.info(f"CUDA Deterministic: {config.deterministic.cuda_deterministic}")
    
    logger.info("=" * 60)


def log_training_metrics(epoch, metrics, prefix=""):
    """
    Log training metrics in a structured way
    
    Args:
        epoch: Current epoch/step
        metrics: Dictionary of metrics
        prefix: Prefix for metric names
    """
    metric_str = " | ".join([f"{prefix}{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"Epoch {epoch}: {metric_str}")


def log_evaluation_results(results, crash_periods):
    """
    Log evaluation results with CVaR and P&L metrics
    
    Args:
        results: Dictionary of evaluation results
        crash_periods: List of crash period names
    """
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    
    for period in crash_periods:
        ai_key = f"{period}_AI"
        delta_key = f"{period}_Delta"
        
        if ai_key in results and delta_key in results:
            ai_data = results[ai_key]
            delta_data = results[delta_key]
            
            logger.info(f"\n{period}:")
            logger.info(f"  AI Hedger - CVaR 95%: {ai_data['cvar_95']:.4f}, Mean P&L: {ai_data['mean_pnl']:.4f}")
            logger.info(f"  Delta Hedger - CVaR 95%: {delta_data['cvar_95']:.4f}, Mean P&L: {delta_data['mean_pnl']:.4f}")
            
            # Calculate improvement
            cvar_improvement = ((delta_data['cvar_95'] - ai_data['cvar_95']) / delta_data['cvar_95']) * 100
            logger.info(f"  CVaR Improvement: {cvar_improvement:.2f}%")


def log_experiment_end(duration, final_metrics=None):
    """
    Log experiment completion
    
    Args:
        duration: Experiment duration in seconds
        final_metrics: Final experiment metrics
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration:.2f} seconds")
    
    if final_metrics:
        logger.info("Final Metrics:")
        for key, value in final_metrics.items():
            logger.info(f"  {key}: {value}")
    
    logger.info("=" * 60)


# Legacy function for backward compatibility
def setup_basic_logger():
    """Basic logger setup for backward compatibility"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('deephedge') 