"""
Configuration file for Deep Hedging Experiment
"""

import torch

class ExperimentConfig:
    """Configuration class for the deep hedging experiment"""
    
    # Data Configuration
    START_DATE = '2015-01-01'
    END_DATE = '2025-01-01'
    DATA_SYMBOL = 'SPY'  # Proxy for ES futures
    DATA_INTERVAL = '1m'
    RESAMPLE_INTERVAL = '8T'  # 8-minute bars
    
    # Option Configuration
    OPTION_TYPE = 'call'
    TIME_TO_EXPIRY = 1/252  # 1 day
    RISK_FREE_RATE = 0.02
    
    # Training Configuration
    GAN_EPOCHS = 50
    HEDGER_EPOCHS = 500
    BATCH_SIZE = 32
    SEQUENCE_LENGTH = 60  # 8-hour trading day
    
    # Model Architecture
    HIDDEN_DIM = 64
    LSTM_LAYERS = 2
    STATE_DIM = 10
    DROPOUT_RATE = 0.2
    
    # Learning Rates
    GAN_LR = 0.0002
    HEDGER_LR = 0.001
    
    # Environment Configuration
    INITIAL_CAPITAL = 100000
    TRANSACTION_COST = 0.0001
    POSITION_SCALE = 1000
    
    # Risk Metrics
    CVAR_CONFIDENCE = 0.95
    VOLATILITY_WINDOW = 20
    
    # Crash Periods for Evaluation
    CRASH_PERIODS = [
        ('COVID-19 Crash', '2020-03-01', '2020-03-31'),
        ('Dec 2018 Sell-off', '2018-12-01', '2018-12-31')
    ]
    
    # Synthetic Data Parameters (if real data unavailable)
    SYNTHETIC_MU = 0.08  # Annual return
    SYNTHETIC_SIGMA = 0.15  # Annual volatility
    SYNTHETIC_INITIAL_PRICE = 100
    
    # Random Seeds and Deterministic Settings
    TORCH_SEED = 42
    NUMPY_SEED = 42
    PYTHON_SEED = 42
    DETERMINISTIC_MODE = True
    CUDA_DETERMINISTIC = True
    CUDA_BENCHMARK = False
    
    # Device Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Output Configuration
    SAVE_PLOTS = True
    PLOT_DPI = 300
    RESULTS_FILENAME = 'deep_hedging_results.png'
    
    @classmethod
    def get_gan_config(cls):
        """Get GAN-specific configuration"""
        return {
            'input_dim': 10,
            'hidden_dim': cls.HIDDEN_DIM,
            'sequence_length': cls.SEQUENCE_LENGTH,
            'dropout_rate': cls.DROPOUT_RATE
        }
    
    @classmethod
    def get_hedger_config(cls):
        """Get hedger-specific configuration"""
        return {
            'state_dim': cls.STATE_DIM,
            'hidden_dim': cls.HIDDEN_DIM,
            'sequence_length': cls.SEQUENCE_LENGTH,
            'dropout_rate': cls.DROPOUT_RATE
        }
    
    @classmethod
    def get_environment_config(cls):
        """Get environment-specific configuration"""
        return {
            'initial_capital': cls.INITIAL_CAPITAL,
            'transaction_cost': cls.TRANSACTION_COST,
            'position_scale': cls.POSITION_SCALE
        }
    
    @classmethod
    def get_training_config(cls):
        """Get training-specific configuration"""
        return {
            'gan_epochs': cls.GAN_EPOCHS,
            'hedger_episodes': cls.HEDGER_EPOCHS,
            'batch_size': cls.BATCH_SIZE,
            'gan_lr': cls.GAN_LR,
            'hedger_lr': cls.HEDGER_LR
        }
    
    @classmethod
    def get_seed_config(cls):
        """Get seed and deterministic configuration"""
        return {
            'torch_seed': cls.TORCH_SEED,
            'numpy_seed': cls.NUMPY_SEED,
            'python_seed': cls.PYTHON_SEED,
            'deterministic': cls.DETERMINISTIC_MODE,
            'cuda_deterministic': cls.CUDA_DETERMINISTIC,
            'cuda_benchmark': cls.CUDA_BENCHMARK
        } 