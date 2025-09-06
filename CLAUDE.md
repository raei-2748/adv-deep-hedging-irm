# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation & Setup
```bash
# Install dependencies (preferred)
pip install -e .[dev]

# Alternative installation
make install
```

### Testing
```bash
# Run quick sanity/smoke tests
pytest -k "sanity or smoke" -q

# Run all tests with coverage
python -m pytest tests/ -v
make test

# Test configuration coverage is in pyproject.toml with --cov=src
```

### Training & Experiments
```bash
# Basic training with default config
python -m src.deephedge.train

# Override specific parameters
python src/deephedge/train.py experiment.seed=123 training.gan.num_epochs=200

# Multi-run experiments for statistical testing
python src/deephedge/train.py -m experiment.seed=42,43,44,45,46

# Legacy experiment runner (fallback)
python experiment.py
make run
```

### Code Quality
```bash
# Format code
black src/ tests/ experiment.py
isort src/ tests/ experiment.py
make format

# Lint code
flake8 src/ tests/ experiment.py
pylint src/ tests/ experiment.py
make lint
```

### Cleanup
```bash
# Clean generated files and caches
make clean
```

## Project Architecture

### Core Components
- **GAN Models** (`src/deephedge/models/gan.py`): LSTM-based Generator and Discriminator for adversarial path generation
- **Actor-Critic Hedger** (`src/deephedge/models/actor_critic.py`): Shared LSTM layers with actor/critic heads for hedging decisions
- **Deep Hedging Environment** (`src/deephedge/envs/deep_hedging_env.py`): Trading environment with transaction costs and P&L calculation
- **Training Manager** (`src/deephedge/train.py`): Orchestrates GAN and hedger training with alternating optimization
- **Data Manager** (`src/deephedge/data/dataloader.py`): Market data fetching, synthetic generation, and option pricing

### Configuration System
Uses **Hydra** for advanced configuration management:
- Main config: `configs/base.yaml`
- Model variants: `configs/model/` directory
- Override any parameter via CLI: `training.gan.batch_size=128`
- Multi-run support: `-m experiment.seed=42,43,44`

### Key Modules Structure
```
src/deephedge/
├── models/          # Neural network architectures (GAN, Actor-Critic)
├── envs/           # Trading environment and simulation
├── data/           # Data loading, preprocessing, option pricing
├── utils/          # Metrics tracking, logging, seeding, registry
└── train.py        # Main training orchestration with Hydra CLI
```

### Experiment Protocol
The project implements "Protocol v1.0" with frozen experimental design:
- **Asset**: S&P 500 (SPY ETF proxy for ES futures)
- **Derivative**: European ATM Call Options
- **Risk Metric**: CVaR 95% on daily P&L
- **Training**: 60-step episodes (8-hour trading day, 8-minute bars)
- **Test Periods**: Predefined crash periods (COVID-19, Volmageddon, etc.)

### Deterministic Reproducibility
Comprehensive seed control across:
- Python random (`PYTHONHASHSEED`)
- NumPy arrays
- PyTorch CPU/GPU operations
- CUDA kernels (deterministic mode)
- Model initialization (Xavier uniform)

Configuration in `src/deephedge/config.py` and `configs/base.yaml` under `deterministic:` section.

### Logging & Experiment Tracking
Multi-layer logging system:
1. **Code Layer**: Python logging + loguru with git commit SHA, device info
2. **Metrics Layer**: Weights & Biases integration for training curves and evaluation
3. **Visuals Layer**: Matplotlib PNG artifacts (loss curves, CVaR comparisons)
4. **CLI Layer**: Hydra-powered configuration overrides and multi-run support

Artifacts saved to `runs/` directory with timestamped experiment logs.

## Development Notes

### Testing Strategy
- Sanity/smoke tests for quick validation: `pytest -k "sanity or smoke" -q`
- Full test suite with coverage reporting
- Pytest configuration in `pyproject.toml` with `testpaths = ["tests"]`

### Code Style
- **Black** formatting (line length 88)
- **isort** import sorting (black profile)
- **flake8** + **pylint** for linting
- All configuration in `pyproject.toml`

### Data Sources
- **Real data**: S&P 500 via yfinance (configurable date ranges)
- **Synthetic data**: Geometric Brownian Motion for demonstrations
- **Option pricing**: Complete Black-Scholes implementation with Greeks

### Research Context
This implements deep hedging research comparing:
- **H0**: GAN-trained vs GBM-trained hedgers show no CVaR difference
- **H1**: Adversarial training achieves 20% lower CVaR95 during market crashes
- Focus on realistic transaction costs and CVaR-based risk metrics