# Deep Hedging Experiment: GAN vs Geometric Brownian Motion

## Research Question

Does training a deep hedger on GAN-generated worst-case price paths reduce the 95% CVaR of its P&L during real-world market crashes compared with a hedger trained on standard geometric-Brownian paths?

## Experimental Design

### Asset and Derivative
- **Asset**: S&P 500 E-mini futures (ES) - represented by SPY ETF
- **Derivative**: European ATM Call Option
- **Risk Metric**: CVaR 95% on daily P&L
- **Training Horizon**: 60-step episode (8-hour trading day with 8-minute bars)

### Architecture
- **GAN**: Generator and Discriminator with LSTM layers for worst-case path generation
- **Hedger**: Actor-Critic with two hidden LSTM layers
- **Environment**: Custom trading environment with transaction costs

### Hypothesis Testing
- **H0**: No difference in CVaR between GAN-trained and GBM-trained hedgers
- **H1**: Adversarial-trained hedger achieves 20% lower CVaR95

## Key Components

### 1. BlackScholesPricer
- Complete Black-Scholes implementation
- Greeks calculation (Delta, Gamma, Theta, Vega)
- Implied volatility calculation

### 2. DataManager
- Market data fetching via yfinance
- Synthetic data generation for demonstration
- Option pricing and Greeks calculation

### 3. GAN Architecture
- **Generator**: LSTM-based network generating price paths
- **Discriminator**: LSTM-based network distinguishing real vs generated paths
- Adversarial training to create worst-case scenarios

### 4. ActorCriticHedger
- Shared LSTM layers for sequence processing
- Actor network: Policy for hedge position sizing
- Critic network: Value function estimation
- Tanh activation for bounded position sizes

### 5. DeepHedgingEnvironment
- Realistic trading environment
- Transaction costs modeling
- P&L calculation including option and hedge components
- CVaR-based reward function

### 6. TrainingManager
- GAN training with adversarial loss
- Actor-Critic training with advantage estimation
- Alternating training between generator and hedger

## Installation and Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the experiment:
```bash
python experiment.py
```

## 🚀 **Hydra Configuration Management**

This project uses Hydra for advanced configuration management and experiment tracking.

### **Basic Usage:**

```bash
# Run with default configuration
python src/deephedge/train.py

# Override parameters from command line
python src/deephedge/train.py experiment.seed=123 training.gan.num_epochs=200

# Run with different configuration
python src/deephedge/train.py experiment.name=quick_test training.hedger.num_episodes=100
```

### **Configuration Overrides:**

```bash
# Change model architecture
python src/deephedge/train.py model.gan.hidden_dim=256 model.hedger.hidden_dim=256

# Adjust training parameters
python src/deephedge/train.py training.gan.batch_size=128 training.gan.lr_generator=0.0001

# Modify environment settings
python src/deephedge/train.py environment.transaction_cost=0.0005 environment.initial_capital=200000

# Change data settings
python src/deephedge/train.py data.start_date=2020-01-01 data.end_date=2023-12-31
```

### **Multi-Run Experiments:**

```bash
# Run with multiple seeds for statistical testing
python src/deephedge/train.py -m experiment.seed=42,43,44,45,46

# Sweep over learning rates
python src/deephedge/train.py -m training.gan.lr_generator=0.0001,0.0002,0.0005

# Grid search over model parameters
python src/deephedge/train.py -m model.gan.hidden_dim=64,128,256 model.hedger.hidden_dim=64,128,256
```

### **Configuration Structure:**

The configuration is organized in `configs/base.yaml`:

- **experiment**: Experiment metadata and settings
- **data**: Data loading and preprocessing parameters
- **model**: Model architecture configurations
- **training**: Training hyperparameters
- **environment**: Trading environment settings
- **risk**: Risk metrics and thresholds
- **logging**: Logging and output settings
- **deterministic**: Reproducibility settings

## 🔒 Deterministic Reproducibility

This experiment uses comprehensive deterministic seed control to ensure reproducible results:

### **Seed Control Features:**
- **Python random**: Controlled via `PYTHONHASHSEED`
- **NumPy random**: Deterministic array generation
- **PyTorch random**: CPU and GPU tensor generation
- **CUDA operations**: Deterministic GPU kernels
- **Model initialization**: Xavier uniform initialization

### **Verification:**
The experiment automatically verifies deterministic behavior by running multiple tests and comparing results.

### **Configuration:**
All seed settings are configurable in `src/deephedge/config.py`:
```python
TORCH_SEED = 42
NUMPY_SEED = 42
PYTHON_SEED = 42
DETERMINISTIC_MODE = True
CUDA_DETERMINISTIC = True
CUDA_BENCHMARK = False
```

## 📊 **Phase 0.5: Logging, Metrics & Experiment Tracking**

This project implements comprehensive experiment tracking across multiple layers:

### **Layer 1: Code Layer (Python logging + loguru)**
- **Tooling**: Python `logging` + `loguru` for enhanced logging
- **Key Items Logged**: 
  - Git commit SHA for reproducibility
  - Docker image hash (if applicable)
  - Configuration hash for deterministic runs
  - Device information (CUDA/CPU)
  - Deterministic settings

### **Layer 2: Metrics Layer (Weights & Biases)**
- **Tooling**: Weights & Biases for experiment tracking
- **Key Items Logged**:
  - GAN loss curves (D_loss, G_loss)
  - Hedger training metrics (CVaR95, turnover)
  - Evaluation results with CVaR comparisons
  - Real-time metric tracking

### **Layer 3: Visuals Layer (Matplotlib + PNG artifacts)**
- **Tooling**: Matplotlib with PNG output, uploaded as artifacts
- **Key Items Logged**:
  - Training loss curves
  - CVaR comparison bar charts
  - P&L traces for each crash period
  - Cumulative performance plots

### **Layer 4: CLI Layer (train.py with argparse)**
- **Tooling**: Hydra-powered CLI with quick overrides
- **Key Items Logged**:
  - Configuration overrides via command line
  - Multi-run experiment support
  - Automatic experiment naming and tagging

### **Usage Examples:**

```bash
# Basic run with full logging
python -m src.deephedge.train

# Run with W&B tracking (requires wandb login)
python -m src.deephedge.train logging.wandb_project=my_project

# Override parameters with logging
python -m src.deephedge.train training.gan.num_epochs=200 experiment.seed=123

# Multi-run experiment for statistical testing
python -m src.deephedge.train -m experiment.seed=42,43,44,45,46
```

### **Generated Artifacts:**
- **Logs**: Timestamped log files in `runs/` directory
- **Plots**: Training curves and CVaR comparisons as PNG files
- **Configs**: Saved experiment configurations
- **W&B**: Real-time experiment tracking and visualization

## Experiment Flow

1. **Data Loading**: Fetch S&P 500 data or generate synthetic data
2. **Option Pricing**: Calculate option prices and Greeks for each timestep
3. **GAN Training**: Train generator to create worst-case price paths
4. **Hedger Training**: Train Actor-Critic hedger on GAN-generated paths
5. **Evaluation**: Test on historical crash periods (COVID-19, Dec 2018)
6. **Results**: Calculate CVaR, mean P&L, and generate visualizations

## Output

The experiment produces:
- Console output with training progress and results
- `deep_hedging_results.png` with four plots:
  - CVaR 95% comparison across crash periods
  - Mean P&L comparison
  - P&L time series
  - Cumulative P&L

## Key Features

### Risk Management
- CVaR 95% as primary risk metric
- Transaction cost modeling
- Realistic market dynamics

### Machine Learning
- LSTM layers for sequential data processing
- Actor-Critic architecture for policy optimization
- GAN for adversarial training

### Financial Modeling
- Black-Scholes option pricing
- Greeks calculation
- Market data integration

## Extensions and Improvements

1. **Enhanced GAN**: Implement TimeGAN or diffusion models
2. **Multiple Assets**: Extend to portfolio hedging
3. **Advanced RL**: Implement PPO or SAC algorithms
4. **Statistical Testing**: Add proper hypothesis testing with multiple runs
5. **Real Market Data**: Integrate with professional data feeds
6. **Risk Metrics**: Add VaR, Expected Shortfall, and other metrics

## Notes

- This is a research implementation for demonstration purposes
- For production use, additional validation and testing is required
- The synthetic data generation ensures the experiment runs without external dependencies
- Real market data integration requires appropriate API keys and data subscriptions

## Citation

If you use this code in your research, please cite:

```
Deep Hedging Experiment: GAN-Generated Worst-Case Paths vs Geometric Brownian Motion
Research on adversarial training for risk management in financial derivatives
``` 