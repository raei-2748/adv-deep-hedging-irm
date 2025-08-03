"""
Utility functions for Deep Hedging Experiment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def calculate_rolling_volatility(returns, window=20, annualization_factor=252):
    """Calculate rolling volatility"""
    return returns.rolling(window).std() * np.sqrt(annualization_factor)

def calculate_var(returns, confidence=0.95):
    """Calculate Value at Risk"""
    return np.percentile(returns, (1 - confidence) * 100)

def calculate_cvar(returns, confidence=0.95):
    """Calculate Conditional Value at Risk (Expected Shortfall)"""
    var = calculate_var(returns, confidence)
    return np.mean(returns[returns <= var])

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate/252
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calculate_calmar_ratio(returns, risk_free_rate=0.02):
    """Calculate Calmar ratio"""
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
    max_dd = abs(calculate_max_drawdown(returns))
    return sharpe / max_dd if max_dd > 0 else 0

def plot_training_curves(losses, title="Training Loss", save_path=None):
    """Plot training loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_price_paths(real_paths, generated_paths, title="Price Paths Comparison", save_path=None):
    """Plot real vs generated price paths"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Real paths
    for i in range(min(5, len(real_paths))):
        ax1.plot(real_paths[i], alpha=0.7, label=f'Real {i+1}' if i == 0 else "")
    ax1.set_title('Real Price Paths')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Generated paths
    for i in range(min(5, len(generated_paths))):
        ax2.plot(generated_paths[i], alpha=0.7, label=f'Generated {i+1}' if i == 0 else "")
    ax2.set_title('Generated Price Paths')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Price')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_risk_metrics_comparison(results_dict, save_path=None):
    """Plot comparison of risk metrics across different methods"""
    metrics = ['cvar_95', 'mean_pnl', 'sharpe_ratio', 'max_drawdown']
    method_names = list(results_dict.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        values = [results_dict[method][metric] for method in method_names]
        axes[i].bar(method_names, values)
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def statistical_significance_test(group1, group2, test_type='t_test'):
    """Perform statistical significance test between two groups"""
    if test_type == 't_test':
        statistic, p_value = stats.ttest_ind(group1, group2)
        test_name = "Independent t-test"
    elif test_type == 'mann_whitney':
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        test_name = "Mann-Whitney U test"
    elif test_type == 'ks_test':
        statistic, p_value = stats.ks_2samp(group1, group2)
        test_name = "Kolmogorov-Smirnov test"
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    return {
        'test_name': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def calculate_hedge_effectiveness(option_pnl, hedge_pnl):
    """Calculate hedge effectiveness (R-squared of regression)"""
    # Hedge effectiveness = 1 - (variance of hedged position / variance of unhedged position)
    unhedged_variance = np.var(option_pnl)
    hedged_variance = np.var(option_pnl + hedge_pnl)
    
    if unhedged_variance == 0:
        return 0
    
    effectiveness = 1 - (hedged_variance / unhedged_variance)
    return max(0, min(1, effectiveness))  # Clamp between 0 and 1

def calculate_tracking_error(option_pnl, hedge_pnl):
    """Calculate tracking error between option and hedge P&L"""
    return np.std(option_pnl + hedge_pnl)

def generate_synthetic_crash_data(n_periods=1000, crash_intensity=0.3, crash_duration=50):
    """Generate synthetic crash data for testing"""
    np.random.seed(42)
    
    # Normal market conditions
    normal_returns = np.random.normal(0.0001, 0.015, n_periods)
    
    # Add crash in the middle
    crash_start = n_periods // 2
    crash_end = crash_start + crash_duration
    
    # Crash period with higher volatility and negative returns
    crash_returns = np.random.normal(-0.002, 0.03, crash_duration)
    
    # Combine normal and crash periods
    returns = normal_returns.copy()
    returns[crash_start:crash_end] = crash_returns
    
    # Generate prices
    prices = 100 * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({
        'Close': prices,
        'Returns': returns,
        'Volatility': calculate_rolling_volatility(pd.Series(returns))
    })

def calculate_performance_metrics(pnl_series):
    """Calculate comprehensive performance metrics"""
    returns = np.diff(pnl_series) / pnl_series[:-1] if len(pnl_series) > 1 else [0]
    
    metrics = {
        'total_return': (pnl_series[-1] - pnl_series[0]) / pnl_series[0] if pnl_series[0] != 0 else 0,
        'volatility': np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0,
        'sharpe_ratio': calculate_sharpe_ratio(np.array(returns)),
        'max_drawdown': calculate_max_drawdown(np.array(returns)),
        'var_95': calculate_var(returns, 0.95),
        'cvar_95': calculate_cvar(returns, 0.95),
        'calmar_ratio': calculate_calmar_ratio(np.array(returns)),
        'skewness': stats.skew(returns) if len(returns) > 2 else 0,
        'kurtosis': stats.kurtosis(returns) if len(returns) > 2 else 0
    }
    
    return metrics

def print_performance_summary(metrics_dict, title="Performance Summary"):
    """Print formatted performance summary"""
    print(f"\n{title}")
    print("=" * 50)
    
    for metric, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"{metric.replace('_', ' ').title()}: {value}")
    
    print("=" * 50)

def save_results_to_csv(results_dict, filename="hedging_results.csv"):
    """Save results to CSV file"""
    df = pd.DataFrame(results_dict).T
    df.to_csv(filename)
    print(f"Results saved to {filename}")

def load_results_from_csv(filename="hedging_results.csv"):
    """Load results from CSV file"""
    return pd.read_csv(filename, index_col=0)

def create_experiment_report(results_dict, config, filename="experiment_report.txt"):
    """Create a comprehensive experiment report"""
    with open(filename, 'w') as f:
        f.write("DEEP HEDGING EXPERIMENT REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("EXPERIMENTAL DESIGN\n")
        f.write("-" * 20 + "\n")
        f.write(f"Asset: {config.DATA_SYMBOL}\n")
        f.write(f"Derivative: European {config.OPTION_TYPE.upper()} Option\n")
        f.write(f"Risk Metric: CVaR {config.CVAR_CONFIDENCE*100}%\n")
        f.write(f"Training Episodes: {config.HEDGER_EPOCHS}\n")
        f.write(f"Sequence Length: {config.SEQUENCE_LENGTH}\n\n")
        
        f.write("HYPOTHESIS TESTING\n")
        f.write("-" * 20 + "\n")
        f.write("H0: No difference in CVaR between GAN-trained and GBM-trained hedgers\n")
        f.write("H1: Adversarial-trained hedger achieves 20% lower CVaR95\n\n")
        
        f.write("RESULTS\n")
        f.write("-" * 20 + "\n")
        
        for method, metrics in results_dict.items():
            f.write(f"\n{method}:\n")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"  {metric}: {value:.4f}\n")
                else:
                    f.write(f"  {metric}: {value}\n")
        
        f.write("\nCONCLUSION\n")
        f.write("-" * 20 + "\n")
        f.write("This report contains the experimental results for the deep hedging study.\n")
        f.write("For detailed analysis, refer to the generated plots and statistical tests.\n")
    
    print(f"Experiment report saved to {filename}") 