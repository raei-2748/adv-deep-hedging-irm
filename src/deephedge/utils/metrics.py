"""
Performance metrics for deep hedging experiments
"""

import numpy as np
import pandas as pd
from scipy import stats


def calculate_cvar(values, confidence: float = 0.95, assume_losses_negative: bool = True):
    """Compute CVaR (Expected Shortfall) at given confidence for a sample.

    Args:
        values: Iterable of episode-level P&L (or returns). If assume_losses_negative=True,
                the left tail (worst outcomes) are the most negative values.
        confidence: Confidence level for VaR/CVaR (e.g., 0.95).
        assume_losses_negative: If True, treat smaller values as worse. If False, treat larger values as worse.

    Returns:
        cvar: The arithmetic mean of the worst (1-confidence) tail.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return 0.0
    # Sort ascending if losses are negative; descending otherwise
    if assume_losses_negative:
        sorted_arr = np.sort(arr)
    else:
        sorted_arr = np.sort(arr)[::-1]

    k = int(np.floor((1.0 - confidence) * n))
    k = max(1, k)  # ensure at least one element in the tail
    tail = sorted_arr[:k]
    return float(np.mean(tail))


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    if len(returns) < 2:
        return 0

    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0


def calculate_max_drawdown(returns):
    """Calculate maximum drawdown"""
    if len(returns) < 2:
        return 0

    cumulative = np.cumprod(1 + np.array(returns))
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)


def calculate_volatility(returns, annualize=True):
    """Calculate volatility"""
    if len(returns) < 2:
        return 0

    volatility = np.std(returns)
    if annualize:
        volatility *= np.sqrt(252)  # Annualize
    return volatility


def calculate_metrics(pnl_series, confidence=0.95):
    """Calculate comprehensive performance metrics"""
    if len(pnl_series) == 0:
        return {
            'cvar_95': 0,
            'mean_pnl': 0,
            'total_pnl': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'volatility': 0,
            'num_trades': 0
        }

    returns = np.diff(pnl_series) if len(pnl_series) > 1 else [0]

    return {
        # For metrics we report CVaR on episode-level P&L differences (loss negative), left-tail mean
        'cvar_95': abs(calculate_cvar(returns, confidence, assume_losses_negative=True)),
        'mean_pnl': np.mean(pnl_series),
        'total_pnl': np.sum(pnl_series),
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'max_drawdown': calculate_max_drawdown(returns),
        'volatility': calculate_volatility(returns),
        'num_trades': len(pnl_series)
    }


def compare_hedgers(ai_results, delta_results):
    """Compare AI hedger vs Delta hedger results"""
    comparison = {}

    for metric in ['cvar_95', 'mean_pnl', 'sharpe_ratio', 'max_drawdown']:
        if metric in ai_results and metric in delta_results:
            ai_value = ai_results[metric]
            delta_value = delta_results[metric]

            if delta_value != 0:
                improvement = ((delta_value - ai_value) / delta_value) * 100
            else:
                improvement = 0

            comparison[f"{metric}_improvement"] = improvement
            comparison[f"ai_{metric}"] = ai_value
            comparison[f"delta_{metric}"] = delta_value

    return comparison


def calculate_rolling_volatility(returns, window=20, annualization_factor=252):
    """Calculate rolling volatility"""
    return returns.rolling(window).std() * np.sqrt(annualization_factor)


def calculate_var(returns, confidence=0.95):
    """Calculate Value at Risk"""
    return np.percentile(returns, (1 - confidence) * 100)


def calculate_calmar_ratio(returns, risk_free_rate=0.02):
    """Calculate Calmar ratio"""
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
    max_dd = abs(calculate_max_drawdown(returns))
    return sharpe / max_dd if max_dd > 0 else 0


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