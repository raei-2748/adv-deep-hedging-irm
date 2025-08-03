"""
Performance metrics for deep hedging experiments
"""

import numpy as np
import pandas as pd


def calculate_cvar(returns, confidence=0.95):
    """Calculate Conditional Value at Risk"""
    if len(returns) < 2:
        return 0
    
    returns_array = np.array(returns)
    var = np.percentile(returns_array, (1 - confidence) * 100)
    cvar = np.mean(returns_array[returns_array <= var])
    return abs(cvar)  # Return absolute value for minimization


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
        'cvar_95': calculate_cvar(returns, confidence),
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