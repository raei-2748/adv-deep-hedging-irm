"""
Trading environment classes for deep hedging
"""

import numpy as np
import pandas as pd


class DeepHedgingEnvironment:
    """Trading environment for deep hedging"""
    
    def __init__(self, data, option_data, transaction_cost=0.0001):
        self.data = data
        self.option_data = option_data
        self.transaction_cost = transaction_cost
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.portfolio_value = 100000  # Initial capital
        self.hedge_position = 0
        self.cumulative_pnl = 0
        self.daily_pnl = []
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation"""
        if self.current_step >= len(self.data) - 1:
            return None
        
        # State features: [price, returns, volatility, option_price, delta, gamma, theta, hedge_position, portfolio_value, step]
        current_data = self.data.iloc[self.current_step]
        current_option = self.option_data.iloc[self.current_step]
        
        state = np.array([
            current_data['Close'] / 100,  # Normalized price
            current_data['Returns'] if not pd.isna(current_data['Returns']) else 0,
            current_option['volatility'],
            current_option['option_price'] / 100,  # Normalized option price
            current_option['delta'],
            current_option['gamma'] * 100,  # Scaled gamma
            current_option['theta'] / 100,  # Scaled theta
            self.hedge_position,
            self.portfolio_value / 100000,  # Normalized portfolio value
            self.current_step / 60  # Normalized step
        ])
        
        return state
    
    def step(self, action):
        """Execute one trading step"""
        if self.current_step >= len(self.data) - 1:
            return None, 0, True, {}
        
        # Current prices
        current_price = self.data.iloc[self.current_step]['Close']
        current_option = self.option_data.iloc[self.current_step]
        
        # Next prices
        next_price = self.data.iloc[self.current_step + 1]['Close']
        next_option = self.option_data.iloc[self.current_step + 1]
        
        # Calculate option P&L
        option_pnl = next_option['option_price'] - current_option['option_price']
        
        # Calculate hedge P&L
        hedge_pnl = -self.hedge_position * (next_price - current_price)
        
        # Calculate transaction costs
        new_position = action * 1000  # Scale action to position size
        position_change = new_position - self.hedge_position
        transaction_costs = abs(position_change) * self.transaction_cost
        
        # Update portfolio
        total_pnl = option_pnl + hedge_pnl - transaction_costs
        self.portfolio_value += total_pnl
        self.hedge_position = new_position
        self.cumulative_pnl += total_pnl
        
        # Store daily P&L
        self.daily_pnl.append(total_pnl)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1 or len(self.daily_pnl) >= 60
        
        # Calculate reward (negative CVaR)
        reward = -self._calculate_cvar(self.daily_pnl, confidence=0.95) if len(self.daily_pnl) > 0 else 0
        
        next_state = self._get_state()
        
        return next_state, reward, done, {
            'option_pnl': option_pnl,
            'hedge_pnl': hedge_pnl,
            'transaction_costs': transaction_costs,
            'total_pnl': total_pnl,
            'portfolio_value': self.portfolio_value
        }
    
    def _calculate_cvar(self, returns, confidence=0.95):
        """Calculate Conditional Value at Risk"""
        if len(returns) < 2:
            return 0
        
        returns_array = np.array(returns)
        var = np.percentile(returns_array, (1 - confidence) * 100)
        cvar = np.mean(returns_array[returns_array <= var])
        return abs(cvar)  # Return absolute value for minimization 