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
        self.hedge_position = 0
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation"""
        if self.current_step >= len(self.data) - 1:
            return None
        
        # State features: [price, returns, volatility, option_price, delta, gamma, theta, hedge_position, step]
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
        transaction_costs = abs(position_change) * (self.tc_bps / 10000)
        
        # Update portfolio
        total_pnl = option_pnl + hedge_pnl - transaction_costs
        self.hedge_position = new_position
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1 or self.current_step >= 60 # Assuming 60 steps per episode
        
        # Calculate reward (negative total P&L for immediate feedback)
        reward = -total_pnl
        
        next_state = self._get_state()
        
        return next_state, reward, done, {
            'option_pnl': option_pnl,
            'hedge_pnl': hedge_pnl,
            'transaction_costs': transaction_costs,
            'total_pnl': total_pnl
        }
    
    # Remove _calculate_cvar as it's no longer used for immediate reward 