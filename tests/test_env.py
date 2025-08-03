"""
Unit tests for deep hedging environment
"""

import pytest
import numpy as np
import pandas as pd
from src.deephedge.envs.deep_hedging_env import DeepHedgingEnvironment


class TestDeepHedgingEnvironment:
    """Test cases for deep hedging environment"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create synthetic data for testing
        np.random.seed(42)
        n_steps = 100
        
        # Generate synthetic price data
        prices = 100 + np.cumsum(np.random.normal(0, 0.5, n_steps))
        dates = pd.date_range('2024-01-01', periods=n_steps, freq='8T')
        
        self.market_data = pd.DataFrame({
            'Open': prices,
            'High': prices * 1.001,
            'Low': prices * 0.999,
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, n_steps),
            'Returns': np.diff(prices, prepend=prices[0]) / prices,
            'Volatility': 0.15 + np.random.normal(0, 0.01, n_steps)
        }, index=dates)
        
        # Generate synthetic option data
        self.option_data = pd.DataFrame({
            'underlying_price': prices,
            'option_price': prices * 0.05 + np.random.normal(0, 0.1, n_steps),
            'delta': np.random.uniform(0.3, 0.7, n_steps),
            'gamma': np.random.uniform(0.01, 0.05, n_steps),
            'theta': -np.random.uniform(0.1, 0.5, n_steps),
            'volatility': 0.15 + np.random.normal(0, 0.01, n_steps),
            'strike': 100
        }, index=dates)
        
        self.env = DeepHedgingEnvironment(self.market_data, self.option_data)
    
    def test_reset(self):
        """Test environment reset"""
        state = self.env.reset()
        
        # Check initial state
        assert self.env.current_step == 0
        assert self.env.portfolio_value == 100000
        assert self.env.hedge_position == 0
        assert self.env.cumulative_pnl == 0
        assert len(self.env.daily_pnl) == 0
        
        # Check state shape and values
        assert state is not None
        assert len(state) == 10  # 10 state features
        assert isinstance(state, np.ndarray)
    
    def test_step(self):
        """Test environment step"""
        state = self.env.reset()
        action = 0.5  # Buy position
        
        next_state, reward, done, info = self.env.step(action)
        
        # Check step results
        assert self.env.current_step == 1
        assert self.env.hedge_position == action * 1000
        assert len(self.env.daily_pnl) == 1
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Check info keys
        expected_keys = ['option_pnl', 'hedge_pnl', 'transaction_costs', 'total_pnl', 'portfolio_value']
        for key in expected_keys:
            assert key in info
    
    def test_multiple_steps(self):
        """Test multiple environment steps"""
        state = self.env.reset()
        
        for i in range(10):
            action = np.random.uniform(-1, 1)
            next_state, reward, done, info = self.env.step(action)
            
            if done:
                break
            
            state = next_state
        
        # Check final state
        assert self.env.current_step > 0
        assert len(self.env.daily_pnl) > 0
    
    def test_episode_completion(self):
        """Test episode completion"""
        state = self.env.reset()
        
        step_count = 0
        while state is not None and step_count < 100:
            action = 0.0  # No action
            next_state, reward, done, info = self.env.step(action)
            step_count += 1
            
            if done:
                break
            
            state = next_state
        
        # Should complete episode
        assert step_count > 0
        assert len(self.env.daily_pnl) > 0
    
    def test_state_features(self):
        """Test state feature calculation"""
        state = self.env.reset()
        
        # Check state features
        assert len(state) == 10
        
        # State should contain normalized values
        assert 0 <= state[0] <= 2  # Normalized price
        assert -1 <= state[1] <= 1  # Returns
        assert state[2] > 0  # Volatility
        assert 0 <= state[3] <= 1  # Normalized option price
        assert -1 <= state[4] <= 1  # Delta
        assert state[5] > 0  # Scaled gamma
        assert state[6] < 0  # Scaled theta (negative)
        assert isinstance(state[7], (int, float))  # Hedge position
        assert 0 <= state[8] <= 2  # Normalized portfolio value
        assert 0 <= state[9] <= 1  # Normalized step
    
    def test_transaction_costs(self):
        """Test transaction cost calculation"""
        self.env.reset()
        
        # No position change
        action1 = 0.0
        _, _, _, info1 = self.env.step(action1)
        
        # Position change
        action2 = 0.5
        _, _, _, info2 = self.env.step(action2)
        
        # Should have higher transaction costs for position change
        assert info2['transaction_costs'] > info1['transaction_costs']
    
    def test_pnl_calculation(self):
        """Test P&L calculation"""
        self.env.reset()
        
        # Take a position
        action = 0.5
        _, _, _, info = self.env.step(action)
        
        # Check P&L components
        assert 'option_pnl' in info
        assert 'hedge_pnl' in info
        assert 'transaction_costs' in info
        assert 'total_pnl' in info
        
        # Total P&L should be option + hedge - transaction costs
        expected_total = info['option_pnl'] + info['hedge_pnl'] - info['transaction_costs']
        assert abs(info['total_pnl'] - expected_total) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__]) 