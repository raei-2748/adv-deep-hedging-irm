"""
End-to-end tests for deep hedging pipeline
"""

import pytest
import numpy as np
import pandas as pd
import torch
from src.deephedge.data.dataloader import DataManager
from src.deephedge.models.gan import GANGenerator, GANDiscriminator
from src.deephedge.models.actor_critic import ActorCriticHedger
from src.deephedge.models.hedge_delta import SimpleDeltaHedger
from src.deephedge.envs.deep_hedging_env import DeepHedgingEnvironment
from src.deephedge.utils.metrics import calculate_metrics, compare_hedgers


class TestEndToEnd:
    """End-to-end test cases for deep hedging pipeline"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Initialize components
        self.data_manager = DataManager()
        self.gan_generator = GANGenerator()
        self.gan_discriminator = GANDiscriminator()
        self.hedger = ActorCriticHedger()
        self.simple_hedger = SimpleDeltaHedger()
    
    def test_data_loading(self):
        """Test data loading and option pricing"""
        # Load data
        market_data = self.data_manager.fetch_sp500_data()
        option_data = self.data_manager.calculate_option_prices(market_data)
        
        # Check data integrity
        assert len(market_data) > 0
        assert len(option_data) > 0
        assert len(market_data) == len(option_data)
        
        # Check required columns
        required_market_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'Volatility']
        required_option_cols = ['underlying_price', 'option_price', 'delta', 'gamma', 'theta', 'volatility']
        
        for col in required_market_cols:
            assert col in market_data.columns
        
        for col in required_option_cols:
            assert col in option_data.columns
    
    def test_gan_models(self):
        """Test GAN model forward pass"""
        # Test generator
        batch_size = 4
        noise = torch.randn(batch_size, 60, 10)
        fake_paths = self.gan_generator(noise)
        
        assert fake_paths.shape == (batch_size, 60, 1)
        assert not torch.isnan(fake_paths).any()
        
        # Test discriminator
        validity = self.gan_discriminator(fake_paths)
        
        assert validity.shape == (batch_size, 1)
        assert not torch.isnan(validity).any()
        assert torch.all((validity >= 0) & (validity <= 1))
    
    def test_hedger_models(self):
        """Test hedger model forward pass"""
        # Test Actor-Critic hedger
        batch_size = 4
        state_sequence = torch.randn(batch_size, 60, 10)
        action, value = self.hedger(state_sequence)
        
        assert action.shape == (batch_size, 1)
        assert value.shape == (batch_size, 1)
        assert not torch.isnan(action).any()
        assert not torch.isnan(value).any()
        assert torch.all((action >= -1) & (action <= 1))  # Tanh output
    
    def test_environment(self):
        """Test environment functionality"""
        # Create synthetic data
        market_data = self.data_manager.generate_fully_synthetic_data()
        option_data = self.data_manager.calculate_option_prices(market_data)
        
        # Create environment
        env = DeepHedgingEnvironment(market_data, option_data)
        
        # Test episode
        state = env.reset()
        episode_rewards = []
        
        for _ in range(10):  # Short episode for testing
            if state is None:
                break
            
            # Random action
            action = np.random.uniform(-1, 1)
            next_state, reward, done, info = env.step(action)
            
            episode_rewards.append(reward)
            
            if done:
                break
            
            state = next_state
        
        # Check episode results
        assert len(episode_rewards) > 0
        assert env.current_step > 0
        assert len(env.daily_pnl) > 0
    
    def test_hedger_comparison(self):
        """Test hedger comparison functionality"""
        # Create synthetic P&L data
        ai_pnl = np.random.normal(0, 1, 100)
        delta_pnl = np.random.normal(0, 1.2, 100)
        
        # Calculate metrics
        ai_metrics = calculate_metrics(ai_pnl)
        delta_metrics = calculate_metrics(delta_pnl)
        
        # Compare hedgers
        comparison = compare_hedgers(ai_metrics, delta_metrics)
        
        # Check comparison results
        assert 'cvar_95_improvement' in comparison
        assert 'mean_pnl_improvement' in comparison
        assert 'sharpe_ratio_improvement' in comparison
        assert 'max_drawdown_improvement' in comparison
    
    def test_complete_pipeline(self):
        """Test complete deep hedging pipeline"""
        # 1. Load data
        market_data = self.data_manager.generate_fully_synthetic_data()
        option_data = self.data_manager.calculate_option_prices(market_data)
        
        # 2. Create environment
        env = DeepHedgingEnvironment(market_data, option_data)
        
        # 3. Test AI hedger
        ai_pnl = []
        state = env.reset()
        
        for _ in range(20):  # Short episode
            if state is None:
                break
            
            # Get AI action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
            action, _ = self.hedger(state_tensor)
            action = action.item()
            
            # Take step
            next_state, reward, done, info = env.step(action)
            ai_pnl.append(info['total_pnl'])
            
            if done:
                break
            
            state = next_state
        
        # 4. Test simple delta hedger
        delta_pnl = []
        state = env.reset()
        
        for i in range(20):  # Short episode
            if state is None:
                break
            
            # Get delta hedge action
            current_option = option_data.iloc[env.current_step]
            action = self.simple_hedger.get_hedge_position(current_option['delta'])
            
            # Take step
            next_state, reward, done, info = env.step(action)
            delta_pnl.append(info['total_pnl'])
            
            if done:
                break
            
            state = next_state
        
        # 5. Compare results
        if len(ai_pnl) > 0 and len(delta_pnl) > 0:
            ai_metrics = calculate_metrics(ai_pnl)
            delta_metrics = calculate_metrics(delta_pnl)
            
            # Both should have valid metrics
            assert 'cvar_95' in ai_metrics
            assert 'cvar_95' in delta_metrics
            assert 'mean_pnl' in ai_metrics
            assert 'mean_pnl' in delta_metrics
    
    def test_model_saving_loading(self):
        """Test model saving and loading"""
        import tempfile
        import os
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            # Save model
            torch.save(self.hedger.state_dict(), model_path)
            
            # Load model
            new_hedger = ActorCriticHedger()
            new_hedger.load_state_dict(torch.load(model_path))
            
            # Test that models are equivalent
            test_input = torch.randn(1, 60, 10)
            
            with torch.no_grad():
                action1, value1 = self.hedger(test_input)
                action2, value2 = new_hedger(test_input)
                
                assert torch.allclose(action1, action2)
                assert torch.allclose(value1, value2)
        
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)


if __name__ == "__main__":
    pytest.main([__file__]) 