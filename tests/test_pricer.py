"""
Unit tests for Black-Scholes pricer
"""

import pytest
import numpy as np
from src.deephedge.data.dataloader import BlackScholesPricer


class TestBlackScholesPricer:
    """Test cases for Black-Scholes option pricing"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.pricer = BlackScholesPricer(risk_free_rate=0.02)
    
    def test_black_scholes_call(self):
        """Test Black-Scholes call option pricing"""
        S = 100  # Stock price
        K = 100  # Strike price
        T = 1.0  # Time to expiry
        sigma = 0.2  # Volatility
        
        price = self.pricer.black_scholes(S, K, T, sigma, 'call')
        
        # Basic sanity checks
        assert price > 0
        assert price < S  # Call price should be less than stock price
        assert isinstance(price, float)
    
    def test_black_scholes_put(self):
        """Test Black-Scholes put option pricing"""
        S = 100
        K = 100
        T = 1.0
        sigma = 0.2
        
        price = self.pricer.black_scholes(S, K, T, sigma, 'put')
        
        # Basic sanity checks
        assert price > 0
        assert isinstance(price, float)
    
    def test_delta(self):
        """Test option delta calculation"""
        S = 100
        K = 100
        T = 1.0
        sigma = 0.2
        
        call_delta = self.pricer.delta(S, K, T, sigma, 'call')
        put_delta = self.pricer.delta(S, K, T, sigma, 'put')
        
        # Delta should be between 0 and 1 for call, -1 and 0 for put
        assert 0 <= call_delta <= 1
        assert -1 <= put_delta <= 0
        assert call_delta - put_delta == 1  # Put-call parity for delta
    
    def test_gamma(self):
        """Test option gamma calculation"""
        S = 100
        K = 100
        T = 1.0
        sigma = 0.2
        
        gamma = self.pricer.gamma(S, K, T, sigma)
        
        # Gamma should be positive
        assert gamma > 0
        assert isinstance(gamma, float)
    
    def test_theta(self):
        """Test option theta calculation"""
        S = 100
        K = 100
        T = 1.0
        sigma = 0.2
        
        call_theta = self.pricer.theta(S, K, T, sigma, 'call')
        put_theta = self.pricer.theta(S, K, T, sigma, 'put')
        
        # Theta should be negative (time decay)
        assert call_theta < 0
        assert put_theta < 0
        assert isinstance(call_theta, float)
        assert isinstance(put_theta, float)
    
    def test_vega(self):
        """Test option vega calculation"""
        S = 100
        K = 100
        T = 1.0
        sigma = 0.2
        
        vega = self.pricer.vega(S, K, T, sigma)
        
        # Vega should be positive
        assert vega > 0
        assert isinstance(vega, float)
    
    def test_implied_volatility(self):
        """Test implied volatility calculation"""
        S = 100
        K = 100
        T = 1.0
        sigma_true = 0.2
        
        # Calculate option price with known volatility
        option_price = self.pricer.black_scholes(S, K, T, sigma_true, 'call')
        
        # Calculate implied volatility
        sigma_implied = self.pricer.calculate_implied_volatility(S, K, T, option_price, 'call')
        
        # Should be close to true volatility
        assert abs(sigma_implied - sigma_true) < 0.01
        assert 0.01 <= sigma_implied <= 2.0  # Within reasonable bounds


if __name__ == "__main__":
    pytest.main([__file__]) 