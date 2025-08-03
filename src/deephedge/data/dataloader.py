"""
Data loading & synthetic data generation for deep hedging experiments
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class BlackScholesPricer:
    """Black-Scholes option pricing and Greeks calculation"""
    
    def __init__(self, risk_free_rate=0.02):
        self.r = risk_free_rate
    
    def calculate_implied_volatility(self, S, K, T, option_price, option_type='call'):
        """Calculate implied volatility using Newton-Raphson method"""
        def objective(sigma):
            return self.black_scholes(S, K, T, sigma, option_type) - option_price
        
        # Initial guess
        sigma = 0.3
        for _ in range(100):
            f = objective(sigma)
            if abs(f) < 1e-6:
                break
            f_prime = self.vega(S, K, T, sigma)
            if abs(f_prime) < 1e-10:
                break
            sigma = sigma - f / f_prime
        return max(0.01, min(2.0, sigma))
    
    def black_scholes(self, S, K, T, sigma, option_type='call'):
        """Black-Scholes option pricing"""
        d1 = (np.log(S/K) + (self.r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-self.r*T)*norm.cdf(d2)
        else:  # put
            price = K*np.exp(-self.r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        return price
    
    def delta(self, S, K, T, sigma, option_type='call'):
        """Option delta"""
        d1 = (np.log(S/K) + (self.r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    def gamma(self, S, K, T, sigma):
        """Option gamma"""
        d1 = (np.log(S/K) + (self.r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return norm.pdf(d1) / (S*sigma*np.sqrt(T))
    
    def theta(self, S, K, T, sigma, option_type='call'):
        """Option theta"""
        d1 = (np.log(S/K) + (self.r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        theta = (-S*sigma*norm.pdf(d1))/(2*np.sqrt(T)) - self.r*K*np.exp(-self.r*T)*norm.cdf(d2)
        if option_type == 'put':
            theta = theta + self.r*K*np.exp(-self.r*T)
        return theta
    
    def vega(self, S, K, T, sigma):
        """Option vega"""
        d1 = (np.log(S/K) + (self.r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return S*np.sqrt(T)*norm.pdf(d1)


class DataManager:
    """Manages market data and option pricing"""
    
    def __init__(self, start_date='2015-01-01', end_date='2025-01-01'):
        self.start_date = start_date
        self.end_date = end_date
        self.pricer = BlackScholesPricer()
        
    def fetch_sp500_data(self):
        """Fetch S&P 500 E-mini futures data"""
        try:
            # Using SPY as proxy for ES futures
            # Limit to recent data due to yfinance limitations
            recent_start = '2024-01-01'
            ticker = "SPY"
            data = yf.download(ticker, start=recent_start, end=self.end_date, interval='1d', auto_adjust=False)
            
            # Resample to 8-minute bars (simulate intraday data)
            data = data.resample('8T').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            # Calculate returns and volatility
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(20).std() * np.sqrt(252 * 24 * 7.5)  # Annualized
            
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            print("Using synthetic data for demonstration...")
            # Generate synthetic data for demonstration
            return self.generate_synthetic_data()
    
    def generate_synthetic_data(self):
        """Generate synthetic S&P 500 data for demonstration"""
        np.random.seed(42)
        n_days = 365 * 2  # 2 years for faster execution
        n_periods = int(n_days * 24 * 7.5)  # 8-minute periods
        
        # Geometric Brownian Motion
        mu = 0.08  # Annual return
        sigma = 0.15  # Annual volatility
        dt = 1 / (252 * 24 * 7.5)  # 8-minute periods
        
        returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_periods)
        prices = 100 * np.exp(np.cumsum(returns))
        
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='8T')[:len(prices)]
        
        data = pd.DataFrame({
            'Open': prices,
            'High': prices * (1 + np.abs(np.random.normal(0, 0.001, len(prices)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.001, len(prices)))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(prices))
        }, index=dates)
        
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(20).std() * np.sqrt(252 * 24 * 7.5)
        
        return data
    
    def calculate_option_prices(self, data, strike_price=None, time_to_expiry=1/252):
        """Calculate option prices and Greeks for each timestep"""
        if strike_price is None:
            strike_price = data['Close'].iloc[0]  # ATM option
        
        option_data = []
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            S = row['Close']
            sigma = row['Volatility'] if not pd.isna(row['Volatility']) else 0.15
            
            # Calculate option price and Greeks
            option_price = self.pricer.black_scholes(S, strike_price, time_to_expiry, sigma)
            delta = self.pricer.delta(S, strike_price, time_to_expiry, sigma)
            gamma = self.pricer.gamma(S, strike_price, time_to_expiry, sigma)
            theta = self.pricer.theta(S, strike_price, time_to_expiry, sigma)
            
            option_data.append({
                'timestamp': timestamp,
                'underlying_price': S,
                'option_price': option_price,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'volatility': sigma,
                'strike': strike_price
            })
        
        option_df = pd.DataFrame(option_data)
        # Set the index to match the original data
        option_df.set_index('timestamp', inplace=True)
        return option_df 