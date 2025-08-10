"""
Data loading & synthetic data generation for deep hedging experiments
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import warnings
import pathlib
warnings.filterwarnings('ignore')

CACHE_DIR = pathlib.Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


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

    def _get_cache_path(self, symbol, interval):
        return CACHE_DIR / f"{symbol}_{self.start_date}_{self.end_date}_{interval}.pkl"

    def _load_from_cache(self, symbol, interval):
        cache_path = self._get_cache_path(symbol, interval)
        if cache_path.exists():
            return pd.read_pickle(cache_path)
        return None

    def _save_to_cache(self, data, symbol, interval):
        cache_path = self._get_cache_path(symbol, interval)
        data.to_pickle(cache_path)
        
    def fetch_sp500_data(self, symbol='SPY', interval='1d', sequence_length=60):
        """Fetch S&P 500 E-mini futures data and generate intraday paths."""
        # Try to load from cache first
        cached_data = self._load_from_cache(symbol, interval)
        if cached_data is not None:
            print("Loaded data from cache.")
            return cached_data

        try:
            print(f"Fetching daily data for {symbol} from {self.start_date} to {self.end_date}...")
            daily_data = yf.download(symbol, start=self.start_date, end=self.end_date, interval=interval, auto_adjust=False)
            
            if daily_data.empty:
                raise ValueError("No daily data fetched. Check symbol and date range.")

            # Save to cache
            self._save_to_cache(daily_data, symbol, interval)
            
            daily_data['Returns'] = daily_data['Close'].pct_change()
            daily_data['Volatility'] = daily_data['Returns'].rolling(20).std() * np.sqrt(252) # Annualized daily volatility
            
            intraday_data_list = []
            
            print("Generating synthetic intraday data...")
            for i in range(len(daily_data) - 1): # Iterate through days
                current_day_close = daily_data['Close'].iloc[i]
                next_day_close = daily_data['Close'].iloc[i+1]
                daily_volatility = daily_data['Volatility'].iloc[i]
                
                if pd.isna(daily_volatility) or daily_volatility == 0:
                    daily_volatility = 0.15 # Default volatility if not available
                
                # Generate intraday path using GBM, starting from current_day_close
                # and ending approximately at next_day_close
                intraday_path = self._generate_gbm_path(
                    S0=current_day_close,
                    mu=(next_day_close - current_day_close) / current_day_close, # Daily return as drift
                    sigma=daily_volatility,
                    dt=1/252, # Daily time step
                    num_steps=sequence_length # Number of 8-minute bars in a day
                )
                
                # Create a DataFrame for the intraday path
                intraday_df = pd.DataFrame({
                    'Close': intraday_path,
                    'Open': intraday_path, # Simplified for now
                    'High': intraday_path * (1 + np.abs(np.random.normal(0, 0.001, len(intraday_path)))),
                    'Low': intraday_path * (1 - np.abs(np.random.normal(0, 0.001, len(intraday_path)))),
                    'Volume': np.random.randint(100000, 1000000, len(intraday_path))
                })
                
                # Assign a time index for the day
                current_date = daily_data.index[i]
                time_index = pd.date_range(start=current_date, periods=sequence_length, freq='8T')
                intraday_df.index = time_index
                
                intraday_data_list.append(intraday_df)
            
            if not intraday_data_list:
                raise ValueError("No intraday data generated.")
                
            combined_intraday_data = pd.concat(intraday_data_list)
            
            # Calculate returns and volatility for the combined intraday data
            combined_intraday_data['Returns'] = combined_intraday_data['Close'].pct_change()
            # Annualize based on the number of intraday steps per year (252 trading days * sequence_length)
            combined_intraday_data['Volatility'] = combined_intraday_data['Returns'].rolling(20).std() * np.sqrt(252 * sequence_length)
            
            return combined_intraday_data.dropna()
            
        except Exception as e:
            print(f"Error fetching or generating data: {e}")
            print("Using fully synthetic data for demonstration...")
            return self.generate_fully_synthetic_data(sequence_length=sequence_length)
    
    def _generate_gbm_path(self, S0, mu, sigma, dt, num_steps):
        """Generates a single Geometric Brownian Motion path."""
        prices = np.zeros(num_steps)
        prices[0] = S0
        for t in range(1, num_steps):
            shock = np.random.normal(0, 1)
            prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shock)
        return prices

    def generate_fully_synthetic_data(self, n_days=365 * 2, sequence_length=60):
        """Generate fully synthetic S&P 500 data for demonstration (if real data fails)."""
        np.random.seed(42)
        
        mu = 0.08  # Annual return
        sigma = 0.15  # Annual volatility
        
        # Total number of intraday periods
        total_periods = n_days * sequence_length
        
        dt_intraday = 1 / (252 * sequence_length) # Time step for intraday periods
        
        returns = np.random.normal(mu * dt_intraday, sigma * np.sqrt(dt_intraday), total_periods)
        prices = 100 * np.exp(np.cumsum(returns))
        
        dates = pd.date_range(start=self.start_date, periods=total_periods, freq='8T')
        
        data = pd.DataFrame({
            'Open': prices,
            'High': prices * (1 + np.abs(np.random.normal(0, 0.001, len(prices)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.001, len(prices)))),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(prices))
        }, index=dates)
        
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(20).std() * np.sqrt(252 * sequence_length)
        
        return data.dropna()
    
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

    def get_data(self, synthetic=False):
        if synthetic:
            market_data = self.generate_fully_synthetic_data()
        else:
            market_data = self.fetch_sp500_data()
        
        option_data = self.calculate_option_prices(market_data)
        return market_data, option_data 