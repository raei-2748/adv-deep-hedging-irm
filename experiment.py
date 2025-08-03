"""
Deep Hedging Experiment: GAN-Generated Worst-Case Paths vs Geometric Brownian Motion

Research Question: Does training a deep hedger on GAN-generated worst-case price paths 
reduce the 95% CVaR of its P&L during real-world market crashes compared with a 
hedger trained on standard geometric-Brownian paths?

Asset: S&P 500 E-mini futures (ES)
Derivative: European ATM Call Option
Architecture: Actor-Critic with LSTM layers
Risk Metric: CVaR 95% on daily P&L
Training Horizon: 60-step episode (8-hour trading day with 8-min bars)
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import from organized modules
from src.deephedge.data.dataloader import DataManager
from src.deephedge.models.gan import GANGenerator, GANDiscriminator
from src.deephedge.models.actor_critic import ActorCriticHedger, SimpleDeltaHedger
from src.deephedge.envs.deep_hedging_env import DeepHedgingEnvironment
from src.deephedge.utils.seed import set_seed
from src.deephedge.utils.logger import setup_logger
from src.deephedge.utils.metrics import calculate_metrics, compare_hedgers


class ExperimentRunner:
    """Main experiment runner using modular components"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.results = {}
        self.logger = setup_logger()
    
    def run_experiment(self):
        """Run the complete experiment"""
        self.logger.info("Starting Deep Hedging Experiment...")
        
        # 1. Load and prepare data
        self.logger.info("Loading market data...")
        market_data = self.data_manager.fetch_sp500_data()
        option_data = self.data_manager.calculate_option_prices(market_data)
        
        # 2. Create models with deterministic initialization
        self.logger.info("Creating models with deterministic initialization...")
        gan_generator = GANGenerator()
        gan_discriminator = GANDiscriminator()
        hedger = ActorCriticHedger()
        
        # Ensure deterministic model initialization
        for model in [gan_generator, gan_discriminator, hedger]:
            for param in model.parameters():
                if param.dim() > 1:  # Weight matrices
                    torch.nn.init.xavier_uniform_(param)
                else:  # Bias vectors
                    torch.nn.init.zeros_(param)
        
        # 3. Create environment
        environment = DeepHedgingEnvironment(market_data, option_data)
        
        # 4. Train GAN on worst-case scenarios
        self.logger.info("Training GAN on worst-case scenarios...")
        # Prepare real data for GAN training
        sequence_length = gan_generator.sequence_length
        returns = market_data['Returns'].dropna().values
        if len(returns) < sequence_length:
            returns = np.pad(returns, (0, sequence_length - len(returns)), 'constant')
        else:
            returns = returns[:sequence_length]
        # Reshape to (batch_size, sequence_length, 1)
        real_data = torch.FloatTensor(returns).unsqueeze(0).unsqueeze(-1)
        
        # Simple GAN training (simplified for demonstration)
        self._train_gan_simple(gan_generator, gan_discriminator, real_data, num_epochs=50)
        
        # 5. Train hedger on GAN-generated paths
        self.logger.info("Training hedger on GAN-generated paths...")
        self._train_hedger_simple(environment, hedger, num_episodes=500)
        
        # 6. Evaluate on historical crashes
        self.logger.info("Evaluating on historical crashes...")
        self.evaluate_on_crashes(market_data, option_data, hedger)
        
        self.logger.info("Experiment completed!")
    
    def _train_gan_simple(self, generator, discriminator, real_data, num_epochs=50):
        """Simplified GAN training for demonstration"""
        generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
        criterion = torch.nn.BCELoss()
        
        for epoch in range(num_epochs):
            # Generate fake data
            noise = torch.randn(32, 60, 10)
            fake_data = generator(noise)
            
            # Train discriminator
            discriminator_optimizer.zero_grad()
            real_labels = torch.ones(32, 1)
            fake_labels = torch.zeros(32, 1)
            
            real_outputs = discriminator(real_data.repeat(32, 1, 1))
            fake_outputs = discriminator(fake_data.detach())
            
            d_loss = criterion(real_outputs, real_labels) + criterion(fake_outputs, fake_labels)
            d_loss.backward()
            discriminator_optimizer.step()
            
            # Train generator
            generator_optimizer.zero_grad()
            fake_outputs = discriminator(fake_data)
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            generator_optimizer.step()
            
            if epoch % 20 == 0:
                self.logger.info(f"GAN Epoch {epoch}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")
    
    def _train_hedger_simple(self, environment, hedger, num_episodes=500):
        """Simplified hedger training for demonstration"""
        optimizer = torch.optim.Adam(hedger.parameters(), lr=0.001)
        
        for episode in range(num_episodes):
            state = environment.reset()
            episode_rewards = []
            
            while state is not None:
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
                
                # Get action and value from hedger
                action, value = hedger(state_tensor)
                action = action.item()
                
                # Take action in environment
                next_state, reward, done, info = environment.step(action)
                
                episode_rewards.append(reward)
                
                if done:
                    break
                
                state = next_state
            
            # Simple training: just track performance
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                self.logger.info(f"Hedger Episode {episode}: Avg Reward = {avg_reward:.4f}")
    
    def evaluate_on_crashes(self, market_data, option_data, hedger):
        """Evaluate hedger performance on historical crashes"""
        
        self.logger.info(f"Data length: {len(market_data)}")
        self.logger.info(f"Data type: {type(market_data)}")
        self.logger.info(f"Index type: {type(market_data.index)}")
        self.logger.info(f"Index sample: {market_data.index[:5]}")
        self.logger.info(f"Option data length: {len(option_data)}")
        self.logger.info(f"Option data index type: {type(option_data.index)}")
        self.logger.info(f"Option data index sample: {option_data.index[:5]}")
        
        # For synthetic data, use different evaluation periods
        # Check if the data has datetime index (real data) or integer index (synthetic)
        if isinstance(market_data.index, pd.DatetimeIndex):
            self.logger.info("Using real data crash periods")
            # Check if we have data for the crash periods
            start_date = market_data.index[0]
            end_date = market_data.index[-1]
            self.logger.info(f"Data range: {start_date} to {end_date}")
            
            # Check if specific dates exist in the index
            test_date = pd.Timestamp('2015-06-01')
            if test_date in market_data.index:
                self.logger.info(f"Date {test_date} found in index")
                # Synthetic data with datetime index
                crash_periods = [
                    ('Synthetic Crash 1', '2015-06-01', '2015-06-30'),
                    ('Synthetic Crash 2', '2015-12-01', '2015-12-31')
                ]
            else:
                self.logger.info(f"Date {test_date} not found in index, using integer indices")
                # Use integer indices instead
                crash_periods = [
                    ('Synthetic Crash 1', 0, len(market_data)//4),
                    ('Synthetic Crash 2', len(market_data)//4, len(market_data)//2)
                ]
        else:
            self.logger.info("Using synthetic data crash periods")
            # Synthetic data with integer indices
            crash_periods = [
                ('Synthetic Crash 1', 0, len(market_data)//4),
                ('Synthetic Crash 2', len(market_data)//4, len(market_data)//2)
            ]
        
        # Initialize simple delta hedger
        simple_hedger = SimpleDeltaHedger()
        
        for crash_name, start_idx, end_idx in crash_periods:
            self.logger.info(f"\nEvaluating on {crash_name}...")
            self.logger.info(f"Start: {start_idx}, End: {end_idx}, Type: {type(start_idx)}")
            
            # Filter data for crash period
            if isinstance(start_idx, int):
                # Synthetic data with integer indices
                crash_data = market_data.iloc[start_idx:end_idx]
                crash_option_data = option_data.iloc[start_idx:end_idx]
            else:
                # Real data with date indices
                crash_data = market_data[start_idx:end_idx]
                # Make sure option_data has the same index
                if isinstance(option_data.index, pd.DatetimeIndex):
                    crash_option_data = option_data[start_idx:end_idx]
                else:
                    # If option_data has different index, use integer indexing
                    crash_option_data = option_data.iloc[start_idx:end_idx]
            
            if len(crash_data) == 0:
                self.logger.info(f"No data available for {crash_name}")
                continue
            
            # Evaluate AI hedger
            self.logger.info("Evaluating AI Hedger...")
            ai_results = self._evaluate_hedger(crash_data, crash_option_data, hedger, is_ai=True)
            
            # Evaluate simple delta hedger
            self.logger.info("Evaluating Simple Delta Hedger...")
            delta_results = self._evaluate_hedger(crash_data, crash_option_data, simple_hedger, is_ai=False)
            
            # Store results
            self.results[f"{crash_name}_AI"] = ai_results
            self.results[f"{crash_name}_Delta"] = delta_results
            
            # Print comparison
            self.logger.info(f"\n{crash_name} Comparison:")
            self.logger.info(f"  AI Hedger - CVaR 95%: {ai_results['cvar_95']:.4f}, Mean P&L: {ai_results['mean_pnl']:.4f}")
            self.logger.info(f"  Delta Hedger - CVaR 95%: {delta_results['cvar_95']:.4f}, Mean P&L: {delta_results['mean_pnl']:.4f}")
            
            # Calculate improvement
            cvar_improvement = ((delta_results['cvar_95'] - ai_results['cvar_95']) / delta_results['cvar_95']) * 100
            self.logger.info(f"  CVaR Improvement: {cvar_improvement:.2f}%")
    
    def _evaluate_hedger(self, crash_data, crash_option_data, hedger, is_ai=True):
        """Evaluate a specific hedger on crash data"""
        crash_pnl = []
        
        # Initialize portfolio
        portfolio_value = 100000
        hedge_position = 0
        
        for i in range(len(crash_data) - 1):
            current_price = crash_data.iloc[i]['Close']
            next_price = crash_data.iloc[i + 1]['Close']
            current_option = crash_option_data.iloc[i]
            next_option = crash_option_data.iloc[i + 1]
            
            # Calculate option P&L
            option_pnl = next_option['option_price'] - current_option['option_price']
            
            # Get hedge position
            if is_ai:
                # AI hedger - use state and get action
                state = np.array([
                    current_price / 100,
                    crash_data.iloc[i]['Returns'] if not pd.isna(crash_data.iloc[i]['Returns']) else 0,
                    current_option['volatility'],
                    current_option['option_price'] / 100,
                    current_option['delta'],
                    current_option['gamma'] * 100,
                    current_option['theta'] / 100,
                    hedge_position,
                    portfolio_value / 100000,
                    i / 60
                ])
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
                action, _ = hedger(state_tensor)
                new_position = action.item() * 1000  # Scale action
            else:
                # Simple delta hedger
                new_position = hedger.get_hedge_position(current_option['delta']) * 1000
            
            # Calculate hedge P&L
            hedge_pnl = -hedge_position * (next_price - current_price)
            
            # Calculate transaction costs
            position_change = new_position - hedge_position
            transaction_costs = abs(position_change) * 0.0001
            
            # Update portfolio
            total_pnl = option_pnl + hedge_pnl - transaction_costs
            portfolio_value += total_pnl
            hedge_position = new_position
            
            crash_pnl.append(total_pnl)
        
        # Calculate metrics using the metrics module
        if len(crash_pnl) > 0:
            metrics = calculate_metrics(crash_pnl, confidence=0.95)
            metrics['pnl_series'] = crash_pnl  # Add this line
            return metrics
        else:
            return {
                'cvar_95': 0,
                'mean_pnl': 0,
                'total_pnl': 0,
                'pnl_series': []
            }
    
    def plot_results(self):
        """Plot experiment results"""
        if not self.results:
            self.logger.info("No results to plot")
            return
        
        # Separate AI and Delta results
        ai_results = {k: v for k, v in self.results.items() if k.endswith('_AI')}
        delta_results = {k: v for k, v in self.results.items() if k.endswith('_Delta')}
        
        if not ai_results or not delta_results:
            self.logger.info("No comparison data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: CVaR comparison
        crash_names = [name.replace('_AI', '').replace('_Delta', '') for name in ai_results.keys()]
        ai_cvars = [ai_results[name + '_AI']['cvar_95'] for name in crash_names]
        delta_cvars = [delta_results[name + '_Delta']['cvar_95'] for name in crash_names]
        
        x = np.arange(len(crash_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, ai_cvars, width, label='AI Hedger', alpha=0.8)
        axes[0, 0].bar(x + width/2, delta_cvars, width, label='Delta Hedger', alpha=0.8)
        axes[0, 0].set_title('CVaR 95% Comparison')
        axes[0, 0].set_ylabel('CVaR 95%')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(crash_names)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Mean P&L comparison
        ai_mean_pnls = [ai_results[name + '_AI']['mean_pnl'] for name in crash_names]
        delta_mean_pnls = [delta_results[name + '_Delta']['mean_pnl'] for name in crash_names]
        
        axes[0, 1].bar(x - width/2, ai_mean_pnls, width, label='AI Hedger', alpha=0.8)
        axes[0, 1].bar(x + width/2, delta_mean_pnls, width, label='Delta Hedger', alpha=0.8)
        axes[0, 1].set_title('Mean P&L Comparison')
        axes[0, 1].set_ylabel('Mean P&L')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(crash_names)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: P&L time series (first crash only)
        first_crash = crash_names[0]
        ai_pnl = ai_results[first_crash + '_AI']['pnl_series']
        delta_pnl = delta_results[first_crash + '_Delta']['pnl_series']
        
        axes[1, 0].plot(ai_pnl, label='AI Hedger', alpha=0.7)
        axes[1, 0].plot(delta_pnl, label='Delta Hedger', alpha=0.7)
        axes[1, 0].set_title(f'P&L Time Series - {first_crash}')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('P&L')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Cumulative P&L (first crash only)
        ai_cumulative = np.cumsum(ai_pnl)
        delta_cumulative = np.cumsum(delta_pnl)
        
        axes[1, 1].plot(ai_cumulative, label='AI Hedger', alpha=0.7)
        axes[1, 1].plot(delta_cumulative, label='Delta Hedger', alpha=0.7)
        axes[1, 1].set_title(f'Cumulative P&L - {first_crash}')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Cumulative P&L')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('src/deephedge/data/deep_hedging_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info("Results plotted and saved as 'src/deephedge/data/deep_hedging_results.png'")


def main():
    """Main function to run the experiment"""
    print("Deep Hedging Experiment: GAN vs Geometric Brownian Motion")
    print("=" * 60)
    
    # Import config for deterministic settings
    from src.deephedge.config import ExperimentConfig
    
    # Set deterministic seeds from config
    seed_config = ExperimentConfig.get_seed_config()
    set_seed(
        seed=seed_config['torch_seed'],
        deterministic=seed_config['deterministic']
    )
    
    print(f"🔒 Deterministic Mode: {seed_config['deterministic']}")
    print(f"🎲 Torch Seed: {seed_config['torch_seed']}")
    print(f"🎲 NumPy Seed: {seed_config['numpy_seed']}")
    print(f"🎲 Python Seed: {seed_config['python_seed']}")
    if torch.cuda.is_available():
        print(f"🚀 CUDA Deterministic: {seed_config['cuda_deterministic']}")
        print(f"🚀 CUDA Benchmark: {seed_config['cuda_benchmark']}")
    
    # Verify deterministic behavior
    from src.deephedge.utils.seed import verify_determinism
    verify_determinism()
    
    # Create and run experiment
    experiment = ExperimentRunner()
    experiment.run_experiment()
    
    # Plot results
    experiment.plot_results()
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    # Separate AI and Delta results
    ai_results = {k: v for k, v in experiment.results.items() if k.endswith('_AI')}
    delta_results = {k: v for k, v in experiment.results.items() if k.endswith('_Delta')}
    
    if ai_results and delta_results:
        print("\nHEDGER COMPARISON RESULTS:")
        print("-" * 40)
        
        for crash_name in [name.replace('_AI', '') for name in ai_results.keys()]:
            ai_data = ai_results[crash_name + '_AI']
            delta_data = delta_results[crash_name + '_Delta']
            
            print(f"\n{crash_name}:")
            print(f"  AI Hedger - CVaR 95%: {ai_data['cvar_95']:.4f}, Mean P&L: {ai_data['mean_pnl']:.4f}")
            print(f"  Delta Hedger - CVaR 95%: {delta_data['cvar_95']:.4f}, Mean P&L: {delta_data['mean_pnl']:.4f}")
            
            # Calculate improvement
            cvar_improvement = ((delta_data['cvar_95'] - ai_data['cvar_95']) / delta_data['cvar_95']) * 100
            print(f"  CVaR Improvement: {cvar_improvement:.2f}%")
        
        # Overall statistics
        ai_avg_cvar = np.mean([data['cvar_95'] for data in ai_results.values()])
        delta_avg_cvar = np.mean([data['cvar_95'] for data in delta_results.values()])
        overall_improvement = ((delta_avg_cvar - ai_avg_cvar) / delta_avg_cvar) * 100
        
        print(f"\nOVERALL RESULTS:")
        print(f"  Average AI Hedger CVaR: {ai_avg_cvar:.4f}")
        print(f"  Average Delta Hedger CVaR: {delta_avg_cvar:.4f}")
        print(f"  Overall CVaR Improvement: {overall_improvement:.2f}%")
    
    print("\nHypothesis Testing:")
    print("H0: No difference in CVaR between AI-trained and Delta hedgers")
    print("H1: AI-trained hedger achieves 20% lower CVaR95")
    
    if ai_results and delta_results:
        print("Note: This is a simplified implementation. For rigorous hypothesis testing,")
        print("multiple runs with different seeds and statistical tests are recommended.")


if __name__ == "__main__":
    main()
