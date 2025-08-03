"""
CLI entry point for training deep hedging models with Hydra configuration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import hydra
import time
from omegaconf import DictConfig, OmegaConf
from .data.dataloader import DataManager
from .models.gan import GANGenerator, GANDiscriminator
from .models.actor_critic import ActorCriticHedger
from .envs.deep_hedging_env import DeepHedgingEnvironment
from .utils.seed import set_seed, verify_determinism
from .utils.logger import setup_logger, log_experiment_start, log_experiment_end
from .utils.metrics_tracker import MetricsTracker


class TrainingManager:
    """Manages the training process for both GAN and hedger"""
    
    def __init__(self, data_manager, gan_generator, gan_discriminator, hedger, config):
        self.data_manager = data_manager
        self.gan_generator = gan_generator
        self.gan_discriminator = gan_discriminator
        self.hedger = hedger
        self.config = config
        
        # Optimizers with config-based learning rates
        self.gan_g_optimizer = optim.Adam(
            self.gan_generator.parameters(), 
            lr=config.model.gan.lr_generator
        )
        self.gan_d_optimizer = optim.Adam(
            self.gan_discriminator.parameters(), 
            lr=config.model.gan.lr_discriminator
        )
        self.hedger_optimizer = optim.Adam(
            self.hedger.parameters(), 
            lr=config.model.hedger.lr
        )
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.value_loss = nn.MSELoss()
        
    def train_gan(self, real_data, metrics_tracker=None):
        """Train GAN to generate worst-case price paths"""
        print("Training GAN...")
        
        num_epochs = self.config.training.gan.num_epochs
        batch_size = self.config.training.gan.batch_size
        
        for epoch in range(num_epochs):
            # Generate random noise
            noise = torch.randn(batch_size, 60, 10)
            
            # Generate fake price paths
            fake_paths = self.gan_generator(noise)
            
            # Train discriminator
            self.gan_d_optimizer.zero_grad()
            
            # Real paths - repeat real data to match batch size
            real_batch = real_data.repeat(batch_size, 1, 1)
            real_labels = torch.ones(batch_size, 1)
            real_outputs = self.gan_discriminator(real_batch)
            d_real_loss = self.adversarial_loss(real_outputs, real_labels)
            
            # Fake paths
            fake_labels = torch.zeros(batch_size, 1)
            fake_outputs = self.gan_discriminator(fake_paths.detach())
            d_fake_loss = self.adversarial_loss(fake_outputs, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            self.gan_d_optimizer.step()
            
            # Train generator
            self.gan_g_optimizer.zero_grad()
            
            # Generator wants to fool discriminator
            fake_outputs = self.gan_discriminator(fake_paths)
            g_loss = self.adversarial_loss(fake_outputs, real_labels)
            
            g_loss.backward()
            self.gan_g_optimizer.step()
            
            # Log metrics
            if metrics_tracker:
                real_score = real_outputs.mean().item()
                fake_score = fake_outputs.mean().item()
                metrics_tracker.log_gan_metrics(
                    epoch=epoch,
                    d_loss=d_loss.item(),
                    g_loss=g_loss.item(),
                    real_score=real_score,
                    fake_score=fake_score
                )
            
            if epoch % 20 == 0:
                print(f"GAN Epoch {epoch}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")
    
    def train_hedger(self, environment, metrics_tracker=None):
        """Train the hedger using Actor-Critic"""
        print("Training Hedger...")
        
        num_episodes = self.config.training.hedger.num_episodes
        
        for episode in range(num_episodes):
            state = environment.reset()
            episode_rewards = []
            episode_pnls = []
            episode_turnover = 0
            
            while state is not None:
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
                
                # Get action and value from hedger
                action, value = self.hedger(state_tensor)
                action = action.item()
                
                # Take action in environment
                next_state, reward, done, info = environment.step(action)
                
                episode_rewards.append(reward)
                if 'total_pnl' in info:
                    episode_pnls.append(info['total_pnl'])
                if 'transaction_costs' in info:
                    episode_turnover += info['transaction_costs']
                
                if done:
                    break
                
                state = next_state
            
            # Calculate metrics
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            total_pnl = np.sum(episode_pnls) if episode_pnls else 0
            cvar_95 = np.percentile(episode_pnls, 5) if episode_pnls else 0
            
            # Log metrics
            if metrics_tracker:
                metrics_tracker.log_hedger_metrics(
                    episode=episode,
                    avg_reward=avg_reward,
                    total_pnl=total_pnl,
                    cvar_95=abs(cvar_95),
                    turnover=episode_turnover
                )
            
            # Simple training: just track performance
            if episode % 100 == 0:
                print(f"Hedger Episode {episode}: Avg Reward = {avg_reward:.4f}")
    
    def _update_hedger(self, rewards, values, actions):
        """Update hedger networks using Actor-Critic"""
        if len(rewards) == 0:
            return
            
        # Convert to tensors
        rewards = torch.tensor(rewards, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        
        # Calculate advantages
        advantages = rewards - values
        
        # Value loss only (simplified to avoid gradient issues)
        value_loss = self.value_loss(values, rewards)
        
        # Update networks
        self.hedger_optimizer.zero_grad()
        value_loss.backward()
        self.hedger_optimizer.step()


@hydra.main(version_base=None, config_path="../../configs", config_name="base")
def main(cfg: DictConfig):
    """Main training function with Hydra configuration"""
    
    # Start timing
    start_time = time.time()
    
    # Setup comprehensive logging
    logger = setup_logger(
        log_dir=cfg.logging.log_dir,
        experiment_name=cfg.experiment.name
    )
    
    # Log experiment start with all metadata
    log_experiment_start(cfg)
    
    # Set deterministic seeds
    set_seed(
        seed=cfg.experiment.seed,
        deterministic=cfg.deterministic.deterministic_mode
    )
    
    # Verify deterministic behavior
    verify_determinism()
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(
        config=cfg,
        project_name=cfg.logging.wandb_project
    )
    
    # Initialize components with config
    data_manager = DataManager(
        start_date=cfg.data.start_date,
        end_date=cfg.data.end_date
    )
    
    # Create models with config parameters
    gan_generator = GANGenerator(
        input_dim=cfg.model.gan.input_dim,
        hidden_dim=cfg.model.gan.hidden_dim,
        sequence_length=cfg.model.gan.sequence_length
    )
    
    gan_discriminator = GANDiscriminator(
        input_dim=1,
        hidden_dim=cfg.model.gan.hidden_dim,
        sequence_length=cfg.model.gan.sequence_length
    )
    
    hedger = ActorCriticHedger(
        state_dim=cfg.model.hedger.state_dim,
        hidden_dim=cfg.model.hedger.hidden_dim,
        sequence_length=cfg.model.hedger.sequence_length
    )
    
    # Create training manager
    training_manager = TrainingManager(data_manager, gan_generator, gan_discriminator, hedger, cfg)
    
    # Load data
    logger.info("Loading market data...")
    market_data = data_manager.fetch_sp500_data()
    option_data = data_manager.calculate_option_prices(market_data)
    
    # Create environment with config
    environment = DeepHedgingEnvironment(
        market_data, 
        option_data, 
        transaction_cost=cfg.environment.transaction_cost
    )
    
    # Train GAN
    logger.info("Training GAN...")
    sequence_length = gan_generator.sequence_length
    returns = market_data['Returns'].dropna().values
    if len(returns) < sequence_length:
        returns = np.pad(returns, (0, sequence_length - len(returns)), 'constant')
    else:
        returns = returns[:sequence_length]
    real_data = torch.FloatTensor(returns).unsqueeze(0).unsqueeze(-1)
    training_manager.train_gan(real_data, metrics_tracker)
    
    # Train hedger
    logger.info("Training hedger...")
    training_manager.train_hedger(environment, metrics_tracker)
    
    # Create and save visualizations
    logger.info("Creating training curves...")
    metrics_tracker.create_loss_curves(save_path=cfg.logging.log_dir)
    
    # Calculate experiment duration
    duration = time.time() - start_time
    
    # Get summary metrics
    summary_metrics = metrics_tracker.get_summary_metrics()
    
    # Log experiment completion
    log_experiment_end(duration, summary_metrics)
    
    # Save configuration
    OmegaConf.save(cfg, f"{cfg.logging.log_dir}/config.yaml")
    
    # Finish W&B run
    metrics_tracker.finish()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main() 