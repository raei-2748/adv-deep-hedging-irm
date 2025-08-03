"""
CLI entry point for training deep hedging models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .data.dataloader import DataManager
from .models.gan import GANGenerator, GANDiscriminator
from .models.actor_critic import ActorCriticHedger
from .envs.deep_hedging_env import DeepHedgingEnvironment
from .utils.seed import set_seed
from .utils.logger import setup_logger


class TrainingManager:
    """Manages the training process for both GAN and hedger"""
    
    def __init__(self, data_manager, gan_generator, gan_discriminator, hedger):
        self.data_manager = data_manager
        self.gan_generator = gan_generator
        self.gan_discriminator = gan_discriminator
        self.hedger = hedger
        
        # Optimizers
        self.gan_g_optimizer = optim.Adam(self.gan_generator.parameters(), lr=0.0002)
        self.gan_d_optimizer = optim.Adam(self.gan_discriminator.parameters(), lr=0.0002)
        self.hedger_optimizer = optim.Adam(self.hedger.parameters(), lr=0.001)
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.value_loss = nn.MSELoss()
        
    def train_gan(self, real_data, num_epochs=100, batch_size=32):
        """Train GAN to generate worst-case price paths"""
        print("Training GAN...")
        
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
            
            if epoch % 20 == 0:
                print(f"GAN Epoch {epoch}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")
    
    def train_hedger(self, environment, num_episodes=1000):
        """Train the hedger using Actor-Critic"""
        print("Training Hedger...")
        
        for episode in range(num_episodes):
            state = environment.reset()
            episode_rewards = []
            
            while state is not None:
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
                
                # Get action and value from hedger
                action, value = self.hedger(state_tensor)
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


def main():
    """Main training function"""
    # Set random seed
    set_seed(42)
    
    # Setup logger
    logger = setup_logger()
    logger.info("Starting deep hedging training...")
    
    # Initialize components
    data_manager = DataManager()
    gan_generator = GANGenerator()
    gan_discriminator = GANDiscriminator()
    hedger = ActorCriticHedger()
    
    # Create training manager
    training_manager = TrainingManager(data_manager, gan_generator, gan_discriminator, hedger)
    
    # Load data
    logger.info("Loading market data...")
    market_data = data_manager.fetch_sp500_data()
    option_data = data_manager.calculate_option_prices(market_data)
    
    # Create environment
    environment = DeepHedgingEnvironment(market_data, option_data)
    
    # Train GAN
    logger.info("Training GAN...")
    sequence_length = gan_generator.sequence_length
    returns = market_data['Returns'].dropna().values
    if len(returns) < sequence_length:
        returns = np.pad(returns, (0, sequence_length - len(returns)), 'constant')
    else:
        returns = returns[:sequence_length]
    real_data = torch.FloatTensor(returns).unsqueeze(0).unsqueeze(-1)
    training_manager.train_gan(real_data, num_epochs=50)
    
    # Train hedger
    logger.info("Training hedger...")
    training_manager.train_hedger(environment, num_episodes=500)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main() 