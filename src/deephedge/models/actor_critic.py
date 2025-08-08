"""
Actor-Critic models for deep hedging
"""

import torch
import torch.nn as nn


class ActorCriticHedger(nn.Module):
    """Actor-Critic deep hedger with LSTM layers"""
    
    def __init__(self, state_dim=10, hidden_dim=64, sequence_length=60):
        super(ActorCriticHedger, self).__init__()
        self.sequence_length = sequence_length
        
        # Shared LSTM layers
        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers=2, batch_first=True)
        
        # Actor network (policy)
        self.actor_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Output position size between -1 and 1
        )
        
        # Critic network (value function)
        self.critic_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state_sequence):
        # state_sequence shape: (batch_size, sequence_length, state_dim)
        lstm_out, _ = self.lstm(state_sequence)
        lstm_out = lstm_out[:, -1, :]  # Take last output
        
        # Actor output (hedge position)
        action = self.actor_fc(lstm_out)
        
        # Critic output (value estimate)
        value = self.critic_fc(lstm_out)
        
        return action, value


 