"""
GAN models for generating worst-case price paths
"""

import torch.nn as nn


class GANGenerator(nn.Module):
    """GAN Generator for worst-case price paths"""
    
    def __init__(self, input_dim=10, hidden_dim=64, sequence_length=60):
        super(GANGenerator, self).__init__()
        self.sequence_length = sequence_length
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, noise):
        # noise shape: (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(noise)
        lstm_out = self.dropout(lstm_out)
        price_changes = self.fc(lstm_out)
        return price_changes


class GANDiscriminator(nn.Module):
    """GAN Discriminator to distinguish real vs generated paths"""
    
    def __init__(self, input_dim=1, hidden_dim=64, sequence_length=60):
        super(GANDiscriminator, self).__init__()
        self.sequence_length = sequence_length
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, price_paths):
        # price_paths shape: (batch_size, sequence_length, 1)
        lstm_out, _ = self.lstm(price_paths)
        # Handle different tensor shapes
        if len(lstm_out.shape) == 3:
            lstm_out = self.dropout(lstm_out[:, -1, :])  # Take last output
        else:
            lstm_out = self.dropout(lstm_out)
        validity = self.sigmoid(self.fc(lstm_out))
        return validity 