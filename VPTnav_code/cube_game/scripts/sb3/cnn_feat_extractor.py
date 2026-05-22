import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
import gymnasium as gym
from gymnasium import spaces

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # CNN layers for feature extraction (bigger model)
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=8, stride=4),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_space.shape)
            cnn_output_size = self.cnn(dummy_input).shape[1]
        
        # Linear layer to map CNN output to desired features_dim
        self.linear = nn.Linear(cnn_output_size, features_dim)

    def forward(self, observations):
        # Handle both single observations and batched observations
        if observations.dim() == 4:
            # Single observation: (batch, channels, height, width)
            observations = observations.float() / 255.0
            cnn_out = self.cnn(observations)
            features = self.linear(cnn_out)
        else:
            # Batched observations: reshape and process
            batch_size = observations.size(0)
            observations = observations.float() / 255.0
            observations = observations.view(-1, *self._observation_space.shape)
            
            cnn_out = self.cnn(observations)
            features = self.linear(cnn_out)
            features = features.view(batch_size, -1)

        return features