import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        """
        Arguments:
            obs_dim: dimensionality of the observation space
            action_dim: dimensionality of the action space
        """
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mu = nn.Linear(64, action_dim)
        self.logstd = nn.Parameter(torch.zeros(action_dim))

    def forward(self, s):
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        return self.mu(x), self.logstd


class ValueFunction(nn.Module):
    def __init__(self, obs_dim):
        """
        Arguments:
            obs_dim: dimensionality of the observation space
        """
        super(ValueFunction, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.v = nn.Linear(64, 1)

    def forward(self, s):
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        return self.v(x)
