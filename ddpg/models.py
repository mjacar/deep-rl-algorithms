import torch
import torch.nn as nn


class ValueFunction(nn.Module):
    def __init__(self, obs_dim, action_dim):
        """
        Arguments:
            obs_dim: dimensionality of the observation space
            action_dim: dimensionality of the action space
        """
        super(ValueFunction, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 400)
        self.fc2 = nn.Linear(400 + action_dim, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, s, a):
        x = torch.relu(self.fc1(s))
        x = torch.relu(self.fc2(torch.cat([x, a], 1)))
        return self.fc3(x)


class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        """
        Arguments:
            obs_dim: dimensionality of the observation space
            action_dim: dimensionality of the action space
        """
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

    def forward(self, s):
        x = torch.relu(self.fc1(s))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
