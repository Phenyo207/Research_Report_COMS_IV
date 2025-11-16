import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        nn.init.constant_(m.bias, 0.0)


class Critic(nn.Module):
    """Twin Q critics with concatenated [state | goal | action]."""
    def __init__(self, state_dim, action_dim, goal_dim, hidden=256):
        super().__init__()
        input_dim = state_dim + goal_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.apply(weights_init_)

    def forward(self, state_goal, action):
        x = torch.cat([state_goal, action], dim=-1)
        return self.q1(x), self.q2(x)


class Actor(nn.Module):
    """Gaussian policy with Tanh squashing."""
    def __init__(self, state_dim, action_dim, goal_dim, action_space=None, hidden=256):
        super().__init__()
        input_dim = state_dim + goal_dim
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
        self.apply(weights_init_)

        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.as_tensor(
                (action_space.high - action_space.low) / 2.0, dtype=torch.float32)
            self.action_bias = torch.as_tensor(
                (action_space.high + action_space.low) / 2.0, dtype=torch.float32)

    def forward(self, state_goal):
        x = F.relu(self.fc1(state_goal))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state_goal, deterministic=False):
        mean, log_std = self.forward(state_goal)
        std = log_std.exp()
        dist = Normal(mean, std)
        if deterministic:
            z = mean
        else:
            z = dist.rsample()
        y = torch.tanh(z)
        action = y * self.action_scale + self.action_bias

        if deterministic:
            log_prob = None
        else:
            log_prob = dist.log_prob(z)
            log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + EPS)
            log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)
