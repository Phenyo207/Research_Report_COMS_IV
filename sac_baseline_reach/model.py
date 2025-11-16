# model.py
import os
import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        nn.init.constant_(m.bias, 0.0)

# ----------------------------------------------------------------------------
# SAC Critic (Twin Q-networks)
# ----------------------------------------------------------------------------
class GCRLCritic(nn.Module):
    """
    Standard SAC critic with twin Q-networks for GCRL
    Input: [state | goal | action]
    Output: Q1, Q2
    """
    def __init__(self, state_dim, action_dim, goal_dim, hidden_dim=256):
        super().__init__()
        input_dim = state_dim + goal_dim + action_dim

        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(weights_init_)

    def forward(self, state_goal, action):
        """Compute both Q-values"""
        x = torch.cat([state_goal, action], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2

    def Q1(self, state_goal, action):
        """Get Q1 value only"""
        x = torch.cat([state_goal, action], dim=-1)
        return self.q1(x)

# ----------------------------------------------------------------------------
# SAC Actor (Gaussian policy)
# ----------------------------------------------------------------------------
class GCRLActor(nn.Module):
    """
    Standard SAC actor with Gaussian policy for GCRL
    Input: [state | goal]
    Output: action distribution (mean, log_std)
    """
    def __init__(self, state_dim, action_dim, goal_dim, hidden_dim=256, action_space=None):
        super().__init__()
        input_dim = state_dim + goal_dim

        # Shared network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Output heads
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # Action space bounds
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.as_tensor(
                (action_space.high - action_space.low) / 2.0,
                dtype=torch.float32
            )
            self.action_bias = torch.as_tensor(
                (action_space.high + action_space.low) / 2.0,
                dtype=torch.float32
            )

        self.apply(weights_init_)

    def forward(self, state_goal):
        """Compute mean and log_std"""
        x = self.net(state_goal)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state_goal, deterministic=False):
        """
        Sample action from policy
        Returns: action, log_prob, mean_action
        """
        mean, log_std = self.forward(state_goal)
        std = log_std.exp()
        normal = Normal(mean, std)

        if deterministic:
            x_t = mean
        else:
            x_t = normal.rsample()  # Reparameterization trick

        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        if deterministic:
            log_prob = None
        else:
            log_prob = normal.log_prob(x_t)
            # Correction for tanh squashing
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + EPS)
            log_prob = log_prob.sum(1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)