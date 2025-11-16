# model.py
# ============================================================================
# Neural Network Models for SAC
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6


def weights_init_(m):
    """Initialize network weights with Xavier uniform."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        nn.init.constant_(m.bias, 0.0)


class Critic(nn.Module):
    """
    Twin Q-network critic for SAC.
    Takes concatenated [state | goal | action] as input.
    """
    
    def __init__(self, state_dim, action_dim, goal_dim, hidden_dim=256):
        """
        Initialize critic networks.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            goal_dim: Dimension of goal space
            hidden_dim: Hidden layer size
        """
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
        """
        Compute Q-values for given state-goal and action.
        
        Args:
            state_goal: Concatenated [state | goal] tensor
            action: Action tensor
            
        Returns:
            Tuple of (q1_value, q2_value)
        """
        x = torch.cat([state_goal, action], dim=-1)
        return self.q1(x), self.q2(x)


class Actor(nn.Module):
    """
    Gaussian policy network for SAC with tanh squashing.
    Takes concatenated [state | goal] as input.
    """
    
    def __init__(self, state_dim, action_dim, goal_dim, action_space=None, hidden_dim=256):
        """
        Initialize actor network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            goal_dim: Dimension of goal space
            action_space: Gym action space (for scaling)
            hidden_dim: Hidden layer size
        """
        super().__init__()
        
        input_dim = state_dim + goal_dim
        
        # Shared feature extraction
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output heads
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
        self.apply(weights_init_)
        
        # Action scaling for bounded action spaces
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

    def forward(self, state_goal):
        """
        Compute mean and log_std for action distribution.
        
        Args:
            state_goal: Concatenated [state | goal] tensor
            
        Returns:
            Tuple of (mean, log_std)
        """
        x = F.relu(self.fc1(state_goal))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_linear(x)
        log_std = torch.clamp(self.log_std_linear(x), LOG_SIG_MIN, LOG_SIG_MAX)
        
        return mean, log_std

    def sample(self, state_goal, deterministic=False):
        """
        Sample actions from the policy.
        
        Args:
            state_goal: Concatenated [state | goal] tensor
            deterministic: If True, return mean action (no noise)
            
        Returns:
            Tuple of (action, log_prob, mean_action)
        """
        mean, log_std = self.forward(state_goal)
        std = log_std.exp()
        
        # Create normal distribution
        normal = Normal(mean, std)
        
        if deterministic:
            # Deterministic action (for evaluation)
            z = mean
        else:
            # Sample with reparameterization trick
            z = normal.rsample()
        
        # Apply tanh squashing
        y = torch.tanh(z)
        action = y * self.action_scale + self.action_bias
        
        if deterministic:
            # No log_prob needed for deterministic actions
            log_prob = None
        else:
            # Compute log probability with correction for tanh squashing
            log_prob = normal.log_prob(z)
            # Correction term for tanh
            log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + EPS)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        
        # Return mean action for reference
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean_action

    def to(self, device):
        """Move model and action scaling tensors to device."""
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)