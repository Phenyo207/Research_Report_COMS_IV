# model.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        nn.init.constant_(m.bias, 0.0)

# -------------------------------------------------
# OPTIONAL: Fourier feature mapping for spatial inputs
# Can improve generalization but adds complexity
# Set use_fourier=False to disable for simpler baseline
# -------------------------------------------------
class FourierEncoder(nn.Module):
    """
    Maps R^d → R^{2*mapping_size} with sin/cos(Bx).
    Improves spatial generalization but optional.
    """
    def __init__(self, in_dim: int, mapping_size: int = 64, scale: float = 10.0):
        super().__init__()
        B = torch.randn(in_dim, mapping_size) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2.0 * torch.pi * (x @ self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# -------------------------------------------------
# Encoders with LayerNorm (works with batch_size=1)
# -------------------------------------------------
class SharedEmbeddingEncoder(nn.Module):
    """
    State/Goal encoder with optional Fourier features → MLP(+LayerNorm).
    LayerNorm allows single-sample inference (no batch size restriction).
    
    For stability testing, try use_fourier=False first.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        use_fourier: bool = False,  # CHANGED: Default False for stability
        fourier_size: int = 64,
        fourier_scale: float = 10.0,
    ):
        super().__init__()
        self.use_fourier = use_fourier
        
        if use_fourier:
            fourier_out = fourier_size * 2
            self.fourier = FourierEncoder(input_dim, mapping_size=fourier_size, scale=fourier_scale)
        else:
            fourier_out = input_dim
            self.fourier = nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(fourier_out, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.apply(weights_init_)

    def forward(self, x):
        x = x.float()
        x = self.fourier(x)
        return self.net(x)

# -------------------------------------------------
# WVF Critic (Q1, Q2 heads only - V head removed)
# -------------------------------------------------
class WVFQCritic(nn.Module):
    """
    FIXED: Removed V-head since we use Q-targets in SAC.
    Keeps dual Q-networks for stability.
    """
    def __init__(self, num_inputs, num_actions, goal_dim, hidden_dim=256,
                 checkpoint_dir='checkpoints', name='wvf_q_critic',
                 use_fourier=False):  # ADDED: Configurable Fourier features
        super().__init__()
        self.observation_dim = num_inputs
        self.goal_dim = goal_dim
        self.num_actions = num_actions
        self.embedding_dim = hidden_dim

        # Separate encoders for state and goal
        self.state_encoder = SharedEmbeddingEncoder(num_inputs, hidden_dim, use_fourier=use_fourier)
        self.goal_encoder  = SharedEmbeddingEncoder(goal_dim, hidden_dim, use_fourier=use_fourier)

        # Dual Q-networks
        q_in = hidden_dim * 2 + num_actions
        self.q1 = nn.Sequential(
            nn.Linear(q_in, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(q_in, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # REMOVED: V-head (not used in corrected SAC implementation)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.apply(weights_init_)

    def encode_state_goal(self, state_goal):
        """Encode state and goal separately, then concatenate"""
        s = state_goal[:, :self.observation_dim]
        g = state_goal[:, self.observation_dim:]
        s_emb = self.state_encoder(s)
        g_emb = self.goal_encoder(g)
        return s_emb, g_emb

    def forward(self, state_goal, action):
        """
        FIXED: Only compute Q-values (no V-head).
        Returns: (q1, q2) for standard SAC update
        """
        s_emb, g_emb = self.encode_state_goal(state_goal)
        q_in = torch.cat([s_emb, g_emb, action], dim=-1)
        return self.q1(q_in), self.q2(q_in)

    def get_q_values_for_goal_policy(self, state_goal, action):
        """Get conservative Q-value estimate for goal selection"""
        with torch.no_grad():
            q1, q2 = self.forward(state_goal, action)
            return torch.min(q1, q2)

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location="cpu"))

# -------------------------------------------------
# Policy
# -------------------------------------------------
class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, goal_dim, hidden_dim=256,
                 action_space=None, checkpoint_dir='checkpoints', 
                 name="wvf_actor_network", use_fourier=False):  # ADDED: Configurable Fourier
        super().__init__()
        self.observation_dim = num_inputs
        self.goal_dim = goal_dim

        # Separate encoders with optional Fourier features
        self.state_encoder = SharedEmbeddingEncoder(num_inputs, hidden_dim, use_fourier=use_fourier)
        self.goal_encoder  = SharedEmbeddingEncoder(goal_dim, hidden_dim, use_fourier=use_fourier)

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

        # Output heads for continuous action distribution
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.apply(weights_init_)

        # Action space scaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.as_tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
            self.action_bias  = torch.as_tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)

    def encode_state_goal(self, state_goal):
        """Encode state and goal separately"""
        s = state_goal[:, :self.observation_dim]
        g = state_goal[:, self.observation_dim:]
        s_emb = self.state_encoder(s)
        g_emb = self.goal_encoder(g)
        return s_emb, g_emb

    def forward(self, state_goal):
        """Compute mean and log_std for policy distribution"""
        s_emb, g_emb = self.encode_state_goal(state_goal)
        x = self.policy_net(torch.cat([s_emb, g_emb], dim=-1))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x).clamp(LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state_goal, deterministic=False):
        """Sample actions from policy with reparameterization trick"""
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
            # Enforce action bounds (correction for tanh squashing)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + EPS)
            log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location="cpu"))

# -------------------------------------------------
# ALTERNATIVE: Simplified concatenated architecture
# Use this if separate encoders + Fourier cause issues
# -------------------------------------------------
class SimpleCritic(nn.Module):
    """
    Simpler baseline: concatenate state+goal directly (like codeset1).
    Use if separate encoders prove unstable.
    """
    def __init__(self, num_inputs, num_actions, goal_dim, hidden_dim=256):
        super().__init__()
        input_dim = num_inputs + goal_dim + num_actions
        
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(weights_init_)
    
    def forward(self, state_goal, action):
        x = torch.cat([state_goal, action], dim=-1)
        return self.q1(x), self.q2(x)

class SimpleActor(nn.Module):
    """
    Simpler baseline policy: concatenate state+goal directly.
    """
    def __init__(self, num_inputs, num_actions, goal_dim, hidden_dim=256, action_space=None):
        super().__init__()
        input_dim = num_inputs + goal_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, num_actions)
        self.log_std = nn.Linear(hidden_dim, num_actions)
        
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.as_tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
            self.action_bias  = torch.as_tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        
        self.apply(weights_init_)
    
    def forward(self, state_goal):
        x = self.net(state_goal)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std
    
    def sample(self, state_goal, deterministic=False):
        mean, log_std = self.forward(state_goal)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        if deterministic:
            x_t = mean
        else:
            x_t = normal.rsample()
        
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        if deterministic:
            log_prob = None
        else:
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + EPS)
            log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)