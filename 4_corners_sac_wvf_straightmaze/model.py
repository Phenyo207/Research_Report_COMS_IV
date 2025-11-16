# model.py — WVF Critic + Actor (stabilized, normalized, consistent with new agent/main)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6


# ============================================================
# Helpers
# ============================================================
def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        nn.init.constant_(m.bias, 0.0)


# ============================================================
# Fourier feature encoder (optional)
# ============================================================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class FourierEncoder(nn.Module):
    def __init__(self, in_dim: int, mapping_size: int = 64, scale: float = 10.0):
        super().__init__()
        B = torch.randn(in_dim, mapping_size) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2.0 * torch.pi * (x @ self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# ============================================================
# Encoders
# ============================================================
class SharedEmbeddingEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        use_fourier: bool = False,
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


# ============================================================
# WVF Critic (stabilized)
# ============================================================
class WVFQCritic(nn.Module):
    """
    Dual Q networks for WVF. Handles [action + done_logit].
    Includes output normalization and clamping to prevent divergence.
    """
    def __init__(self, num_inputs, num_actions, goal_dim, hidden_dim=256,
                 checkpoint_dir='checkpoints', name='wvf_q_critic',
                 use_fourier=False):
        super().__init__()
        self.observation_dim = num_inputs
        self.goal_dim = goal_dim
        self.num_actions = num_actions
        self.embedding_dim = hidden_dim

        self.state_encoder = SharedEmbeddingEncoder(num_inputs, hidden_dim, use_fourier=use_fourier)
        self.goal_encoder  = SharedEmbeddingEncoder(goal_dim, hidden_dim, use_fourier=use_fourier)

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

        self.output_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.output_bias = nn.Parameter(torch.tensor(0.0), requires_grad=False)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.apply(weights_init_)

    def encode_state_goal(self, state_goal):
        s = state_goal[:, :self.observation_dim]
        g = state_goal[:, self.observation_dim:]
        s_emb = self.state_encoder(s)
        g_emb = self.goal_encoder(g)
        return s_emb, g_emb

    def forward(self, state_goal, action):
        s_emb, g_emb = self.encode_state_goal(state_goal)
        q_in = torch.cat([s_emb, g_emb, action], dim=-1)
        q1 = self.q1(q_in)
        q2 = self.q2(q_in)

        # === Normalization / clamping to prevent explosion ===
        q1 = torch.tanh(q1 / 10.0) * 10.0  # keep roughly in [-10,10]
        q2 = torch.tanh(q2 / 10.0) * 10.0
        return q1, q2

    def get_q_values_for_goal_policy(self, state_goal, action):
        with torch.no_grad():
            q1, q2 = self.forward(state_goal, action)
            return torch.min(q1, q2)

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location="cpu"))


# ============================================================
# WVF Actor (policy)
# ============================================================
class Actor(nn.Module):
    """
    WVF Actor outputs continuous action + done logit.
    The last output dimension is the 'done' gate (unsquashed).
    """
    def __init__(self, num_inputs, num_actions, goal_dim, hidden_dim=256,
                 action_space=None, checkpoint_dir='checkpoints',
                 name="wvf_actor_network", use_fourier=False):
        super().__init__()
        self.observation_dim = num_inputs
        self.goal_dim = goal_dim
        self.num_actions = num_actions  # includes done dim
        self.action_dim = num_actions

        self.state_encoder = SharedEmbeddingEncoder(num_inputs, hidden_dim, use_fourier=use_fourier)
        self.goal_encoder  = SharedEmbeddingEncoder(goal_dim, hidden_dim, use_fourier=use_fourier)

        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.apply(weights_init_)

        # Action scaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.as_tensor(
                (action_space.high - action_space.low) / 2.0, dtype=torch.float32)
            self.action_bias = torch.as_tensor(
                (action_space.high + action_space.low) / 2.0, dtype=torch.float32)

    def encode_state_goal(self, state_goal):
        s = state_goal[:, :self.observation_dim]
        g = state_goal[:, self.observation_dim:]
        s_emb = self.state_encoder(s)
        g_emb = self.goal_encoder(g)
        return s_emb, g_emb

    def forward(self, state_goal):
        s_emb, g_emb = self.encode_state_goal(state_goal)
        x = self.policy_net(torch.cat([s_emb, g_emb], dim=-1))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x).clamp(LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state_goal, deterministic=False):
        mean, log_std = self.forward(state_goal)
        std = log_std.exp()
        normal = Normal(mean, std)

        if deterministic:
            x_t = mean
        else:
            x_t = normal.rsample()

        # Apply tanh only to movement dims
        y_t = torch.tanh(x_t)
        move_part = y_t[:, :-1]
        done_logit = x_t[:, -1:].clamp(-1.0, 1.0)  # stabilized

        # Scale only movement part
        action_move = move_part * self.action_scale[:-1] + self.action_bias[:-1]
        action = torch.cat([action_move, done_logit], dim=-1)

        # --- log prob (movement only) ---
        if deterministic:
            total_log_prob = None
        else:
            mean_m, std_m = mean[:, :-1], std[:, :-1]
            x_m = x_t[:, :-1]
            y_m = y_t[:, :-1]
            normal_m = Normal(mean_m, std_m)
            log_prob_move = normal_m.log_prob(x_m)
            log_prob_move -= torch.log(self.action_scale[:-1] * (1 - y_m.pow(2)) + EPS)
            log_prob_move = log_prob_move.sum(1, keepdim=True)

            done_normal = Normal(mean[:, -1:], std[:, -1:])
            log_prob_done = done_normal.log_prob(x_t[:, -1:]).sum(1, keepdim=True)
            total_log_prob = log_prob_move + log_prob_done

        mean_act = torch.tanh(mean[:, :-1]) * self.action_scale[:-1] + self.action_bias[:-1]
        mean_out = torch.cat([mean_act, mean[:, -1:].clone().clamp(-1.0, 1.0)], dim=-1)

        return action, total_log_prob, mean_out

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location="cpu"))
