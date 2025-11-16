import numpy as np
import torch
import torch.nn as nn

# ============================================================
# Simple continuous WVF helpers + EBM/Langevin goal sampling
# ============================================================

class ContinuousWVFNetwork(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, hidden_dims=[128, 128]):
        super().__init__()
        inp = state_dim + goal_dim + action_dim
        layers = []
        d = inp
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, s, g, a):
        return self.net(torch.cat([s, g, a], dim=-1))


class ContinuousWVFPolicy(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, hidden_dims=[128, 128], action_space=None):
        super().__init__()
        inp = state_dim + goal_dim
        layers, d = [], inp
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.feat = nn.Sequential(*layers)
        self.mean = nn.Linear(d, action_dim)
        self.std  = nn.Linear(d, action_dim)

        if action_space is not None:
            self.action_scale = torch.as_tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
            self.action_bias  = torch.as_tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        else:
            self.action_scale = torch.ones(action_dim)
            self.action_bias  = torch.zeros(action_dim)

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, s, g):
        x = self.feat(torch.cat([s, g], dim=-1))
        mean = self.mean(x)
        log_std = self.std(x).clamp(-20, 2)
        return mean, log_std

    def sample(self, s, g, deterministic=False):
        mean, log_std = self.forward(s, g)
        if deterministic:
            a = torch.tanh(mean) * self.action_scale + self.action_bias
            return a, None, a
        std = log_std.exp()
        n = torch.distributions.Normal(mean, std)
        x = n.rsample()
        y = torch.tanh(x)
        a = y * self.action_scale + self.action_bias
        logp = n.log_prob(x)
        logp -= torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        return a, logp.sum(-1, keepdim=True), torch.tanh(mean) * self.action_scale + self.action_bias


# ============================================================
# Langevin goal sampler (EBM on −Q)
# ============================================================
class LangevinGoalSampler:
    """
    Langevin dynamics for sampling goals from an energy-based model:
    E(g) = -Q(s, π(s,g), g)
    """
    def __init__(self, goal_dim, step_size=0.02, num_steps=15, noise_scale=0.08,
                 goal_bounds=(-2.5, 2.5), device='cpu', temperature_start=1.0, temperature_end=0.3):
        self.goal_dim = goal_dim
        self.step_size = step_size
        self.num_steps = num_steps
        self.noise_scale = noise_scale
        self.goal_bounds = goal_bounds
        self.device = device
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end

    def _temp(self, t):
        # simple cosine anneal across steps
        if self.num_steps <= 1:
            return self.temperature_end
        cos = 0.5 * (1 + np.cos(np.pi * t / (self.num_steps - 1)))
        return self.temperature_end + (self.temperature_start - self.temperature_end) * cos

    def sample_goals(self, state, critic, policy, n_goals=5, init_goals=None):
        # Init
        if init_goals is not None:
            goals = torch.tensor(init_goals, dtype=torch.float32, device=self.device, requires_grad=True)
        else:
            goals = torch.rand(n_goals, self.goal_dim, device=self.device)
            lo, hi = self.goal_bounds
            goals = (hi - lo) * goals + lo
            goals.requires_grad = True

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        state_batch = state_tensor.repeat(n_goals, 1)

        for step in range(self.num_steps):
            temp = self._temp(step)
            with torch.enable_grad():
                sg = torch.cat([state_batch, goals], dim=-1)
                _, _, actions = policy.sample(sg, deterministic=True)
                q1, q2 = critic(sg, actions)
                q_values = torch.min(q1, q2).squeeze(-1)
                energy = -(q_values / max(temp, 1e-3)).sum()
                grad = torch.autograd.grad(energy, goals, create_graph=False)[0]

            noise = torch.randn_like(goals) * self.noise_scale
            goals = goals - self.step_size * grad + np.sqrt(2 * self.step_size) * noise
            goals = torch.clamp(goals, self.goal_bounds[0], self.goal_bounds[1])
            goals = goals.detach().requires_grad_(True)

        return goals.detach().cpu().numpy()

    def sample_single_goal(self, state, critic, policy, init_goal=None):
        if init_goal is not None:
            init_goals = init_goal.reshape(1, -1)
        else:
            init_goals = None
        goals = self.sample_goals(state, critic, policy, n_goals=1, init_goals=init_goals)
        return goals[0]


# ============================================================
# Goal agent: curriculum + EBM proposals
# ============================================================
class ContinuousGoalOrientedAgent:
    def __init__(self, state_dim, goal_dim, action_space,
                 hidden_dims=[128, 128], learning_rate=3e-4, gamma=0.99, tau=0.005, device='cpu'):
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_space = action_space
        self.device = device

        # Coarse grid (for curriculum)
        xs = np.linspace(-2.0, 2.0, 21)
        ys = np.linspace(-2.0, 2.0, 21)
        grid = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
        self.goal_space = grid.astype(np.float32)

        # Langevin sampler (with temperature anneal)
        self.langevin_sampler = LangevinGoalSampler(
            goal_dim=goal_dim,
            step_size=0.02,
            num_steps=15,
            noise_scale=0.08,
            goal_bounds=(-2.5, 2.5),
            device=device,
            temperature_start=1.0,
            temperature_end=0.3
        )

    def sample_goal_curriculum(self, agent_pos, mastery_score, temperature=0.5):
        """
        Adaptive curriculum: soft-select goals by distance with temperature and a
        mastery-controlled reach radius.
        """
        agent_pos = np.array(agent_pos, dtype=np.float32).reshape(1, -1)
        dists = np.linalg.norm(self.goal_space - agent_pos, axis=1)
        max_dist = float(np.clip(0.4 + mastery_score * 2.0, 0.4, 2.5))
        probs = np.exp(-dists / (temperature + 1e-6))
        probs[dists > max_dist] = 0.0
        if probs.sum() <= 0:
            probs = np.ones_like(dists)
        probs = probs / probs.sum()
        return self.goal_space[np.random.choice(len(self.goal_space), p=probs)]

    def sample_goals_from_ebm(self, n_goals=3, state=None, critic=None, policy=None):
        if state is not None and critic is not None and policy is not None:
            try:
                init_goals = np.random.uniform(-2.0, 2.0, size=(n_goals, self.goal_dim)).astype(np.float32)
                sampled_goals = self.langevin_sampler.sample_goals(
                    state, critic, policy, n_goals=n_goals, init_goals=init_goals
                )
                return [g for g in sampled_goals]
            except Exception as e:
                print(f"[Langevin] Sampling failed: {e}, falling back to random")
        return [np.random.uniform(-2.0, 2.0, size=self.goal_dim).astype(np.float32)
                for _ in range(n_goals)]

    def refine_goal_with_langevin(self, state, initial_goal, critic, policy, num_steps=10):
        sampler = LangevinGoalSampler(
            goal_dim=self.goal_dim,
            step_size=0.01,
            num_steps=num_steps,
            noise_scale=0.05,
            goal_bounds=(-2.5, 2.5),
            device=self.device
        )
        return sampler.sample_single_goal(state, critic, policy, init_goal=initial_goal)


# ============================================================
# Logical WVF composition (continuous proxy on Q-values)
# ============================================================
def wvf_max(wvf1, wvf2): return torch.maximum(wvf1, wvf2)
def wvf_min(wvf1, wvf2): return torch.minimum(wvf1, wvf2)
def wvf_and(wvf1, wvf2): return torch.minimum(wvf1, wvf2)
def wvf_or(wvf1, wvf2):  return torch.maximum(wvf1, wvf2)
def wvf_not(wvf):        return -wvf


# ============================================================
# Diagnostics: WVF-based reward reconstruction (optional)
# ============================================================
@torch.no_grad()
def reconstruct_reward(critic, state, goal, action, next_state, gamma=0.99):
    """
    r_hat ≈ Q(s,g,a) - γ * Q(s',g,π(s',g))  (using min(Q1,Q2))
    Useful for sanity-checking shaping magnitudes.
    """
    sg = torch.cat([state, goal], dim=-1)
    q1, q2 = critic(sg, action)
    q = torch.min(q1, q2)

    # Greedy next action from policy is typically used; here, caller can supply it if needed.
    # This function expects 'action' already consistent; for full correctness, pass policy.sample on next_state.
    return q  # return raw q proxy; caller can subtract next state's q if desired
