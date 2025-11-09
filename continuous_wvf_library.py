import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class LangevinGoalSampler:
    def __init__(self, goal_dim, step_size=0.01, num_steps=20, noise_scale=0.05, 
                 goal_bounds=(-3.0, 3.0), device='cpu'):
        self.goal_dim = goal_dim
        self.step_size = step_size
        self.num_steps = num_steps
        self.noise_scale = noise_scale
        self.goal_bounds = goal_bounds
        self.device = device
        
    def sample_goals(self, state, critic, policy, n_goals=5, init_goals=None):
        """
        Sample goals using Langevin dynamics with Q-values as energy.
        
        Args:
            state: Current state (numpy array)
            critic: Critic network (provides Q-values)
            policy: Policy network (provides actions for states+goals)
            n_goals: Number of goals to sample
            init_goals: Optional initial goal proposals (numpy array)
        
        Returns:
            List of sampled goals (numpy arrays)
        """
        # Initialize goals
        if init_goals is not None:
            goals = torch.tensor(init_goals, dtype=torch.float32, device=self.device, requires_grad=True)
        else:
            # Random initialization within bounds
            goals = torch.rand(n_goals, self.goal_dim, device=self.device) * \
                    (self.goal_bounds[1] - self.goal_bounds[0]) + self.goal_bounds[0]
            goals.requires_grad = True
        
        # Prepare state tensor
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        state_batch = state_tensor.repeat(n_goals, 1)
        
        # Langevin dynamics
        for step in range(self.num_steps):
            # Compute Q-values (energy) for current goals
            state_goal = torch.cat([state_batch, goals], dim=-1)
            
            with torch.enable_grad():
                # Get actions from policy
                _, _, actions = policy.sample(state_goal, deterministic=True)
                
                # Get Q-values (use minimum of Q1, Q2 for conservative estimate)
                q1, q2 = critic(state_goal, actions)
                q_values = torch.min(q1, q2).squeeze(-1)
                
                # Energy = -Q (we want to maximize Q, minimize -Q)
                energy = -q_values.sum()
                
                # Compute gradient
                grad = torch.autograd.grad(energy, goals, create_graph=False)[0]
            
            # Langevin update with gradient ascent (maximize Q)
            noise = torch.randn_like(goals) * self.noise_scale
            goals = goals - self.step_size * grad + np.sqrt(2 * self.step_size) * noise
            
            # Clip to bounds
            goals = torch.clamp(goals, self.goal_bounds[0], self.goal_bounds[1])
            goals = goals.detach().requires_grad_(True)
        
        # Return as numpy arrays
        return goals.detach().cpu().numpy()
    
    def sample_single_goal(self, state, critic, policy, init_goal=None):
        """Convenience method to sample a single goal"""
        if init_goal is not None:
            init_goals = init_goal.reshape(1, -1)
        else:
            init_goals = None
        
        goals = self.sample_goals(state, critic, policy, n_goals=1, init_goals=init_goals)
        return goals[0]


class ContinuousGoalOrientedAgent:

    def __init__(self, state_dim, goal_dim, action_space,
                 hidden_dims=[128, 128], learning_rate=3e-4, gamma=0.99, tau=0.005, device='cpu'):
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_space = action_space
        self.device = device

        # Predefine a coarse goal grid for curriculum sampling
        xs = np.linspace(-2.0, 2.0, 21)
        ys = np.linspace(-2.0, 2.0, 21)
        grid = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
        self.goal_space = grid.astype(np.float32)
        
        # Initialize Langevin sampler
        self.langevin_sampler = LangevinGoalSampler(
            goal_dim=goal_dim,
            step_size=0.02,  # Larger steps for faster exploration
            num_steps=15,    # Moderate number of steps
            noise_scale=0.08, # Sufficient noise for exploration
            goal_bounds=(-2.5, 2.5),
            device=device
        )

    def sample_goal_curriculum(self, agent_pos, mastery_score):
        """
        Distance-based curriculum: near goals at low mastery, farther as mastery grows.
        """
        agent_pos = np.array(agent_pos, dtype=np.float32).reshape(1, -1)
        dists = np.linalg.norm(self.goal_space - agent_pos, axis=1)
        # Max reachable distance grows with mastery
        max_dist = float(np.clip(0.4 + mastery_score * 1.6, 0.4, 2.5))
        mask = dists < max_dist
        candidates = self.goal_space[mask] if np.any(mask) else self.goal_space
        return candidates[np.random.randint(len(candidates))]

    def sample_goals_from_ebm(self, n_goals=3, state=None, critic=None, policy=None):
        """
        Sample goals using Langevin dynamics if critic/policy available,
        otherwise fall back to random sampling.
        
        Args:
            n_goals: Number of goals to sample
            state: Current state (for Langevin sampling)
            critic: Critic network (for Langevin sampling)
            policy: Policy network (for Langevin sampling)
        
        Returns:
            List of sampled goals
        """
        if state is not None and critic is not None and policy is not None:
            # Use Langevin sampling with Q-values as energy
            try:
                # Initialize from diverse locations
                init_goals = np.random.uniform(-2.0, 2.0, size=(n_goals, self.goal_dim)).astype(np.float32)
                sampled_goals = self.langevin_sampler.sample_goals(
                    state, critic, policy, n_goals=n_goals, init_goals=init_goals
                )
                return [g for g in sampled_goals]
            except Exception as e:
                print(f"[Langevin] Sampling failed: {e}, falling back to random")
        
        # Fallback: random sampling
        return [np.random.uniform(-2.0, 2.0, size=self.goal_dim).astype(np.float32) 
                for _ in range(n_goals)]
    
    def refine_goal_with_langevin(self, state, initial_goal, critic, policy, num_steps=10):
        """
        Refine a single goal using Langevin dynamics.
        Useful for improving suboptimal goal proposals.
        """
        sampler = LangevinGoalSampler(
            goal_dim=self.goal_dim,
            step_size=0.01,
            num_steps=num_steps,
            noise_scale=0.05,
            goal_bounds=(-2.5, 2.5),
            device=self.device
        )
        
        return sampler.sample_single_goal(state, critic, policy, init_goal=initial_goal)