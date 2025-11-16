# agent.py — Continuous WVF with Dyna imagination, adaptive curriculum, and dense shaping

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from model import WVFQCritic, Actor
from buffer import ReplayBuffer
from continuous_wvf_library import (
    ContinuousGoalOrientedAgent,
    wvf_and, wvf_or, wvf_min, wvf_max, wvf_not,
)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def hard_update(target, source):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(s.data)

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


# -------------------------------------------------
# Agent
# -------------------------------------------------
class Agent(object):
    def __init__(self, num_inputs, action_space, goal_dim, gamma, tau, alpha,
                 target_update_interval, hidden_size, learning_rate,
                 exploration_scaling_factor, her_strategy='future', her_k=4,
                 use_langevin=True, corners=None,
                 normalize_inputs=False, clip_rewards=False):
        # Core cfg
        self.gamma = gamma
        self.tau = tau
        self.goal_dim = goal_dim
        self.her_strategy = her_strategy
        self.her_k = her_k
        self.observation_dim = num_inputs
        self.target_update_interval = target_update_interval
        self.use_langevin = use_langevin
        self.corners = corners
        self.normalize_inputs = normalize_inputs
        self.clip_rewards = clip_rewards

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch = torch  # for heatmap helper
        print(f"[Agent] device = {self.device}")
        print(f"[Agent] Langevin sampling = {self.use_langevin}")

        # Networks
        self.critic = WVFQCritic(num_inputs, action_space.shape[0], goal_dim, hidden_size).to(self.device)
        self.critic_target = WVFQCritic(num_inputs, action_space.shape[0], goal_dim, hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.critic_optim = Adam(self.critic.parameters(), lr=learning_rate, weight_decay=1e-4)

        self.policy = Actor(num_inputs, action_space.shape[0], goal_dim, hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)

        # Entropy temperature
        self.target_entropy = -0.5 * float(action_space.shape[0])
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=learning_rate)
        self.alpha = float(alpha)

        # WVF reward parameters (WVF “done” semantics kept)
        self.R_MIN = -10.0
        self.goal_achievement_reward = 1.0
        self.goal_achievement_threshold = 0.5

        # Dense shaping (annealed): r_dense = k(t) * (d_prev - d_next)
        self.use_dense_progress = True
        self.dense_coef_start = 0.10     # initial scaling of dense shaping
        self.dense_coef_end   = 0.02     # final scaling after anneal
        self.dense_anneal_steps = 200_000

        # Dyna imagination schedule (safe defaults)
        self.use_dyna = True
        self.dyna_start_steps = 25_000
        self.dyna_rollout_steps = 3
        self.dyna_samples = 64
        self.dyna_eta = 0.01   # step size on ∇_s (-Q)
        self.dyna_noise = 0.01

        # Goal management
        self.internal_goals = set()
        self.successful_goals = set()
        self.goal_success_counts = {}
        self.current_goal_type = 'environment'
        self.current_chosen_goal = None

        # Stats
        self.done_action_count = 0
        self.total_action_count = 0
        self.learn_step = 0

        # Mastery tracking
        self.mastery_history = []

        # Helper for curriculum & EBM/Langevin
        self.continuous_wvf = ContinuousGoalOrientedAgent(
            state_dim=num_inputs, goal_dim=goal_dim, action_space=action_space,
            hidden_dims=[128, 128], learning_rate=learning_rate,
            gamma=gamma, tau=tau, device=self.device
        )

        self.langevin_goal_count = 0
        self.langevin_success_count = 0

    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------
    def to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float32, device=self.device)
        elif isinstance(x, torch.Tensor):
            return x.to(self.device)
        else:
            return torch.tensor(x, dtype=torch.float32, device=self.device)

    def goal_distance(self, achieved_goal, desired_goal):
        return np.linalg.norm(achieved_goal - desired_goal, axis=-1)

    def is_goal_achieved(self, achieved_goal, desired_goal, distance_threshold=None):
        thr = self.goal_achievement_threshold if distance_threshold is None else distance_threshold
        return self.goal_distance(achieved_goal, desired_goal) <= thr

    def set_current_goal(self, goal, goal_type='environment'):
        self.current_chosen_goal = goal.copy() if isinstance(goal, np.ndarray) else np.array(goal)
        self.current_goal_type = goal_type

    def update_internal_goals(self, state, achieved_goal, info):
        if info is not None and info.get('agent_chose_done', False):
            goal_tuple = tuple(np.round(achieved_goal, 2))
            if goal_tuple not in self.internal_goals:
                self.internal_goals.add(goal_tuple)
            self.done_action_count += 1

    # -------------------------------------------------
    # Goal policy (GPI-flavored)
    # -------------------------------------------------
    def goal_policy(self, state, goals, epsilon=0.1):
        # Try EBM/Langevin proposals sometimes
        if self.use_langevin and (not goals or np.random.rand() < epsilon):
            try:
                extra = self.continuous_wvf.sample_goals_from_ebm(
                    n_goals=5, state=state, critic=self.critic, policy=self.policy
                )
                if extra:
                    selected_goal = extra[np.random.randint(len(extra))]
                    self.langevin_goal_count += 1
                    return selected_goal, 'langevin'
            except Exception:
                pass

        # Value-aware selection across internal goals (GPI-like)
        if goals:
            max_goals = 64
            if len(goals) > max_goals:
                idx = np.random.choice(len(goals), max_goals, replace=False)
                goals_eval = [goals[i] for i in idx]
            else:
                goals_eval = goals

            q_vals = []
            batch = 32
            for b in range(0, len(goals_eval), batch):
                chunk = goals_eval[b:b+batch]
                states = np.tile(state, (len(chunk), 1))
                sg = np.concatenate([states, np.array(chunk)], axis=1)
                sg_t = self.to_tensor(sg)
                with torch.no_grad():
                    _, _, a = self.policy.sample(sg_t, deterministic=True)
                    q1, q2 = self.critic(sg_t, a)
                    q_vals.extend(torch.min(q1, q2).cpu().numpy().ravel())
            return goals_eval[int(np.argmax(q_vals))], 'existing'

        # Fallback random
        return np.random.uniform(-2.0, 2.0, size=self.goal_dim).astype(np.float32), 'random'

    # -------------------------------------------------
    # Action sampling (continuous move + done)
    # -------------------------------------------------
    def split_move_and_done(self, action_tensor, original_act_dim: int):
        if action_tensor.shape[-1] == original_act_dim:
            return action_tensor, torch.zeros_like(action_tensor[..., :1])
        move = action_tensor[..., :-1]
        done_gate = action_tensor[..., -1:]
        return move, done_gate

    def sample_action(self, sg_tensor, deterministic=False):
        with torch.no_grad():
            _, _, full_action = self.policy.sample(sg_tensor, deterministic=deterministic)
        move_action, done_logit = self.split_move_and_done(
            full_action, original_act_dim=self.policy.num_actions - 1
        )
        done_prob = torch.sigmoid(done_logit)

        # Mild early “done” exploration
        self.total_action_count += 1
        if not deterministic and self.total_action_count < 50_000:
            if np.random.rand() < 0.10:
                done_prob = torch.tensor([[1.0]], device=self.device)
        return move_action, done_prob

    def chose_done(self, done_prob, threshold=0.5):
        if isinstance(done_prob, torch.Tensor):
            done_prob = done_prob.item()
        return done_prob > threshold

    # -------------------------------------------------
    # Rewards (WVF + dense shaping)
    # -------------------------------------------------
    def _dense_coef(self):
        # Linear anneal dense shaping coefficient
        t = float(min(self.learn_step, self.dense_anneal_steps)) / max(1.0, float(self.dense_anneal_steps))
        return (1 - t) * self.dense_coef_start + t * self.dense_coef_end

    def compute_reward(self, state, next_state, goal, chose_done: bool, goal_reached: bool):
        """
        WVF reward:
        - +1.0 if chose 'done' AND goal_reached
        - -10.0 if chose 'done' but not reached
        Dense shaping (optional): +k * (||s-goal||_prev - ||s'-goal||)
        """
        if chose_done:
            return self.goal_achievement_reward if goal_reached else self.R_MIN

        if self.use_dense_progress:
            d_prev = float(np.linalg.norm(state[:self.goal_dim] - goal))
            d_next = float(np.linalg.norm(next_state[:self.goal_dim] - goal))
            return self._dense_coef() * (d_prev - d_next)

        # Fallback (not used in this improved setup)
        return 0.0

    # -------------------------------------------------
    # Learning (SAC-style on WVF)
    # -------------------------------------------------
    def learn(self, memory: ReplayBuffer, batch_size):
        if not memory.can_sample(batch_size):
            return None

        sg, actions, rewards, next_sg, dones = memory.sample_buffer_balanced(batch_size)
        sg = torch.tensor(sg, dtype=torch.float32, device=self.device)
        next_sg = torch.tensor(next_sg, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        if self.clip_rewards:
            rewards = torch.clamp(rewards, -10.0, 10.0)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        # Critic
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.policy.sample(next_sg, deterministic=False)
            q1_next, q2_next = self.critic_target(next_sg, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            target_q = rewards + (1 - dones) * self.gamma * q_next
            target_q = torch.clamp(target_q, -5.0, 5.0)  # keep bounded targets

        q1, q2 = self.critic(sg, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()

        # Policy
        pi, log_pi, _ = self.policy.sample(sg, deterministic=False)
        q1_pi, q2_pi = self.critic(sg, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * log_pi - q_pi).mean()
        self.policy_optim.zero_grad(set_to_none=True)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optim.step()

        # Alpha
        alpha_loss = (self.log_alpha * (-log_pi.detach() - self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = float(self.log_alpha.exp().clamp(0.01, 0.50).item())

        # Target update
        soft_update(self.critic_target, self.critic, self.tau)
        self.learn_step += 1

        return dict(
            critic_loss=float(critic_loss.item()),
            policy_loss=float(policy_loss.item()),
            alpha=float(self.alpha),
            alpha_loss=float(alpha_loss.item())
        )

    # -------------------------------------------------
    # Dyna-style imagination (safe, gradient-based)
    # -------------------------------------------------
    def model_rollout(self, memory: ReplayBuffer, rollout_steps=None, num_samples=None):
        if not self.use_dyna or self.learn_step < self.dyna_start_steps:
            return
        rollout_steps = rollout_steps or self.dyna_rollout_steps
        num_samples = num_samples or self.dyna_samples

        # Use existing utility to grab random (state, action, goal, reward)
        samples = memory.sample_state_action_pairs(n_samples=num_samples)
        if samples is None:
            return

        states_np = samples['states']
        goals_np  = samples['goals']
        actions_np= samples['actions']

        # Convert to tensors, allow grad on states to ascend Q
        s = torch.tensor(states_np, dtype=torch.float32, device=self.device, requires_grad=True)
        g = torch.tensor(goals_np,  dtype=torch.float32, device=self.device)
        a = torch.tensor(actions_np, dtype=torch.float32, device=self.device)

        for _ in range(rollout_steps):
            sg = torch.cat([s, g], dim=-1)
            q1, q2 = self.critic(sg, a)
            energy = -torch.min(q1, q2).sum()  # maximize Q => descend energy

            grad_s = torch.autograd.grad(energy, s, create_graph=False, retain_graph=False)[0]
            s_next = s - self.dyna_eta * grad_s + self.dyna_noise * torch.randn_like(s)

            # Compute dense progress reward (no 'done' here)
            s_np = s.detach().cpu().numpy()
            s_next_np = s_next.detach().cpu().numpy()
            g_np = g.detach().cpu().numpy()
            # Only first goal_dim elements enter distance
            d_prev = np.linalg.norm(s_np[:, :self.goal_dim] - g_np, axis=1)
            d_next = np.linalg.norm(s_next_np[:, :self.goal_dim] - g_np, axis=1)
            r_dense = self._dense_coef() * (d_prev - d_next)

            # Store synthetic transition (done=False)
            for i in range(s.shape[0]):
                memory.add(
                    state=s_np[i], action=a.detach().cpu().numpy()[i],
                    reward=float(r_dense[i]), next_state=s_next_np[i],
                    goal=g_np[i], done=False, achieved_goal=None, goal_type='dyna'
                )
            s = s_next.detach().requires_grad_(True)  # continue rollouts

    # -------------------------------------------------
    # HER relabeling (vectorized + adaptive)
    # -------------------------------------------------
    def generate_her_transitions(self, episode_transitions, memory: ReplayBuffer, corners=None):
        if len(episode_transitions) == 0:
            return
        ep_len = len(episode_transitions)
        achieved_seq = np.array([tr[5] for tr in episode_transitions])

        # Base rate; can be tuned
        her_rate = 0.5

        corner_array = np.array(list(corners.values()), dtype=np.float32) if corners else None
        internal_array = np.array(list(self.internal_goals), dtype=np.float32) if len(self.internal_goals) > 0 else None

        for t in range(ep_len):
            if np.random.rand() >= her_rate:
                continue
            state, action, _, next_state, old_goal, achieved_goal, done_flag, info = episode_transitions[t]

            # Choose relabeled goal
            if np.random.rand() < 0.5:
                j = np.random.randint(t, ep_len)
                new_goal = achieved_seq[j]
                gtype = 'her_future'
            else:
                if corner_array is not None and np.random.rand() < 0.5:
                    dists = np.linalg.norm(corner_array - achieved_goal, axis=1)
                    new_goal = corner_array[np.argmin(dists)]
                    gtype = 'her_corner'
                elif internal_array is not None:
                    dists = np.linalg.norm(internal_array - achieved_goal, axis=1)
                    new_goal = internal_array[np.argmin(dists)]
                    gtype = 'her_internal'
                else:
                    j = np.random.randint(t, ep_len)
                    new_goal = achieved_seq[j]
                    gtype = 'her_fallback'

            # WVF + dense reward for relabeled goal
            goal_reached = self.is_goal_achieved(achieved_goal, new_goal)
            chose_done = bool(info.get('agent_chose_done', False))
            r_wvf = self.compute_reward(state, next_state, new_goal, chose_done, goal_reached)
            terminal = bool(done_flag or chose_done)

            memory.add(
                state=state, action=action, reward=r_wvf, next_state=next_state,
                goal=new_goal, done=terminal, achieved_goal=achieved_goal, goal_type=gtype
            )

    # -------------------------------------------------
    # Diagnostics
    # -------------------------------------------------
    def get_done_action_ratio(self):
        return self.done_action_count / max(1, self.total_action_count)

    def get_langevin_stats(self):
        return {
            'langevin_goals': self.langevin_goal_count,
            'langevin_success_rate': self.langevin_success_count / max(1, self.langevin_goal_count)
        }

    def update_goal_success(self, goal, success: bool):
        g = tuple(np.round(goal, 2))
        if success:
            self.successful_goals.add(g)
            self.goal_success_counts[g] = self.goal_success_counts.get(g, 0) + 1
            if self.current_goal_type == 'langevin':
                self.langevin_success_count += 1

    # -------------------------------------------------
    # Checkpointing
    # -------------------------------------------------
    def save_checkpoint(self, path="checkpoints"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pt"))
        torch.save(self.critic_target.state_dict(), os.path.join(path, "critic_target.pt"))
        torch.save(self.policy.state_dict(), os.path.join(path, "policy.pt"))

    def load_checkpoint(self, path="checkpoints"):
        try:
            self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pt"), map_location=self.device))
            self.critic_target.load_state_dict(torch.load(os.path.join(path, "critic_target.pt"), map_location=self.device))
            self.policy.load_state_dict(torch.load(os.path.join(path, "policy.pt"), map_location=self.device))
            self.critic.eval(); self.critic_target.eval(); self.policy.eval()
            print(f"✓ Loaded checkpoint from {path}")
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
