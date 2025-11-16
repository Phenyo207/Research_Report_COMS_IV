# agent.py
import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from model import WVFQCritic, Actor
from buffer import ReplayBuffer
from continuous_wvf_library import ContinuousGoalOrientedAgent

# ---------------------------------------------
# Utilities
# ---------------------------------------------
def hard_update(target, source):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(s.data)

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)

# ---------------------------------------------
# Agent
# ---------------------------------------------
class Agent(object):
    def __init__(self, num_inputs, action_space, goal_dim, gamma, tau, alpha,
                 target_update_interval, hidden_size, learning_rate,
                 exploration_scaling_factor, her_strategy='future', her_k=4,
                 use_langevin=True):

        self.gamma = gamma
        self.tau = tau
        self.goal_dim = goal_dim
        self.her_strategy = her_strategy
        self.her_k = her_k
        self.observation_dim = num_inputs
        self.target_update_interval = target_update_interval
        self.use_langevin = use_langevin

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Agent] device = {self.device}")
        print(f"[Agent] Langevin sampling = {self.use_langevin}")

        # Networks
        self.critic = WVFQCritic(num_inputs, action_space.shape[0], goal_dim, hidden_size).to(self.device)
        self.critic_target = WVFQCritic(num_inputs, action_space.shape[0], goal_dim, hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.critic_optim = Adam(self.critic.parameters(), lr=learning_rate)

        self.policy = Actor(num_inputs, action_space.shape[0], goal_dim, hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate)

        # Temperature (wider bounds for stability)
        self.target_entropy = -0.5 * float(action_space.shape[0])
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=learning_rate)
        self.alpha = float(alpha)

        # Stable WVF reward params
        self.R_MIN = -10.0
        self.step_reward = -0.1
        self.goal_achievement_reward = 1.0
        self.goal_achievement_threshold = 0.5

        # Goal bookkeeping
        self.internal_goals = set()
        self.successful_goals = set()
        self.goal_success_counts = {}

        self.current_goal_type = 'environment'
        self.current_chosen_goal = None

        self.done_action_count = 0
        self.total_action_count = 0

        # Mastery
        self.mastery_history = []
        self.mastery_eval_states = []
        self.mastery_eval_goals = []

        # Helper for curriculum/Langevin-based proposals
        self.continuous_wvf = ContinuousGoalOrientedAgent(
            state_dim=num_inputs, goal_dim=goal_dim, action_space=action_space,
            hidden_dims=[128, 128], learning_rate=learning_rate,
            gamma=gamma, tau=tau, device=self.device
        )
        
        # Langevin sampling stats
        self.langevin_goal_count = 0
        self.langevin_success_count = 0

    # --------------------------
    # Goal utilities
    # --------------------------
    def goal_distance(self, achieved_goal, desired_goal):
        return np.linalg.norm(achieved_goal - desired_goal, axis=-1)

    def set_current_goal(self, goal, goal_type='environment'):
        self.current_chosen_goal = goal.copy() if isinstance(goal, np.ndarray) else np.array(goal)
        self.current_goal_type = goal_type

    def is_goal_achieved(self, achieved_goal, desired_goal, distance_threshold=None):
        thr = self.goal_achievement_threshold if distance_threshold is None else distance_threshold
        return self.goal_distance(achieved_goal, desired_goal) <= thr

    def update_internal_goals(self, state, achieved_goal, info):
        if info is not None and info.get('agent_chose_done', False):
            goal_tuple = tuple(np.round(achieved_goal, 2))
            if goal_tuple not in self.internal_goals:
                self.internal_goals.add(goal_tuple)
            self.done_action_count += 1

    def goal_policy(self, state, goals, epsilon=0.1):
        """
        GPI over candidate goals with Langevin sampling:
        1. Evaluate existing goals with Q-values
        2. Generate new proposals via Langevin dynamics
        3. Pick the best overall
        """
        # Exploration: sample from Langevin with some probability
        if self.use_langevin and (not goals or np.random.rand() < epsilon):
            try:
                # Generate goals using Langevin dynamics
                extra = self.continuous_wvf.sample_goals_from_ebm(
                    n_goals=5, 
                    state=state, 
                    critic=self.critic, 
                    policy=self.policy
                )
                if extra:
                    selected_goal = extra[np.random.randint(len(extra))]
                    self.langevin_goal_count += 1
                    return selected_goal
            except Exception as e:
                print(f"[goal_policy] Langevin failed: {e}")
        
        # Fallback: random selection from existing goals
        if (not goals) or (np.random.rand() < epsilon):
            extra = self.continuous_wvf.sample_goals_from_ebm(n_goals=3)
            if extra:
                return extra[np.random.randint(len(extra))]
            return goals[np.random.randint(len(goals))] if goals else None

        # Exploitation: evaluate goals and pick best based on Q-values
        eval_goals = goals[:min(10, len(goals))]
        sg = [np.concatenate([state, np.array(g, dtype=np.float32)], axis=-1) for g in eval_goals]
        sg_t = torch.tensor(np.stack(sg), dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            _, _, actions = self.policy.sample(sg_t, deterministic=True)
            q1, q2 = self.critic(sg_t, actions)
            vals = torch.min(q1, q2).squeeze(-1).cpu().numpy()
        
        best_idx = int(np.argmax(vals))
        
        # Optionally refine the best goal with Langevin (with small probability)
        if self.use_langevin and np.random.rand() < 0.1:
            try:
                refined_goal = self.continuous_wvf.refine_goal_with_langevin(
                    state, eval_goals[best_idx], self.critic, self.policy, num_steps=5
                )
                return refined_goal
            except Exception:
                pass
        
        return np.array(eval_goals[best_idx], dtype=np.float32)

    def choose_goal_for_episode(self, env_goal, state):
        """Choose goal using GPI + Langevin sampling"""
        available_goals = [env_goal]
        
        # Add internal goals
        if len(self.internal_goals) > 0:
            available_goals.extend([np.array(g, dtype=np.float32) for g in self.internal_goals])
        
        # Add Langevin-sampled goals (exploration boost)
        if self.use_langevin and np.random.rand() < 0.3:
            try:
                langevin_goals = self.continuous_wvf.sample_goals_from_ebm(
                    n_goals=3, state=state, critic=self.critic, policy=self.policy
                )
                available_goals.extend(langevin_goals)
            except Exception as e:
                print(f"[choose_goal] Langevin sampling failed: {e}")
        
        # Select best goal via GPI
        chosen = self.goal_policy(state, available_goals, epsilon=0.10)
        if chosen is None:
            chosen = env_goal
        
        # Determine goal type
        goal_tuple = tuple(np.round(chosen, 2))
        if goal_tuple in self.internal_goals:
            goal_type = 'internal'
        elif np.allclose(chosen, env_goal, atol=0.1):
            goal_type = 'environment'
        else:
            goal_type = 'langevin'
        
        self.set_current_goal(chosen, goal_type)
        return chosen, goal_type

    # --------------------------
    # Acting
    # --------------------------
    def select_action(self, state, goal, evaluate=False):
        state_goal = np.concatenate([state, goal], axis=-1)
        state_goal = torch.tensor(state_goal, dtype=torch.float32, device=self.device).unsqueeze(0)
        if evaluate:
            _, _, action = self.policy.sample(state_goal, deterministic=True)
        else:
            action, _, _ = self.policy.sample(state_goal, deterministic=False)
        self.total_action_count += 1
        return action.detach().cpu().numpy()[0]

    def get_done_action_ratio(self):
        return 0.0 if self.total_action_count == 0 else self.done_action_count / self.total_action_count
    
    def get_langevin_stats(self):
        """Get statistics about Langevin-sampled goals"""
        if self.langevin_goal_count == 0:
            return {'langevin_goals': 0, 'langevin_success_rate': 0.0}
        return {
            'langevin_goals': self.langevin_goal_count,
            'langevin_success_rate': self.langevin_success_count / self.langevin_goal_count
        }

    # --------------------------
    # Rewards (stable, no annealing)
    # --------------------------
    def compute_reward(self, achieved_goal, desired_goal, info=None, distance_threshold=None):
        """
        +1 whenever within threshold.
        Wrong-DONE gets R_MIN.
        All other steps get step_reward.
        """
        goal_achieved = self.is_goal_achieved(achieved_goal, desired_goal, distance_threshold)
        agent_chose_done = bool(info and info.get('agent_chose_done', False))

        if goal_achieved:
            return self.goal_achievement_reward

        if agent_chose_done:
            return self.R_MIN

        return self.step_reward

    # --------------------------
    # Mastery evaluation support
    # --------------------------
    def collect_eval_samples(self, env, n_states=8, n_goals=8):
        states, goals = [], []
        for _ in range(n_states):
            obs, _ = env.reset()
            states.append(obs['observation'].copy())

        if len(self.internal_goals) > 0:
            sample_int = list(self.internal_goals)[:max(1, n_goals // 2)]
            goals.extend([np.array(g, dtype=np.float32) for g in sample_int])

        for _ in range(max(0, n_goals - len(goals))):
            obs, _ = env.reset()
            goals.append(obs['desired_goal'].copy())

        self.mastery_eval_states = states
        self.mastery_eval_goals = goals
        print(f"[Mastery] Collected eval samples: {len(states)} states, {len(goals)} goals")

    def evaluate_mastery(self, env, horizon=100):
        if len(self.mastery_eval_states) == 0 or len(self.mastery_eval_goals) == 0:
            return 0.0
        successes = []
        for s in self.mastery_eval_states:
            for g in self.mastery_eval_goals:
                if hasattr(env, 'reset_to'):
                    obs, _ = env.reset_to(s)
                else:
                    obs, _ = env.reset()
                state = obs['observation']
                for _ in range(horizon):
                    a = self.select_action(state, g, evaluate=True)
                    next_obs, _, done, truncated, info = env.step(a)
                    state = next_obs['observation']
                    ag = next_obs['achieved_goal']
                    if self.is_goal_achieved(ag, g):
                        successes.append(1)
                        break
                    if done or truncated:
                        successes.append(0)
                        break
        score = float(np.mean(successes)) if successes else 0.0
        self.mastery_history.append(score)
        return score

    def get_mastery_stats(self):
        if not self.mastery_history:
            return dict(current_mastery=0.0, best_mastery=0.0, avg_mastery=0.0, mastery_trend=0.0)
        cur = self.mastery_history[-1]
        best = max(self.mastery_history)
        avg = float(np.mean(self.mastery_history))
        if len(self.mastery_history) >= 10:
            recent = np.mean(self.mastery_history[-5:])
            past = np.mean(self.mastery_history[-10:-5])
            trend = float(recent - past)
        else:
            trend = 0.0
        return dict(current_mastery=cur, best_mastery=best, avg_mastery=avg, mastery_trend=trend)

    # --------------------------
    # Training with correct Q-targets
    # --------------------------
    def update_parameters(self, memory: ReplayBuffer, batch_size, updates, total_numsteps):
        current_size = memory.size()
        pos_count = np.sum(memory.reward_memory[:current_size] > 0.0) if current_size > 0 else 0
        
        if hasattr(memory, "sample_buffer_balanced") and pos_count >= 100:
            sg, act, rew, next_sg, done = memory.sample_buffer_balanced(batch_size, pos_frac=0.20)
        else:
            sg, act, rew, next_sg, done = memory.sample_buffer(batch_size)

        sg      = torch.tensor(sg,      dtype=torch.float32, device=self.device)
        act     = torch.tensor(act,     dtype=torch.float32, device=self.device)
        rew     = torch.tensor(rew,     dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_sg = torch.tensor(next_sg, dtype=torch.float32, device=self.device)
        done    = torch.tensor(done,    dtype=torch.float32, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            next_a, next_logp, _ = self.policy.sample(next_sg, deterministic=False)
            q1_next, q2_next = self.critic_target(next_sg, next_a)
            min_q_next = torch.min(q1_next, q2_next)
            target_q = rew + (1.0 - done) * self.gamma * (min_q_next - self.alpha * next_logp)
            target_q = target_q.clamp(-20.0, 5.0)

        q1, q2 = self.critic(sg, act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(sg, deterministic=False)
        q1_pi, q2_pi = self.critic(sg, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * log_pi - q_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optim.step()

        alpha_loss = (self.log_alpha * (-log_pi.detach() - self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = float(self.log_alpha.exp().clamp(0.01, 0.50).item())

        soft_update(self.critic_target, self.critic, self.tau)

        return dict(critic_loss=float(critic_loss.item()),
                    policy_loss=float(policy_loss.item()),
                    alpha=float(self.alpha),
                    alpha_loss=float(alpha_loss.item()))

    # --------------------------
    # HER: reachable-only relabels
    # --------------------------
    def generate_her_transitions(self, episode_transitions, memory: ReplayBuffer):
        if len(episode_transitions) == 0:
            return

        ep_len = len(episode_transitions)
        achieved = [tr[5] for tr in episode_transitions]

        for t in range(ep_len):
            state, action, _, next_state, goal, achieved_goal, done_flag, info = episode_transitions[t]

            if np.random.rand() < 0.5:
                if t + 1 < ep_len:
                    j = np.random.randint(t + 1, ep_len)
                    new_goal = achieved[j]
                else:
                    new_goal = achieved[-1]
            else:
                if len(self.internal_goals) > 0:
                    ig = np.asarray(list(self.internal_goals), dtype=np.float32)
                    d = np.linalg.norm(ig - achieved_goal, axis=1)
                    new_goal = ig[int(np.argmin(d))]
                else:
                    if t + 1 < ep_len:
                        j = np.random.randint(t + 1, ep_len)
                        new_goal = achieved[j]
                    else:
                        new_goal = achieved[-1]

            r = self.compute_reward(achieved_goal, new_goal, info)
            memory.store_transition(state, action, r, next_state, new_goal, achieved_goal, done_flag, 'her_generated')

    def update_goal_success(self, goal, success: bool):
        g = tuple(np.round(goal, 2))
        if success:
            self.successful_goals.add(g)
            self.goal_success_counts[g] = self.goal_success_counts.get(g, 0) + 1
            
            # Track Langevin success
            if self.current_goal_type == 'langevin':
                self.langevin_success_count += 1

    # --------------------------
    # Checkpointing
    # --------------------------
    def save_checkpoint(self, path="checkpoints"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pt"))
        torch.save(self.critic_target.state_dict(), os.path.join(path, "critic_target.pt"))
        torch.save(self.policy.state_dict(), os.path.join(path, "policy.pt"))
    
    def load_checkpoint(self, path="checkpoints"):
        """Load agent checkpoint from disk"""
        try:
            critic_path = os.path.join(path, "critic.pt")
            critic_target_path = os.path.join(path, "critic_target.pt")
            policy_path = os.path.join(path, "policy.pt")
            
            if not os.path.exists(critic_path):
                raise FileNotFoundError(f"Critic checkpoint not found: {critic_path}")
            if not os.path.exists(critic_target_path):
                raise FileNotFoundError(f"Critic target checkpoint not found: {critic_target_path}")
            if not os.path.exists(policy_path):
                raise FileNotFoundError(f"Policy checkpoint not found: {policy_path}")
            
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            self.critic_target.load_state_dict(torch.load(critic_target_path, map_location=self.device))
            self.policy.load_state_dict(torch.load(policy_path, map_location=self.device))
            
            self.critic.eval()
            self.critic_target.eval()
            self.policy.eval()
            
            print(f"Successfully loaded checkpoint from {path}/")
            
        except Exception as e:
            raise Exception(f"Failed to load checkpoint: {e}")