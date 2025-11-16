# agent.py
# ============================================================================
# Simple SAC Agent for Goal-Conditioned RL
# ============================================================================
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from model import Actor, Critic


def soft_update(target, source, tau):
    """Soft update of target network parameters."""
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(tau * s.data + (1.0 - tau) * t.data)


class Agent:
    def __init__(self, state_dim, action_space, goal_dim,
                 gamma=0.98, tau=0.005, lr=3e-4, alpha=0.2,
                 success_threshold=0.5):
        """
        Simple SAC agent for goal-conditioned RL.
        
        Args:
            state_dim: Dimension of state space
            action_space: Gym action space
            goal_dim: Dimension of goal space
            gamma: Discount factor
            tau: Soft update coefficient
            lr: Learning rate
            alpha: Initial entropy coefficient
            success_threshold: Distance threshold for success
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.success_threshold = success_threshold
        
        print(f"[Agent] Device: {self.device}")
        print(f"[Agent] Success threshold: {success_threshold}")

        # ---- Networks ----
        self.critic = Critic(state_dim, action_space.shape[0], goal_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_space.shape[0], goal_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.actor = Actor(state_dim, action_space.shape[0], goal_dim, action_space).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr)

        # ---- Entropy tuning (automatic temperature adjustment) ----
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=lr)
        self.target_entropy = -float(action_space.shape[0])  # Heuristic: -|A|
        self.alpha = alpha

    # ---------------------------------------------------------
    # Reward: +1 if success within threshold, else -0.1
    # ---------------------------------------------------------
    def compute_reward(self, achieved_goal, desired_goal):
        """
        Compute reward based on distance to goal.
        
        Args:
            achieved_goal: Current achieved goal
            desired_goal: Target goal
            
        Returns:
            +1.0 if within success_threshold, else -0.1
        """
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return 1.0 if distance <= self.success_threshold else -0.1

    # ---------------------------------------------------------
    # Action selection
    # ---------------------------------------------------------
    def select_action(self, state, goal, eval_mode=False):
        """
        Select action given state and goal.
        
        Args:
            state: Current state
            goal: Target goal
            eval_mode: If True, use deterministic policy
            
        Returns:
            Action (numpy array)
        """
        state_goal = torch.tensor(
            np.concatenate([state, goal]), 
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)
        
        with torch.no_grad():
            if eval_mode:
                _, _, action = self.actor.sample(state_goal, deterministic=True)
            else:
                action, _, _ = self.actor.sample(state_goal, deterministic=False)
        
        return action.detach().cpu().numpy().flatten()

    # ---------------------------------------------------------
    # SAC parameter update
    # ---------------------------------------------------------
    def update(self, replay_buffer, batch_size):
        """
        Perform one step of SAC update.
        
        Args:
            replay_buffer: ReplayBuffer instance
            batch_size: Batch size for sampling
            
        Returns:
            Tuple of (critic_loss, actor_loss)
        """
        # Sample batch
        sg, act, rew, next_sg, done = replay_buffer.sample(batch_size)
        
        # Convert to tensors
        sg = torch.tensor(sg, dtype=torch.float32, device=self.device)
        act = torch.tensor(act, dtype=torch.float32, device=self.device)
        rew = torch.tensor(rew, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_sg = torch.tensor(next_sg, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # ---- Critic update ----
        with torch.no_grad():
            # Sample next actions from current policy
            next_a, next_logp, _ = self.actor.sample(next_sg)
            
            # Compute target Q-values with clipped double-Q
            q1_next, q2_next = self.critic_target(next_sg, next_a)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_logp
            
            # Compute target
            target_q = rew + (1 - done) * self.gamma * q_next

        # Compute current Q-values
        q1, q2 = self.critic(sg, act)
        
        # Critic loss (MSE for both Q-networks)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        # Update critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ---- Actor update ----
        # Sample new actions from current policy
        new_a, logp, _ = self.actor.sample(sg)
        
        # Compute Q-values for new actions
        q1_new, q2_new = self.critic(sg, new_a)
        q_new = torch.min(q1_new, q2_new)
        
        # Actor loss (maximize Q - α*entropy)
        actor_loss = (self.alpha * logp - q_new).mean()
        
        # Update actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ---- Temperature (alpha) update ----
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        
        # Update alpha value
        self.alpha = self.log_alpha.exp().item()

        # ---- Soft update of target network ----
        soft_update(self.critic_target, self.critic, self.tau)
        
        return float(critic_loss.item()), float(actor_loss.item())

    # ---------------------------------------------------------
    # Save/Load checkpoints
    # ---------------------------------------------------------
    def save_checkpoint(self, path="checkpoints"):
        """Save agent parameters to disk."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pt"))
        torch.save(self.critic_target.state_dict(), os.path.join(path, "critic_target.pt"))
        print(f"[Agent] Checkpoint saved to {path}/")
    
    def load_checkpoint(self, path="checkpoints"):
        """Load agent parameters from disk."""
        try:
            self.actor.load_state_dict(
                torch.load(os.path.join(path, "actor.pt"), map_location=self.device)
            )
            self.critic.load_state_dict(
                torch.load(os.path.join(path, "critic.pt"), map_location=self.device)
            )
            self.critic_target.load_state_dict(
                torch.load(os.path.join(path, "critic_target.pt"), map_location=self.device)
            )
            print(f"[Agent] Checkpoint loaded from {path}/")
        except Exception as e:
            print(f"[Agent] Failed to load checkpoint: {e}")
            raise