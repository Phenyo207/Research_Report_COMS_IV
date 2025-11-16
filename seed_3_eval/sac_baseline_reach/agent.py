# agent.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from model import GCRLCritic, GCRLActor

class Agent:
    """
    Standard GCRL SAC agent with Hindsight Experience Replay
    """
    def __init__(self, state_dim, action_dim, goal_dim, action_space,
                 gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4, 
                 hidden_dim=256, her_k=4, distance_threshold=0.05, device='cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.gamma = gamma
        self.tau = tau
        self.her_k = her_k
        self.distance_threshold = distance_threshold
        self.device = torch.device(device)

        # Networks
        self.actor = GCRLActor(state_dim, action_dim, goal_dim, hidden_dim, action_space).to(self.device)
        self.critic = GCRLCritic(state_dim, action_dim, goal_dim, hidden_dim).to(self.device)
        self.critic_target = GCRLCritic(state_dim, action_dim, goal_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=lr)
        self.alpha = alpha

        # Checkpointing
        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def select_action(self, state, goal, evaluate=False):
        """Select action given state and goal"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        goal = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
        state_goal = torch.cat([state, goal], dim=-1)

        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(state_goal, deterministic=True)
            else:
                action, _, _ = self.actor.sample(state_goal, deterministic=False)

        return action.cpu().numpy()[0]

    def compute_reward(self, achieved_goal, desired_goal):
        """
        Compute reward based on distance to goal
        +1.0 if success (distance < threshold)
        -0.1 otherwise
        """
        distance = np.linalg.norm(achieved_goal - desired_goal)
        if distance < self.distance_threshold:
            return 1.0
        else:
            return -0.1

    def update_parameters(self, memory, batch_size, updates):
        """Standard SAC update"""
        # Sample batch
        state_goal, actions, rewards, next_state_goal, terminals = memory.sample_buffer(batch_size)

        state_goal = torch.FloatTensor(state_goal).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state_goal = torch.FloatTensor(next_state_goal).to(self.device)
        terminals = torch.FloatTensor(terminals).unsqueeze(1).to(self.device)

        # Critic update
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(next_state_goal)
            target_q1, target_q2 = self.critic_target(next_state_goal, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_pi
            target_q = rewards + (1 - terminals) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state_goal, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        new_actions, log_pi, _ = self.actor.sample(state_goal)
        q1_new, q2_new = self.critic(state_goal, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_pi - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # Soft update target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss.item()
        }

    def apply_hindsight(self, episode_transitions, original_goal, memory):
        """
        Apply Hindsight Experience Replay
        Strategy: 'future' - replace goal with future achieved goal
        Reward: +1.0 for success, -0.1 otherwise
        """
        episode_length = len(episode_transitions)
        
        for t in range(episode_length):
            # Sample k future timesteps
            future_indices = np.random.randint(t, episode_length, size=self.her_k)
            
            for future_t in future_indices:
                # Use future achieved goal as the goal
                hindsight_goal = episode_transitions[future_t]['achieved_goal']
                
                # Compute reward: +1.0 if within threshold, -0.1 otherwise
                reward = self.compute_reward(
                    episode_transitions[t]['achieved_goal'],
                    hindsight_goal
                )
                
                # Store transition with hindsight goal
                memory.store_transition(
                    state=episode_transitions[t]['state'],
                    action=episode_transitions[t]['action'],
                    reward=reward,
                    next_state=episode_transitions[t]['next_state'],
                    goal=hindsight_goal,
                    achieved_goal=episode_transitions[t]['achieved_goal'],
                    terminal=episode_transitions[t]['terminal']
                )

    def save_checkpoint(self):
        """Save model checkpoints"""
        torch.save(self.actor.state_dict(), 
                   os.path.join(self.checkpoint_dir, 'actor.pth'))
        torch.save(self.critic.state_dict(), 
                   os.path.join(self.checkpoint_dir, 'critic.pth'))
        torch.save(self.critic_target.state_dict(), 
                   os.path.join(self.checkpoint_dir, 'critic_target.pth'))
        print(f"[checkpoint] Saved to {self.checkpoint_dir}")

    def load_checkpoint(self):
        """Load model checkpoints"""
        self.actor.load_state_dict(
            torch.load(os.path.join(self.checkpoint_dir, 'actor.pth')))
        self.critic.load_state_dict(
            torch.load(os.path.join(self.checkpoint_dir, 'critic.pth')))
        self.critic_target.load_state_dict(
            torch.load(os.path.join(self.checkpoint_dir, 'critic_target.pth')))
        print(f"[checkpoint] Loaded from {self.checkpoint_dir}")