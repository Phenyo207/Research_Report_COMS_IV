# buffer.py
# ============================================================================
# Simple Replay Buffer for Goal-Conditioned SAC
# ============================================================================
import numpy as np


class ReplayBuffer:
    """
    Experience replay buffer for goal-conditioned RL.
    Stores transitions as (state, action, reward, next_state, goal, achieved_goal, next_achieved_goal, done).
    """
    
    def __init__(self, max_size, state_dim, action_dim, goal_dim):
        """
        Initialize replay buffer.
        
        Args:
            max_size: Maximum number of transitions to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            goal_dim: Dimension of goal space
        """
        self.max_size = int(max_size)
        self.ptr = 0  # Current position in buffer
        self.size = 0  # Current number of stored transitions

        # Main storage for training (concatenated state+goal)
        self.state_goal = np.zeros((self.max_size, state_dim + goal_dim), dtype=np.float32)
        self.next_state_goal = np.zeros((self.max_size, state_dim + goal_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.max_size,), dtype=np.float32)
        self.dones = np.zeros((self.max_size,), dtype=np.float32)

    def __len__(self):
        """Return current buffer size."""
        return self.size

    def store(self, state, action, reward, next_state, goal, achieved_goal, next_achieved_goal, done):
        """
        Store a transition in the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            goal: Goal being pursued
            achieved_goal: Goal achieved at current state
            next_achieved_goal: Goal achieved at next state
            done: Whether episode terminated
        """
        idx = self.ptr % self.max_size
        
        # Store concatenated state+goal for efficient sampling
        self.state_goal[idx] = np.concatenate([state, goal], axis=-1)
        self.next_state_goal[idx] = np.concatenate([next_state, goal], axis=-1)
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        
        # Update pointer and size
        self.ptr += 1
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (state_goal, actions, rewards, next_state_goal, dones)
        """
        # Sample random indices
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.state_goal[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_state_goal[indices],
            self.dones[indices],
        )
    
    def get_statistics(self):
        """
        Get buffer statistics for monitoring.
        
        Returns:
            Dictionary with buffer stats
        """
        if self.size == 0:
            return {
                'size': 0,
                'avg_reward': 0.0,
                'success_rate': 0.0,
                'min_reward': 0.0,
                'max_reward': 0.0
            }
        
        current_rewards = self.rewards[:self.size]
        success_count = np.sum(current_rewards > 0.0)
        
        return {
            'size': self.size,
            'avg_reward': float(np.mean(current_rewards)),
            'success_rate': float(success_count / self.size),
            'min_reward': float(np.min(current_rewards)),
            'max_reward': float(np.max(current_rewards))
        }