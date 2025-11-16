# buffer.py
import numpy as np

class ReplayBuffer:
    """
    Standard replay buffer for GCRL with HER support
    """
    def __init__(self, max_size, state_dim, action_dim, goal_dim):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        # Store concatenated [state|goal] for network input
        self.state_goal = np.zeros((self.max_size, state_dim + goal_dim), dtype=np.float32)
        self.next_state_goal = np.zeros((self.max_size, state_dim + goal_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.max_size, dtype=np.float32)
        self.terminals = np.zeros(self.max_size, dtype=np.float32)

        # Separate storage for HER relabeling
        self.states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.next_states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.goals = np.zeros((self.max_size, goal_dim), dtype=np.float32)
        self.achieved_goals = np.zeros((self.max_size, goal_dim), dtype=np.float32)

        self.state_dim = state_dim
        self.goal_dim = goal_dim

    def store_transition(self, state, action, reward, next_state, goal, achieved_goal, terminal):
        idx = self.ptr
        
        # Store concatenated versions
        self.state_goal[idx] = np.concatenate([state, goal]).astype(np.float32)
        self.next_state_goal[idx] = np.concatenate([next_state, goal]).astype(np.float32)
        self.actions[idx] = np.array(action, dtype=np.float32)
        self.rewards[idx] = float(reward)
        self.terminals[idx] = float(terminal)

        # Store separate components for HER
        self.states[idx] = np.array(state, dtype=np.float32)
        self.next_states[idx] = np.array(next_state, dtype=np.float32)
        self.goals[idx] = np.array(goal, dtype=np.float32)
        self.achieved_goals[idx] = np.array(achieved_goal, dtype=np.float32)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_buffer(self, batch_size):
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        batch_size = min(batch_size, self.size)
        indices = np.random.choice(self.size, batch_size, replace=False)

        return (
            self.state_goal[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_state_goal[indices],
            self.terminals[indices]
        )

    def can_sample(self, batch_size):
        return self.size >= batch_size

    def __len__(self):
        return self.size

    def get_buffer_stats(self):
        """
        Get buffer statistics for monitoring (matching original codebase)
        """
        if self.size == 0:
            return {
                'size': 0,
                'positive_count': 0,
                'positive_ratio': 0.0,
                'avg_reward': 0.0,
                'min_reward': 0.0,
                'max_reward': 0.0
            }
        
        rewards = self.rewards[:self.size]
        positive_count = int(np.sum(rewards > 0.0))
        
        return {
            'size': self.size,
            'positive_count': positive_count,
            'positive_ratio': float(positive_count / self.size),
            'avg_reward': float(np.mean(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards))
        }