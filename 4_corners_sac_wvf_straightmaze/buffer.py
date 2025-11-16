# buffer.py
import numpy as np

class ReplayBuffer:
    """
    WVF replay buffer with balanced and planning-aware sampling.
    Stores concatenated [state|goal] for efficiency.
    """

    def __init__(self, max_size, input_size, n_actions, goal_size):
        self.mem_size = int(max_size)
        self.mem_ctr = 0

        # Core memories
        self.state_goal_memory      = np.zeros((self.mem_size, input_size + goal_size), dtype=np.float32)
        self.next_state_goal_memory = np.zeros((self.mem_size, input_size + goal_size), dtype=np.float32)
        self.action_memory          = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory          = np.zeros((self.mem_size,), dtype=np.float32)
        self.terminal_memory        = np.zeros((self.mem_size,), dtype=np.bool_)

        # Separate fields (for HER & diagnostics)
        self.state_memory        = np.zeros((self.mem_size, input_size), dtype=np.float32)
        self.next_state_memory   = np.zeros((self.mem_size, input_size), dtype=np.float32)
        self.goal_memory         = np.zeros((self.mem_size, goal_size), dtype=np.float32)
        self.achieved_goal_memory= np.zeros((self.mem_size, goal_size), dtype=np.float32)
        self.goal_type_memory    = np.array(['environment'] * self.mem_size, dtype=object)

        self.goal_size = goal_size
        self.input_size = input_size

    # --------------------------
    # Core ops
    # --------------------------
    def size(self):
        return min(self.mem_ctr, self.mem_size)

    def __len__(self):
        return self.size()

    def can_sample(self, batch_size):
        return self.mem_ctr >= max(batch_size * 5, batch_size + 256)

    def store_transition(self, state, action, reward, next_state, goal, achieved_goal, done, goal_type='environment'):
        idx = self.mem_ctr % self.mem_size
        sg  = np.concatenate([state, goal]).astype(np.float32)
        nsg = np.concatenate([next_state, goal]).astype(np.float32)

        self.state_goal_memory[idx]      = sg
        self.next_state_goal_memory[idx] = nsg
        self.action_memory[idx]          = np.asarray(action, dtype=np.float32)
        self.reward_memory[idx]          = float(reward)
        self.terminal_memory[idx]        = bool(done)

        self.state_memory[idx]        = np.asarray(state, dtype=np.float32)
        self.next_state_memory[idx]   = np.asarray(next_state, dtype=np.float32)
        self.goal_memory[idx]         = np.asarray(goal, dtype=np.float32)
        self.achieved_goal_memory[idx]= np.asarray(achieved_goal, dtype=np.float32)
        self.goal_type_memory[idx]    = goal_type

        self.mem_ctr += 1

    # --------------------------
    # Standard sampling
    # --------------------------
    def sample_buffer(self, batch_size):
        current = self.size()
        if current == 0:
            raise ValueError("Cannot sample from empty buffer")
        batch_size = min(batch_size, current)
        idx = np.random.choice(current, batch_size, replace=False)
        sg   = self.state_goal_memory[idx]
        nsg  = self.next_state_goal_memory[idx]
        act  = self.action_memory[idx]
        rew  = self.reward_memory[idx]
        done = self.terminal_memory[idx].astype(np.float32)
        return sg, act, rew, nsg, done

    # --------------------------
    # Balanced sampling (for HER)
    # --------------------------
    def sample_buffer_balanced(self, batch_size, pos_frac=0.20):
        """
        Mixes positive and random transitions.
        Conservative by default (pos_frac=0.20).
        """
        current = self.size()
        if current == 0:
            raise ValueError("Cannot sample from empty buffer")

        batch_size = min(batch_size, current)
        pos_idx = np.where(self.reward_memory[:current] > 0.0)[0]

        # Fallback if too few positives
        if pos_idx.size < 10:
            return self.sample_buffer(batch_size)

        pos_n = int(max(1, round(batch_size * pos_frac)))
        pos_n = min(pos_n, pos_idx.size)
        pos_sample = np.random.choice(pos_idx, pos_n, replace=False)

        rem_n = batch_size - pos_n
        all_idx = np.arange(current)
        rem_sample = np.random.choice(all_idx, rem_n, replace=False) if rem_n > 0 else np.array([], dtype=int)

        idx = np.concatenate([pos_sample, rem_sample])
        np.random.shuffle(idx)

        sg   = self.state_goal_memory[idx]
        nsg  = self.next_state_goal_memory[idx]
        act  = self.action_memory[idx]
        rew  = self.reward_memory[idx]
        done = self.terminal_memory[idx].astype(np.float32)

        return sg, act, rew, nsg, done

    # --------------------------
    # Planning support utilities
    # --------------------------
    def sample_recent_trajectories(self, n_steps=128):
        """
        Return a recent sequence of transitions for model-based planning or goal inference.
        """
        current = self.size()
        if current == 0:
            return None
        n_steps = min(n_steps, current)
        idx = np.arange(max(0, current - n_steps), current)
        return dict(
            states=self.state_memory[idx],
            actions=self.action_memory[idx],
            rewards=self.reward_memory[idx],
            next_states=self.next_state_memory[idx],
            goals=self.goal_memory[idx],
            achieved=self.achieved_goal_memory[idx]
        )

    def sample_state_action_pairs(self, n_samples=256):
        """
        Returns random (state, action, goal, reward) tuples for policy refinement or planning priors.
        """
        current = self.size()
        if current == 0:
            return None
        n_samples = min(n_samples, current)
        idx = np.random.choice(current, n_samples, replace=False)
        return dict(
            states=self.state_memory[idx],
            actions=self.action_memory[idx],
            goals=self.goal_memory[idx],
            rewards=self.reward_memory[idx]
        )

    # --------------------------
    # Diagnostics
    # --------------------------
    def get_positive_count(self):
        current = self.size()
        if current == 0:
            return 0
        return int(np.sum(self.reward_memory[:current] > 0.0))
    
    def get_buffer_stats(self):
        current = self.size()
        if current == 0:
            return {
                'size': 0,
                'positive_count': 0,
                'positive_ratio': 0.0,
                'avg_reward': 0.0,
                'min_reward': 0.0,
                'max_reward': 0.0
            }
        rewards = self.reward_memory[:current]
        pos_count = np.sum(rewards > 0.0)
        return {
            'size': current,
            'positive_count': int(pos_count),
            'positive_ratio': float(pos_count / current),
            'avg_reward': float(np.mean(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards))
        }

    def get_recent_stats(self, last_n=500):
        """
        Returns moving window stats for monitoring reward trends & goal distribution.
        """
        current = self.size()
        if current == 0:
            return {'recent_avg_reward': 0.0, 'goal_type_distribution': {}}

        start = max(0, current - last_n)
        rewards = self.reward_memory[start:current]
        goals = self.goal_type_memory[start:current]
        goal_dist = {k: int(np.sum(goals == k)) for k in np.unique(goals)}
        return {
            'recent_avg_reward': float(np.mean(rewards)),
            'recent_positive_ratio': float(np.mean(rewards > 0.0)),
            'goal_type_distribution': goal_dist
        }
    def add(self, state, action, reward, next_state, goal, done, achieved_goal=None, goal_type='environment'):
        """
        Compatibility alias for training loops expecting `add()`.
        Pads actions automatically if the 'done' component is missing.
        """
        # Auto-pad actions to expected dimensionality
        expected_dim = self.action_memory.shape[1]
        action = np.asarray(action, dtype=np.float32)
        if action.shape[-1] < expected_dim:
            pad_width = expected_dim - action.shape[-1]
            action = np.concatenate([action, np.zeros(pad_width, dtype=np.float32)])

        # Handle achieved_goal default
        if achieved_goal is None:
            achieved_goal = next_state[:self.goal_size]

        self.store_transition(state, action, reward, next_state, goal, achieved_goal, done, goal_type)
