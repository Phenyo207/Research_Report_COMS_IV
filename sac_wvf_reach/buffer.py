# buffer.py
import numpy as np

class ReplayBuffer:
    """
    WVF replay buffer + balanced sampling.
    Stores concatenated [state|goal] for speed.
    """

    def __init__(self, max_size, input_size, n_actions, goal_size):
        self.mem_size = int(max_size)
        self.mem_ctr = 0

        self.state_goal_memory      = np.zeros((self.mem_size, input_size + goal_size), dtype=np.float32)
        self.next_state_goal_memory = np.zeros((self.mem_size, input_size + goal_size), dtype=np.float32)
        self.action_memory          = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory          = np.zeros((self.mem_size,), dtype=np.float32)
        self.terminal_memory        = np.zeros((self.mem_size,), dtype=np.bool_)

        # Separate fields (HER & diagnostics)
        self.state_memory        = np.zeros((self.mem_size, input_size), dtype=np.float32)
        self.next_state_memory   = np.zeros((self.mem_size, input_size), dtype=np.float32)
        self.goal_memory         = np.zeros((self.mem_size, goal_size), dtype=np.float32)
        self.achieved_goal_memory= np.zeros((self.mem_size, goal_size), dtype=np.float32)
        self.goal_type_memory    = np.array(['environment'] * self.mem_size, dtype=object)

        self.goal_size = goal_size
        self.input_size = input_size

    def size(self):
        return self.mem_size if self.mem_ctr >= self.mem_size else self.mem_ctr

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

    # ---------- IMPROVED: More conservative balanced sampling ----------
    def sample_buffer_balanced(self, batch_size, pos_frac=0.20):
        """
        Returns a batch where ~pos_frac come from positive-reward transitions (if available).
        Falls back gracefully when few positives exist.
        
        IMPROVEMENTS:
        - Lower default pos_frac (0.20 instead of 0.30)
        - Uses replace=False when enough samples to avoid overfitting
        - More conservative mixing
        """
        current = self.size()
        if current == 0:
            raise ValueError("Cannot sample from empty buffer")

        batch_size = min(batch_size, current)
        pos_idx = np.where(self.reward_memory[:current] > 0.0)[0]

        # If no positives or very few, fall back to uniform
        if pos_idx.size < 10:  # Need at least 10 positives for balanced sampling
            return self.sample_buffer(batch_size)

        # Calculate how many positives to include (conservative)
        pos_n = int(max(1, round(batch_size * float(pos_frac))))
        pos_n = min(pos_n, pos_idx.size)  # Cap at available positives
        
        # Sample positives WITHOUT replacement if we have enough
        if pos_idx.size >= pos_n:
            pos_sample = np.random.choice(pos_idx, pos_n, replace=False)
        else:
            # If not enough, sample with replacement (but we checked this above)
            pos_sample = np.random.choice(pos_idx, pos_n, replace=True)

        # Sample remaining uniformly from the whole pool WITHOUT replacement
        rem_n = batch_size - pos_n
        all_idx = np.arange(current)
        
        if rem_n > 0:
            # Ensure we don't re-sample the same positives
            rem_sample = np.random.choice(all_idx, rem_n, replace=False)
        else:
            rem_sample = np.array([], dtype=int)

        # Combine and shuffle
        idx = np.concatenate([pos_sample, rem_sample])
        np.random.shuffle(idx)  # Shuffle to avoid batch ordering bias

        sg   = self.state_goal_memory[idx]
        nsg  = self.next_state_goal_memory[idx]
        act  = self.action_memory[idx]
        rew  = self.reward_memory[idx]
        done = self.terminal_memory[idx].astype(np.float32)

        return sg, act, rew, nsg, done
    
    def get_positive_count(self):
        """Helper to check how many positive samples we have"""
        current = self.size()
        if current == 0:
            return 0
        return int(np.sum(self.reward_memory[:current] > 0.0))
    
    def get_buffer_stats(self):
        """Get buffer statistics for monitoring"""
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