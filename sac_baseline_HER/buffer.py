import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, goal_dim):
        self.max_size = int(max_size)
        self.ptr = 0

        self.sg = np.zeros((self.max_size, state_dim + goal_dim), dtype=np.float32)
        self.next_sg = np.zeros((self.max_size, state_dim + goal_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.max_size,), dtype=np.float32)
        self.dones = np.zeros((self.max_size,), dtype=np.float32)

        # store for HER
        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.goal = np.zeros((self.max_size, goal_dim), dtype=np.float32)
        self.ag = np.zeros((self.max_size, goal_dim), dtype=np.float32)
        self.next_ag = np.zeros((self.max_size, goal_dim), dtype=np.float32)

    def __len__(self):
        return min(self.ptr, self.max_size)

    def store(self, state, action, reward, next_state, goal, ag, next_ag, done):
        i = self.ptr % self.max_size
        self.sg[i] = np.concatenate([state, goal], axis=-1)
        self.next_sg[i] = np.concatenate([next_state, goal], axis=-1)
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done

        self.state[i] = state
        self.next_state[i] = next_state
        self.goal[i] = goal
        self.ag[i] = ag
        self.next_ag[i] = next_ag

        self.ptr += 1

    def sample(self, batch_size):
        idxs = np.random.randint(0, len(self), size=batch_size)
        return (
            self.sg[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_sg[idxs],
            self.dones[idxs],
        )
