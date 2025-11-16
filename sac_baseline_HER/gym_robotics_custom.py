import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper, ActionWrapper
from gymnasium.spaces import Discrete, Box

class WVFObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info

class PureWVFActionWrapper(ActionWrapper):
    """
    Adds a 'done' action (discrete) or an extra 'done' dimension (continuous).
    """
    def __init__(self, env):
        super().__init__(env)

        if isinstance(env.action_space, Discrete):
            self.is_discrete = True
            self.original_n = env.action_space.n
            self.done_action_index = self.original_n
            self.action_space = Discrete(self.original_n + 1)
        elif isinstance(env.action_space, Box):
            self.is_discrete = False
            low  = np.append(env.action_space.low,  0.0)
            high = np.append(env.action_space.high, 1.0)
            self.original_dim = env.action_space.shape[0]
            self.action_space = Box(low=low, high=high, dtype=env.action_space.dtype)
            self.done_threshold = 0.5
        else:
            raise NotImplementedError("Unsupported action space")

        self.agent_chose_done = False
        self.internal_goals = {}
        self._tmp = np.zeros(2, dtype=np.float32)

    def action(self, action):
        if self.is_discrete:
            if action == self.done_action_index:
                self.agent_chose_done = True
                # Send a noop to env
                return 0
            self.agent_chose_done = False
            return action
        else:
            action = np.asarray(action, dtype=np.float32)
            if action.shape[0] > self.original_dim and action[-1] > self.done_threshold:
                self.agent_chose_done = True
                return action[:-1]
            self.agent_chose_done = False
            return action[:-1] if action.shape[0] > self.original_dim else action

    def step(self, action):
        env_action = self.action(action)
        obs, reward, done, truncated, info = self.env.step(env_action)

        if self.agent_chose_done:
            # Use achieved_goal when available
            if isinstance(obs, dict) and 'achieved_goal' in obs:
                current_state = obs['achieved_goal']
            elif isinstance(obs, dict) and 'observation' in obs:
                current_state = obs['observation'][:2]
            else:
                current_state = obs[:2] if isinstance(obs, np.ndarray) else np.zeros(2, dtype=np.float32)

            self._tmp[0] = np.round(current_state[0], 2)
            self._tmp[1] = np.round(current_state[1], 2)
            goal_tuple = tuple(self._tmp.tolist())
            if goal_tuple not in self.internal_goals:
                self.internal_goals[goal_tuple] = True

            done = True
            info['agent_chose_done'] = True
            info['termination_state'] = current_state
            info['internal_goal_discovered'] = goal_tuple
        else:
            info['agent_chose_done'] = False

        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        self.agent_chose_done = False
        return self.env.reset(**kwargs)

    def get_internal_goals(self):
        return set(self.internal_goals.keys())

class RoboGymObservationWrapper(WVFObservationWrapper):
    def __init__(self, env):
        # First, wrap actions to add WVF 'done' capability
        env = PureWVFActionWrapper(env)
        # Then keep observation wrapper for structure compatibility
        super().__init__(env)
