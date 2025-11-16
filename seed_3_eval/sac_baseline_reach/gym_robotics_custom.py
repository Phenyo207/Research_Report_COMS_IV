# gym_robotics_custom.py
import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper

class RoboGymObservationWrapper(ObservationWrapper):
    """
    Simple wrapper to ensure observation structure compatibility
    """
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info

    def observation(self, observation):
        return observation