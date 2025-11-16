import gymnasium as gym

class RoboGymObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)
