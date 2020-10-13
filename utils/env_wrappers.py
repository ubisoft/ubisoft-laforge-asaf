import gym
from gym import spaces
import numpy as np

class ScaledActionsWrapper(gym.ActionWrapper):
    # clips action in range [-1,1]^n and convert it to env action ranges
    def __init__(self, env):
        assert isinstance(env.action_space, spaces.Box)
        super().__init__(env)
        self.action_space = env.action_space
        self.low = self.action_space.low
        self.high = self.action_space.high

    def action(self, action):
        assert len(action) == len(self.action_space.low)
        action = np.clip(action, -1, 1)
        scaled_action = self.low + 0.5 * (self.high - self.low) * (action + 1)
        return scaled_action
