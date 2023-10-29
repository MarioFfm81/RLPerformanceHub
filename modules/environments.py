# Author: Mario Niemann

# creating or integrating new environments, outside the standard Gymnasium, is done here

import gymnasium as gym
import numpy as np
from ray.tune.registry import register_env

class ActionNoiseWrapper(gym.ActionWrapper):
    def __init__(self, env, noise_std=0.1):
        super(ActionNoiseWrapper, self).__init__(env)
        self.noise_std = noise_std

    def action(self, action):
        noise = np.random.normal(0, self.noise_std, size=action.shape)
        noisy_action = action + noise
        return np.clip(noisy_action, self.action_space.low, self.action_space.high)

# create Noisy Pendullum Environment, based on Pendulum V1
def PendulumNoisyEnv(envConfig):
    PEnv = gym.make('Pendulum-v1')
    return ActionNoiseWrapper(PEnv, noise_std=1)


# main function to register the environments
def registerEnvironments():
    register_env('noisy_Pendulum-v1', PendulumNoisyEnv)