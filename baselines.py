"""
baselines.py — Non-learning reference agents.

Each exposes the same interface as the learned agents (act, train_step,
decay_epsilon, epsilon) so train/experiment code can treat them uniformly.
"""

import numpy as np


class RandomAgent:
    epsilon = 0.0
    def act(self, obs): return np.random.randint(5)
    def train_step(self, *a): pass
    def decay_epsilon(self): pass


class AlwaysGasAgent(RandomAgent):
    def act(self, obs): return 1          # always boost gas


class RuleAgent(RandomAgent):
    def act(self, obs):                   # the poster's fixed-rule idea, as code
        solar, wind, gas, demand = obs
        if demand > solar + wind + 0.6 * gas - 0.1:
            return 1                      # boost gas when supply looks tight
        return 0
