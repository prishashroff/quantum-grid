"""
classical_agent.py — Classical Q-Table Agent (baseline for comparison)

Uses a discretized state space and a lookup table for Q-values.
Intentionally simple — no neural network — to give the VQC a fair comparison.
"""

import numpy as np


N_ACTIONS = 5
N_BINS    = 5       # bins per observation dimension


def discretize(obs, bins=N_BINS):
    """Map continuous observation [0,1]^4 to a discrete integer tuple."""
    clipped = np.clip(obs, 0.0, 1.0)
    indices = np.floor(clipped * bins).astype(int)
    indices = np.clip(indices, 0, bins - 1)
    return tuple(indices)


class ClassicalAgent:
    def __init__(
        self,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.97,
    ):
        self.lr            = lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: shape = (bins, bins, bins, bins, n_actions)
        table_shape = tuple([N_BINS] * 4) + (N_ACTIONS,)
        self.q_table = np.zeros(table_shape)

        self.history = []

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        s = discretize(state)
        return int(np.argmax(self.q_table[s]))

    def train_step(self, state, action, reward, next_state, done):
        s  = discretize(state)
        s2 = discretize(next_state)
        current_q = self.q_table[s][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[s2])
        self.q_table[s][action] += self.lr * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
