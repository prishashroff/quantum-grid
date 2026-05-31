"""
vqc_agent.py — Variational Quantum Circuit Q-Learning Agent

5 qubits (one per action) so VQC directly outputs one Q-value per action.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


N_QUBITS  = 5
N_LAYERS  = 2
N_ACTIONS = 5
N_OBS     = 4

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="autograd")
def vqc(inputs, weights):
    padded = pnp.concatenate([inputs, pnp.array([0.5])])
    qml.AngleEmbedding(padded * np.pi, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class VQCAgent:
    def __init__(
        self,
        lr: float = 0.02,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.97,
    ):
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.weights = pnp.random.uniform(
            -np.pi, np.pi, (N_LAYERS, N_QUBITS, 3), requires_grad=True
        )
        self.opt = qml.AdamOptimizer(stepsize=lr)
        self.history = []

    def q_values(self, state):
        raw = vqc(pnp.array(state, requires_grad=False), self.weights)
        return np.array(raw, dtype=float)

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        return int(np.argmax(self.q_values(state)))

    def train_step(self, state, action, reward, next_state, done):
        # Compute scalar target outside loss_fn — keeps autograd clean
        if done:
            target = float(reward)
        else:
            next_qs = np.array(
                vqc(pnp.array(next_state, requires_grad=False), self.weights),
                dtype=float
            )
            target = float(reward) + self.gamma * float(np.max(next_qs))

        def loss_fn(weights):
            qs = vqc(pnp.array(state, requires_grad=False), weights)
            return (qs[action] - target) ** 2

        # This version of PennyLane's AdamOptimizer.step() returns just
        # the updated weights (not a 3-tuple) — assign directly
        self.weights = self.opt.step(loss_fn, self.weights)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
