"""
grid_env.py — Simulated Electric Grid Environment
A Gymnasium-compatible environment where an RL agent manages power dispatch.

Observation: [solar_output, wind_output, gas_capacity, demand]  (all 0-1 normalized)
Actions:
  0 = do nothing (rely on current dispatch)
  1 = boost gas plant output
  2 = curtail demand (rolling blackout)
  3 = draw from battery reserve
  4 = dump excess to battery
Reward: +2 balanced supply, -5 blackout, -1 unnecessary curtailment, +0.5 renewable use
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, seed=None):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)
        self.battery = 0.5          # battery state-of-charge (0-1)
        self.step_count = 0
        self.max_steps = 20
        self.blackouts = 0
        self.rng = np.random.default_rng(seed)

    def _get_obs(self):
        hour = (self.step_count % 24) / 24.0
        solar = float(np.clip(np.sin(hour * np.pi) + self.rng.normal(0, 0.05), 0, 1))
        wind  = float(np.clip(self.rng.uniform(0.1, 0.9) + self.rng.normal(0, 0.08), 0, 1))
        gas   = float(np.clip(self.gas_level + self.rng.normal(0, 0.02), 0, 1))
        demand= float(np.clip(0.5 + 0.3 * np.sin(hour * np.pi * 2) + self.rng.normal(0, 0.05), 0.1, 1))
        return np.array([solar, wind, gas, demand], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.battery = 0.5
        self.gas_level = self.rng.uniform(0.4, 0.8)
        self.step_count = 0
        self.blackouts = 0
        obs = self._get_obs()
        self._last_obs = obs
        return obs, {}

    def step(self, action):
        obs = self._get_obs()
        solar, wind, gas, demand = obs
        supply = solar + wind + gas * 0.6     # gas is partial contributor

        reward = 0.0
        info = {"blackout": False, "curtailment": False}

        if action == 1:    # boost gas
            self.gas_level = min(1.0, self.gas_level + 0.2)
            supply += 0.2
        elif action == 2:  # curtail demand
            demand = max(0.1, demand - 0.3)
            reward -= 1.0
            info["curtailment"] = True
        elif action == 3:  # draw battery
            draw = min(self.battery, 0.25)
            supply += draw
            self.battery -= draw
        elif action == 4:  # charge battery
            excess = max(0, supply - demand)
            stored = min(excess * 0.5, 1.0 - self.battery)
            self.battery += stored

        balance = supply - demand
        if balance >= 0:
            reward += 2.0
            if solar + wind > 0.6:     # bonus for renewable-heavy dispatch
                reward += 0.5
        else:
            reward -= 5.0
            self.blackouts += 1
            info["blackout"] = True

        self.step_count += 1
        self._last_obs = obs
        terminated = self.step_count >= self.max_steps
        return obs, reward, terminated, False, info

    def render(self):
        obs = self._last_obs
        print(f"Step {self.step_count:2d} | Solar:{obs[0]:.2f} Wind:{obs[1]:.2f} "
              f"Gas:{obs[2]:.2f} Demand:{obs[3]:.2f} | Battery:{self.battery:.2f} "
              f"| Blackouts:{self.blackouts}")
