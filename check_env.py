"""check_env.py — Blackout rate of a uniformly random policy.

If this is under ~5%, no agent has anything to learn. Target: 15-30%.
Run:  python check_env.py
"""
import numpy as np
from grid_env import GridEnv

env = GridEnv(seed=0)
blackouts, steps = 0, 0
for _ in range(200):
    env.reset()
    done = False
    while not done:
        _, _, done, _, info = env.step(env.action_space.sample())
        blackouts += info["blackout"]; steps += 1
print("random-policy blackout rate:", blackouts / steps)
