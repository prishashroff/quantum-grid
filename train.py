"""
train.py — Train both agents and save results

Run:  python train.py
Outputs: results.npz (numpy arrays for plotting), training log to stdout
"""

import numpy as np
from grid_env import GridEnv
from vqc_agent import VQCAgent
from classical_agent import ClassicalAgent

EPISODES  = 60
SEED      = 42


def run_episode(env, agent, train=True):
    obs, _ = env.reset(seed=None)
    total_reward = 0
    done = False
    while not done:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if train:
            agent.train_step(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += reward
    if train:
        agent.decay_epsilon()
    return env.blackouts, total_reward


def train():
    print("=" * 55)
    print("  Quantum vs Classical Grid RL — Training")
    print("=" * 55)

    q_agent = VQCAgent(lr=0.02, epsilon_decay=0.97)
    c_agent = ClassicalAgent(lr=0.1,  epsilon_decay=0.97)
    env     = GridEnv(seed=SEED)

    q_blackouts, c_blackouts = [], []
    q_rewards,   c_rewards   = [], []

    for ep in range(1, EPISODES + 1):
        q_bl, q_r = run_episode(env, q_agent)
        c_bl, c_r = run_episode(env, c_agent)

        q_blackouts.append(q_bl)
        c_blackouts.append(c_bl)
        q_rewards.append(q_r)
        c_rewards.append(c_r)

        if ep % 5 == 0 or ep == 1:
            print(f"Ep {ep:3d} | Quantum: {q_bl} blackouts (ε={q_agent.epsilon:.2f}) "
                  f"| Classical: {c_bl} blackouts (ε={c_agent.epsilon:.2f})")

    total_q  = sum(q_blackouts)
    total_c  = sum(c_blackouts)
    savings  = total_c - total_q
    pct      = savings / total_c * 100 if total_c > 0 else 0

    print("\n" + "=" * 55)
    print(f"  Quantum total blackouts  : {total_q}")
    print(f"  Classical total blackouts: {total_c}")
    print(f"  Quantum saved {savings} blackouts ({pct:.1f}% fewer)")
    print("=" * 55)

    np.savez(
        "results.npz",
        q_blackouts=q_blackouts,
        c_blackouts=c_blackouts,
        q_rewards=q_rewards,
        c_rewards=c_rewards,
    )
    print("\nResults saved to results.npz — run  python plot.py  to see charts.")
    return q_blackouts, c_blackouts


if __name__ == "__main__":
    train()
