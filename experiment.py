"""
experiment.py — Multi-seed evaluation of all five agents.

For each seed: seed numpy + PennyLane, build fresh agents, train the learned
agents for EPISODES episodes, and score every agent with a greedy (epsilon=0)
evaluation on 100 fixed-seed episodes. The learned agents are also evaluated
every CURVE_EVERY episodes to build a learning curve.

Run:  python experiment.py            # 20 seeds, in parallel
      python experiment.py --seeds 4  # quick check
Outputs: experiment.npz, results.md (table)
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # one thread per worker process

import argparse
from multiprocessing import Pool

import numpy as np
from pennylane import numpy as pnp

from grid_env import GridEnv
from vqc_agent import VQCAgent
from classical_agent import ClassicalAgent
from baselines import RandomAgent, AlwaysGasAgent, RuleAgent
from train import EPISODES, REWARD_SCALE, run_episode

AGENTS       = ["Quantum (VQC)", "Classical (Q-table)", "Rule-based", "Always gas", "Random"]
LEARNED      = AGENTS[:2]
CURVE_EVERY  = 10
EVAL_EPISODES = 100
EVAL_SEED     = 1000


def evaluate(agent, n_episodes=EVAL_EPISODES, seed0=EVAL_SEED):
    """Blackouts per step with exploration off, on a fixed sequence of episodes."""
    saved, agent.epsilon = agent.epsilon, 0.0
    env, bl, steps = GridEnv(seed=seed0), 0, 0
    for _ in range(n_episodes):
        obs, _ = env.reset(); done = False
        while not done:
            obs, _, done, _, info = env.step(agent.act(obs))
            bl += info["blackout"]; steps += 1
    agent.epsilon = saved
    return bl / steps


def run_seed(s):
    np.random.seed(s)
    pnp.random.seed(s)
    agents = {
        "Quantum (VQC)":       VQCAgent(lr=0.02, epsilon_decay=0.97),
        "Classical (Q-table)": ClassicalAgent(lr=0.1, epsilon_decay=0.97),
        "Rule-based":          RuleAgent(),
        "Always gas":          AlwaysGasAgent(),
        "Random":              RandomAgent(),
    }
    curves = {name: [evaluate(agents[name])] for name in LEARNED}   # episode 0
    for name in LEARNED:
        env = GridEnv(seed=s)        # same training env stream for both learners
        for ep in range(1, EPISODES + 1):
            run_episode(env, agents[name])
            if ep % CURVE_EVERY == 0:
                curves[name].append(evaluate(agents[name]))
    final = {name: (curves[name][-1] if name in LEARNED else evaluate(agents[name]))
             for name in AGENTS}
    print(f"seed {s:2d} | " + " | ".join(f"{n}: {final[n]:.3f}" for n in AGENTS), flush=True)
    return final, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    with Pool(args.workers) as pool:
        out = pool.map(run_seed, seeds)

    rates  = np.array([[f[n] for n in AGENTS] for f, _ in out])          # (seeds, agents)
    curves = np.array([[c[n] for n in LEARNED] for _, c in out])         # (seeds, learned, checkpoints)
    np.savez("experiment.npz", rates=rates, curves=curves, agents=AGENTS,
             learned=LEARNED, seeds=seeds, curve_every=CURVE_EVERY)

    n = len(seeds)
    mean = rates.mean(0); ci = 1.96 * rates.std(0, ddof=1) / np.sqrt(n)
    lines = [f"| Agent | Blackout rate (mean ± 95% CI, {n} seeds) |", "|---|---|"]
    for name, m, c in zip(AGENTS, mean, ci):
        lines.append(f"| {name} | {100*m:.1f}% ± {100*c:.1f} |")
    table = "\n".join(lines)
    print("\n" + table)

    q, c = 0, 1
    overlap = (mean[q] - ci[q]) <= (mean[c] + ci[c]) and (mean[c] - ci[c]) <= (mean[q] + ci[q])
    verdict = ("Quantum and classical intervals OVERLAP: no difference can be claimed."
               if overlap else
               f"Quantum and classical intervals do NOT overlap: "
               f"{'quantum' if mean[q] < mean[c] else 'classical'} has the lower blackout rate.")
    print(verdict)
    with open("results.md", "w") as f:
        f.write(table + "\n\n" + verdict + "\n")


if __name__ == "__main__":
    main()
