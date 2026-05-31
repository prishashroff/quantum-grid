"""
plot.py — Visualize training results from results.npz

Run:  python plot.py
Requires: matplotlib, numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


def smooth(data, window=5):
    """Moving average smoothing."""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="same")


def plot_results(npz_path="results.npz"):
    if not Path(npz_path).exists():
        print("results.npz not found — run  python train.py  first.")
        return

    data = np.load(npz_path)
    q_bl = data["q_blackouts"]
    c_bl = data["c_blackouts"]
    eps  = np.arange(1, len(q_bl) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Quantum vs Classical RL — Electric Grid Optimization", fontsize=14, fontweight="bold")

    # 1. Raw blackouts
    ax = axes[0, 0]
    ax.plot(eps, q_bl, color="#185FA5", alpha=0.4, linewidth=1, label="Quantum (raw)")
    ax.plot(eps, c_bl, color="#3B6D11", alpha=0.4, linewidth=1, label="Classical (raw)")
    ax.plot(eps, smooth(q_bl), color="#185FA5", linewidth=2.5, label="Quantum (smoothed)")
    ax.plot(eps, smooth(c_bl), color="#3B6D11", linewidth=2.5, linestyle="--", label="Classical (smoothed)")
    ax.set_title("Blackouts per Episode (Learning Curve)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Blackouts")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 2. Cumulative blackouts
    ax = axes[0, 1]
    ax.plot(eps, np.cumsum(q_bl), color="#185FA5", linewidth=2.5, label="Quantum cumulative")
    ax.plot(eps, np.cumsum(c_bl), color="#3B6D11", linewidth=2.5, linestyle="--", label="Classical cumulative")
    ax.fill_between(eps, np.cumsum(q_bl), np.cumsum(c_bl), alpha=0.15, color="#1D9E75", label="Quantum advantage")
    ax.set_title("Cumulative Blackouts")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total blackouts")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 3. Advantage per episode
    ax = axes[1, 0]
    advantage = np.array(c_bl) - np.array(q_bl)
    colors = ["#185FA5" if v >= 0 else "#E24B4A" for v in advantage]
    ax.bar(eps, advantage, color=colors, alpha=0.7, width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Per-Episode Advantage (positive = quantum wins)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Classical – Quantum blackouts")
    ax.grid(alpha=0.3, axis="y")

    # 4. Summary bar
    ax = axes[1, 1]
    totals  = [sum(q_bl), sum(c_bl)]
    labels  = ["Quantum\n(VQC)", "Classical\n(Q-table)"]
    bar_colors = ["#185FA5", "#3B6D11"]
    bars = ax.bar(labels, totals, color=bar_colors, width=0.5, alpha=0.85)
    for bar, val in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontweight="bold")
    savings = totals[1] - totals[0]
    pct = savings / totals[1] * 100 if totals[1] > 0 else 0
    ax.set_title(f"Total Blackouts — Quantum saves {savings} ({pct:.1f}%)")
    ax.set_ylabel("Total blackouts across all episodes")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("results.png", dpi=150, bbox_inches="tight")
    print("Chart saved to results.png")
    plt.show()


if __name__ == "__main__":
    plot_results()
