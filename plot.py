"""
plot.py — Charts from experiment.npz (produced by experiment.py)

Run:  python plot.py
Outputs: results.png  (a) mean blackout rate per agent, 95% CI error bars
                      (b) learning curve for the learned agents, 95% CI bands
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"Quantum (VQC)": "#2a78d6", "Classical (Q-table)": "#eb6834"}
BASELINE = "#a3a29c"


def ci95(x, axis=0):
    return 1.96 * x.std(axis=axis, ddof=1) / np.sqrt(x.shape[axis])


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_results(path="experiment.npz"):
    if not Path(path).exists():
        print(f"{path} not found — run  python experiment.py  first.")
        return
    d = np.load(path)
    rates, curves = d["rates"] * 100, d["curves"] * 100
    agents, learned = list(d["agents"]), list(d["learned"])
    every, n = int(d["curve_every"]), rates.shape[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("white")

    # (a) bar chart
    mean, err = rates.mean(0), ci95(rates)
    colors = [COLORS.get(a, BASELINE) for a in agents]
    x = np.arange(len(agents))
    ax1.bar(x, mean, color=colors, width=0.6, edgecolor="white", linewidth=2)
    ax1.errorbar(x, mean, yerr=err, fmt="none", ecolor=INK, elinewidth=1.2, capsize=4)
    for xi, m, e in zip(x, mean, err):
        ax1.text(xi, m + e + 0.4, f"{m:.1f}%", ha="center", va="bottom", fontsize=9, color=INK)
    ax1.set_xticks(x, [a.replace(" (", "\n(") for a in agents])
    ax1.set_ylabel("Blackout rate (% of steps)", color=INK_2)
    ax1.set_title(f"(a) Greedy evaluation, mean ± 95% CI ({n} seeds)", loc="left", fontsize=11, color=INK)
    style(ax1)

    # (b) learning curves
    eps = np.arange(curves.shape[2]) * every
    for i, name in enumerate(learned):
        m, e = curves[:, i].mean(0), ci95(curves[:, i])
        ax2.plot(eps, m, color=COLORS[name], linewidth=2, label=name)
        ax2.fill_between(eps, m - e, m + e, color=COLORS[name], alpha=0.18, linewidth=0)
    for name in ("Rule-based", "Random"):
        j = agents.index(name)
        ax2.axhline(rates[:, j].mean(), color=BASELINE, linewidth=1, linestyle="--")
        ax2.text(2, rates[:, j].mean() + 0.4, f"{name} baseline", color=INK_2, fontsize=9, va="bottom")
    ax2.set_xlim(0, eps[-1])
    ax2.set_xlabel("Training episodes", color=INK_2)
    ax2.set_ylabel("Blackout rate (% of steps)", color=INK_2)
    ax2.set_title("(b) Learning curve, greedy eval every 10 episodes", loc="left", fontsize=11, color=INK)
    ax2.legend(frameon=False, fontsize=9, loc="upper right")
    style(ax2)

    fig.tight_layout()
    fig.savefig("results.png", dpi=150, bbox_inches="tight")
    print("Chart saved to results.png")


if __name__ == "__main__":
    plot_results()
