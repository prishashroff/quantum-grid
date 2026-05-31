"""
app.py — Streamlit interactive demo

Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from grid_env import GridEnv
from vqc_agent import VQCAgent
from classical_agent import ClassicalAgent

st.set_page_config(page_title="Quantum Grid RL", page_icon="⚡", layout="wide")

st.title("⚡ Quantum-Enhanced RL for Grid Optimization")
st.caption("A Variational Quantum Circuit agent vs a Classical Q-table agent managing an electric grid.")

with st.sidebar:
    st.header("Simulation settings")
    episodes     = st.slider("Episodes to train", 10, 100, 40, step=5)
    q_lr         = st.slider("Quantum learning rate", 0.005, 0.1, 0.02, step=0.005, format="%.3f")
    c_lr         = st.slider("Classical learning rate", 0.05, 0.5, 0.1, step=0.05)
    eps_decay    = st.slider("Epsilon decay", 0.90, 0.99, 0.97, step=0.01)
    run_btn      = st.button("▶ Train agents", type="primary", use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
q_metric = col1.empty()
c_metric = col2.empty()
savings_metric = col3.empty()
stab_metric    = col4.empty()

chart_ph = st.empty()
adv_ph   = st.empty()
log_ph   = st.empty()

def smooth(data, w=5):
    return np.convolve(data, np.ones(w)/w, mode="same")

def make_figure(q_bl, c_bl):
    eps = np.arange(1, len(q_bl)+1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(eps, q_bl, color="#185FA5", alpha=0.3, linewidth=1)
    ax1.plot(eps, c_bl, color="#3B6D11", alpha=0.3, linewidth=1)
    if len(q_bl) >= 5:
        ax1.plot(eps, smooth(q_bl), color="#185FA5", linewidth=2.5, label="Quantum (VQC)")
        ax1.plot(eps, smooth(c_bl), color="#3B6D11", linewidth=2.5, linestyle="--", label="Classical")
    ax1.set_title("Blackouts per Episode"); ax1.set_xlabel("Episode"); ax1.set_ylabel("Blackouts")
    ax1.legend(); ax1.grid(alpha=0.3)

    cum_q = np.cumsum(q_bl); cum_c = np.cumsum(c_bl)
    ax2.plot(eps, cum_q, color="#185FA5", linewidth=2.5, label="Quantum cumulative")
    ax2.plot(eps, cum_c, color="#3B6D11", linewidth=2.5, linestyle="--", label="Classical cumulative")
    ax2.fill_between(eps, cum_q, cum_c, alpha=0.15, color="#1D9E75")
    ax2.set_title("Cumulative Blackouts"); ax2.set_xlabel("Episode"); ax2.set_ylabel("Total")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig

if run_btn:
    q_agent = VQCAgent(lr=q_lr, epsilon_decay=eps_decay)
    c_agent = ClassicalAgent(lr=c_lr, epsilon_decay=eps_decay)
    env     = GridEnv()

    q_bl_hist, c_bl_hist = [], []
    logs = []

    progress = st.progress(0, text="Training…")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        while not done:
            a = q_agent.act(obs)
            n, r, term, trunc, _ = env.step(a)
            q_agent.train_step(obs, a, r, n, term or trunc)
            obs = n; done = term or trunc
        q_bl = env.blackouts; q_agent.decay_epsilon()

        obs, _ = env.reset()
        done = False
        while not done:
            a = c_agent.act(obs)
            n, r, term, trunc, _ = env.step(a)
            c_agent.train_step(obs, a, r, n, term or trunc)
            obs = n; done = term or trunc
        c_bl = env.blackouts; c_agent.decay_epsilon()

        q_bl_hist.append(q_bl); c_bl_hist.append(c_bl)
        logs.append(f"Ep {ep:3d} | Quantum: {q_bl} BL | Classical: {c_bl} BL | ε={q_agent.epsilon:.2f}")

        tq = sum(q_bl_hist); tc = sum(c_bl_hist)
        saves = tc - tq
        pct   = saves/tc*100 if tc > 0 else 0

        q_metric.metric("Quantum blackouts", tq)
        c_metric.metric("Classical blackouts", tc)
        savings_metric.metric("Quantum saves", f"{saves} ({pct:.0f}%)")
        stab_metric.metric("Current episode", ep)

        if ep % 3 == 0 or ep == episodes:
            chart_ph.pyplot(make_figure(q_bl_hist, c_bl_hist))
            plt.close()

        log_ph.code("\n".join(logs[-12:]))
        progress.progress(ep / episodes, text=f"Episode {ep}/{episodes}")

    progress.empty()
    st.success(f"Training complete! Quantum agent saved {sum(c_bl_hist)-sum(q_bl_hist)} blackouts vs Classical.")
    fig = make_figure(q_bl_hist, c_bl_hist)
    st.pyplot(fig)
    plt.close()
