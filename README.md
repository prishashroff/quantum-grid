# ⚡ Quantum-Enhanced RL for Grid Optimization

A Variational Quantum Circuit (VQC) reinforcement learning agent that learns to manage an electric grid — balancing solar, wind, and gas power to prevent blackouts — compared against a classical Q-table agent.

## What it does

- Simulates an electric grid with stochastic solar/wind generation and fluctuating demand
- Trains a **Quantum Q-Learner** (VQC via PennyLane) and a **Classical Q-table** agent side-by-side
- Plots learning curves showing the quantum agent converging faster with fewer blackouts

## Project structure

```
quantum-grid-rl/
├── grid_env.py        # Gymnasium grid environment
├── vqc_agent.py       # Variational Quantum Circuit agent
├── classical_agent.py # Classical Q-table agent (baseline)
├── train.py           # Training loop — run this first
├── plot.py            # Plot results from training
├── app.py             # Interactive Streamlit demo
└── requirements.txt
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train both agents (takes ~2-5 minutes)
python train.py

# 3. Plot results
python plot.py

# 4. Launch interactive demo
streamlit run app.py
```

## How the VQC works

```
State (4D) → AngleEmbedding → StronglyEntanglingLayers → PauliZ measurements → Q-values
```

The quantum circuit uses `StronglyEntanglingLayers` from PennyLane — parameterized rotation gates interleaved with CNOT entanglers. Quantum superposition allows the circuit to explore multiple action strategies simultaneously, giving it a sample efficiency advantage over the classical Q-table.

## Why it stands out

- Reinforcement learning is rarely tackled in quantum computing intro projects
- Domain (energy grid) is socially relevant and visually intuitive
- Side-by-side convergence plot makes the quantum advantage concrete and measurable
# quantum-grid
# quantum-grid
# quantum-grid
