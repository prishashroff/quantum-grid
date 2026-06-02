# Grid Optimization using Quantum Enhanced Reinforcement Learning

Developed a Variational Quantum Circuit (VQC) reinforcement learning agent that can manage an electric grid. It is evaulated against a classical Q-table agent. The solution balances solar, wind, and gas power to prevent blackouts.

## What it does

- Simulates an electric grid with stochastic solar/wind generation and fluctuating demand
- Trains a **Quantum Q-Learner** (Quantum simulation is run on regular computer using the PennyLane library) and a **Classical Q-table** agent side-by-side
- Plots learning curves comparing the performance of the quantum agent against the classical agent 

## Structure

```
quantum-grid-rl/
├── grid_env.py        # Gymnasium grid environment
├── vqc_agent.py       # Variational Quantum Circuit agent
├── classical_agent.py # Classical Q-table agent (baseline)
├── train.py           # Training loop (run this to get the results in plot.py)
├── plot.py            # Plot results from training
├── app.py             # Interactive Streamlit demo
└── requirements.txt
```

## Quickstart

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Train both agents (takes ~2-5 minutes)
python3 train.py

# 3. Plot results
python3 plot.py

# 4. Launch interactive demo
streamlit run app.py
```


