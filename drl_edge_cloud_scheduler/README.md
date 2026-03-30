# DRL-Based Multi-Objective Task Scheduling for Edge-Cloud Computing
**Latency, Energy, and SLA Optimisation**

Implementation of the paper:
> *DRL-Based Multi-Objective Task Scheduling for Edge-Cloud Computing: Latency, Energy, and SLA Optimisation*
> Padala Sravan, Dr Mohammed Ali Shaik — SR University, Warangal

---

## Overview

This repository implements an Adaptive-Weight Deep Q-Network (AW-DQN) framework for task scheduling in edge-cloud computing environments. The framework jointly optimises latency, energy consumption, and SLA compliance through a dynamic multi-objective reward function.

## Project Structure

```
drl_edge_cloud_scheduler/
├── README.md
├── config/config.yaml                  # All hyperparameters
├── environment/
│   ├── edge_cloud_env.py               # MDP environment
│   ├── resource_manager.py             # Edge/cloud resource modeling
│   └── workload_generator.py           # Synthetic + real dataset loaders
├── datasets/
│   ├── preprocess_google_traces.py     # Google Cluster Traces preprocessing
│   └── preprocess_azure.py            # Azure Functions preprocessing
├── agents/
│   ├── dqn_agent.py                   # Proposed AW-DQN (main)
│   ├── ddqn_agent.py                  # DDQN baseline
│   ├── ppo_agent.py                   # PPO baseline
│   └── replay_buffer.py               # Experience replay
├── models/dqn_network.py              # DNN / LSTM Q-network
├── baselines/
│   ├── fifo.py                        # FIFO
│   ├── round_robin.py                 # Round-Robin
│   ├── min_min.py                     # Min-Min
│   └── greedy_energy.py              # Greedy-Energy
├── reward/reward_function.py          # Algorithm 3: adaptive reward
├── state_representation/state_builder.py  # Algorithm 2: state vector + PCA
├── training/trainer.py                # Algorithm 1: DQN training loop
├── evaluation/
│   ├── metrics.py                     # All performance metrics
│   ├── evaluator.py                   # Runs all methods
│   └── statistical_tests.py          # t-tests, Wilcoxon, CI
├── experiments/
│   ├── run_synthetic.py
│   ├── run_google_traces.py
│   ├── run_azure.py
│   ├── run_ablation.py
│   ├── run_fault_tolerance.py
│   └── run_scalability.py            
└── results/                           # outputs
```

---

## Installation

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib seaborn pyyaml
```

---

## Quick Start

### 1. Train the proposed AW-DQN on synthetic workload
```bash
python experiments/run_synthetic.py
```

### 2. Evaluate on Google Cluster Traces
```bash
# First preprocess the raw traces
python datasets/preprocess_google_traces.py \
    --input data/raw/google_traces.csv \
    --output data/processed/google_processed.csv

python experiments/run_google_traces.py \
    --data data/processed/google_processed.csv
```

### 3. Evaluate on Azure Functions
```bash
python datasets/preprocess_azure.py \
    --input data/raw/azure_functions.csv \
    --output data/processed/azure_processed.csv

python experiments/run_azure.py \
    --data data/processed/azure_processed.csv
```

### 4. Run ablation study (Tables 10, 14)
```bash
python experiments/run_ablation.py
```

### 5. Fault tolerance experiments (Table 12)
```bash
python experiments/run_fault_tolerance.py
```

### 6. Scalability experiments
```bash
python experiments/run_scalability.py
```

---

## Datasets

| Dataset | Source | Tasks | Description |
|---------|--------|-------|-------------|
| Synthetic | Generated | configurable | Poisson arrivals, uniform attributes |
| Google Cluster Traces | [Google](https://github.com/google/cluster-data) | 700K+ | Cloud task scheduling, 29 days |
| Azure Functions | [Microsoft](https://github.com/Azure/AzurePublicDataset) | 50K+ | Serverless computing workloads |

---

## System Configuration (config/config.yaml)

- **20 edge devices** + **3 cloud servers**
- Task CPU: [1000, 10000] MI, Memory: [100, 512] MB, Deadline: [1, 10]s
- DQN: lr=0.001, γ=0.99, ε decay 1.0→0.01, buffer=50K, batch=64
- Reward base weights: λ₁=0.35 (latency), λ₂=0.25 (energy), λ₃=0.30 (SLA), λ₄=0.10 (overload)

---

## Algorithms Implemented

| Algorithm | File | Description |
|-----------|------|-------------|
| AW-DQN (Proposed) | agents/dqn_agent.py | Adaptive-weight DQN with dynamic reward |
| DDQN | agents/ddqn_agent.py | Double DQN baseline |
| PPO | agents/ppo_agent.py | Proximal Policy Optimisation baseline |
| FIFO | baselines/fifo.py | First-In-First-Out |
| Round-Robin | baselines/round_robin.py | Cyclic assignment |
| Min-Min | baselines/min_min.py | Shortest execution time first |
| Greedy-Energy | baselines/greedy_energy.py | Minimum energy assignment |

---

## Citation

```
@article{sravan2025drl,
  title={DRL-Based Multi-Objective Task Scheduling for Edge-Cloud Computing: Latency, Energy, and SLA Optimisation},
  author={Padala Sravan and Mohammed Ali Shaik},
  journal={SR University},
  year={2026}
}
```
