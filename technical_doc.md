# Q-RLSTC Technical Documentation & Architecture Guide

## Overview
**Q-RLSTC** is a hybrid quantum-classical reinforcement learning framework designed for parameter-efficient trajectory segmentation. It represents a ground-up refactor, modernization, and significant architectural improvement over the legacy `RLSTCcode-main` classical baseline.

The primary objective of Q-RLSTC is to solve the Trajectory Clustering Segmentation problem using a highly compressed Variational Quantum Circuit (VQC) as the Deep Q-Network policy function—specifically, a 5-qubit VQ-DQN requiring only 34 parameters, achieving a massive compression ratio (~15-38x fewer parameters) compared to the classical dense networks, whilst surviving in small-data and gradient-free (SPSA) optimization regimes.

## 1. Directory Structure

```text
q_rlstc/
├── q_rlstc/
│   ├── data/
│   │   ├── rlstc_mdp.py          # Ported TrajRLclus RL environment
│   │   ├── rlstc_cluster.py      # Trajectory similarity metrics & clustering
│   │   └── observation_tracker.py# [NEW] Streaming Welford Z-scaling normalization
│   ├── rl/
│   │   ├── vqdqn_agent.py        # [NEW] Hybrid Quantum-Classical VQ-DQN agent
│   │   ├── classical_agent.py    # [NEW] Modern PyTorch baseline DQN agent
│   │   ├── cmdp.py               # [NEW] Constrained MDP (Lagrangian multipliers)
│   │   ├── reward_shaping.py     # [NEW] Reward utilities (Geometric scale, CUT penalties)
│   │   ├── replay_buffer.py      # [NEW] Vectorized Experience Replay
│   │   ├── spsa.py               # [NEW] Simultaneous Perturbation Stochastic Approximation
│   │   └── adaptive_shots.py     # [NEW] Hoeffding-bound early stopping mechanism for Qiskit
│   └── quantum/
│       ├── vqdqn_circuit.py      # [NEW] Qiskit 5-qubit parameterized quantum circuit
│       └── backends.py           # [NEW] AerSimulator & Fake Hardware backend provisioners
├── experiments/
│   ├── run_cross_comparison.py   # Primary benchmarking script testing all agents
│   └── run_all.sh                # Shell orchestrator to sweep configurations
└── RLSTCcode-main/               # The legacy classical implementation (Baseline)
```

## 2. Key Architectural Upgrades vs. Legacy `RLSTCcode-main`

The original TF1-based implementation located in `RLSTCcode-main` suffered from several fundamental metric pathologies and software engineering anti-patterns. The Q-RLSTC refactor completely reimagined the architecture:

### A. Constrained MDP (CMDP) instead of Implicit Truncation
- **Legacy Issue:** The baseline implicitly layered static `EXTEND_COST` penalties to prevent "ValCR fragmentation degeneracy" (the network learning to aggressively cut every segment to artifically lower the Overlap Distance). 
- **Q-RLSTC Fix:** Introduced `cmdp.py`. The framework now models trajectory segmentation as a true Constrained Markov Decision Process, enforcing a strict global CUT budget. We implemented adaptive Lagrangian multipliers that dynamically tune the `EXTEND_COST` pseudo-reward to guarantee constraint satisfaction.

### B. Streaming Observation Normalization
- **Legacy Issue:** `MDP.py` passed raw, unscaled point metrics (`overall_sim`, `minsim`) directly into the Keras Dense layers, causing massive internal covariate shift and training instability.
- **Q-RLSTC Fix:** Built `observation_tracker.py`. The agent now accumulates real-time state statistics using Welford's algorithm and `float64` precision scaling, standardizing input vectors (`Z-scaling`) prior to them hitting the VQC data-encoding gates.

### C. True Double-DQN and Vectorized Replay
- **Legacy Issue:** `rl_nn.py` used a simple Python `deque` which slowed down off-policy sampling exponentially. It also lacked a Target Network decoupling during evaluation.
- **Q-RLSTC Fix:** We replaced the lists with a fast Numpy array `ReplayBuffer`. The VQ-DQN and PyTorch baseline both feature explicit target networks, Double-DQN polyak averaging (`update_target_network`), and soft-entropy regularization optionality.

### D. Quantum Hardware-Aware Exeuction
- **Legacy Issue:** N/A (Purely Classical).
- **Q-RLSTC Feature:** The `vqdqn_circuit.py` dynamically handles classical-to-quantum embedding parameters using SPSA. Moreover, `adaptive_shots.py` wraps the Qiskit measurement accumulator in a Hoeffding-bound constraint. If the quantum circuit builds a confident value separation (Q-gap > Bound) early in its sampling phase, it terminates measurement—vastly reducing NISQ simulation overhead.

## 3. The `run_cross_comparison.py` Evaluator

To rigorously compare the quantum and classical paradigms without environment bias, the entire test harness was extracted into `run_cross_comparison.py`.

This script:
1. Provisions the exact same underlying trajectories and target clusters.
2. Normalizes state vectors via shared `ObservationTracker` instances.
3. Steps the agents simultaneously under identical random seeds.
4. Generates a summary validation matrix showcasing:
   - Final Validation Compression Ratios (ValCR).
   - Constrained Optimization Performance ($\lambda$ values, CUT percentages).
   - Parameter compression scaling (34 parameters [Quantum] vs 1200+ parameters [Classical PyTorch/Adam]).

## 5. Visual Reproduction Constraints (Figure 16)

Inspection of the serialized legacy `clusters_E` structure indicated that it did not retain enough information to faithfully reconstruct the global representative-trajectory visualization shown in the original paper's Figure 16. In practice, the surviving saved center geometry was spatially concentrated and produced east-biased overlays inconsistent with the published figure. 

To obtain a visually comparable city-scale summary, we constructed an alternative visualization directly from `env.trajsdata`, selecting a small set of long, real trajectories across directional sectors around the urban core. This yields a figure with a similar global hub-and-spoke appearance, but it should be interpreted as a reconstruction heuristic rather than a guaranteed reproduction of the paper’s exact plotting pipeline.

## 6. How to Run

You can invoke the complete pipeline testing adaptive allocation constraints, statevectors, and classical counterparts via the UV-managed runner:

```bash
uv run python experiments/run_cross_comparison.py --run all --adaptive-shots
```
