<p align="center">
  <img src="https://img.shields.io/badge/Qiskit-1.x-6929C4?logo=qiskit&logoColor=white" alt="Qiskit 1.x" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License" />
  <img src="https://img.shields.io/badge/Platform-NISQ-orange" alt="NISQ Platform" />
</p>

# Q-RLSTC

**Quantum-Enhanced Reinforcement Learning for Sub-Trajectory Clustering**

A hybrid quantum-classical framework that replaces the classical Deep Q-Network in trajectory segmentation with a Variational Quantum Circuit — achieving **22× parameter reduction** (20 vs. 450) while targeting comparable clustering quality on NISQ hardware.

---

## Why Q-RLSTC Exists

Sub-trajectory clustering groups portions of GPS trajectories that share similar movement patterns. Classical [RLSTC](https://github.com/llianga/RLSTCcode) solves this with a Deep Q-Network that learns _where_ to segment. Q-RLSTC asks: **can a 5-qubit quantum circuit learn the same policy with 95% fewer parameters?**

| | Classical RLSTC | Q-RLSTC |
|---|---|---|
| **Policy network** | Dense 5→64→2 MLP | 5-qubit VQ-DQN circuit |
| **Trainable parameters** | ~450 | 20 (Version A) / 32 (Version B) |
| **Optimizer** | SGD + backprop | SPSA (gradient-free) |
| **Hardware** | CPU / GPU | NISQ simulator (Aer) / IBM Quantum |

> **Honest caveat.** Q-RLSTC does not claim quantum speedup. The contribution is _parameter efficiency_ — demonstrating that a shallow quantum circuit can match a classical network on a real RL task — and a validated testbed for quantum RL research.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Q-RLSTC System                            │
├────────────────────────┬────────────────────────────────────────────┤
│   CLASSICAL            │   QUANTUM                                  │
│                        │                                            │
│   Feature extraction   │   VQ-DQN policy network                    │
│   Distance / OD proxy  │   Angle encoding → HEA → Z-expectation    │
│   Reward computation   │   SPSA parameter updates                   │
│   K-means evaluation   │   (Optional) Swap test verification        │
│   Replay buffer        │                                            │
└────────────────────────┴────────────────────────────────────────────┘
```

**Only the policy network is quantum.** Everything else — features, distances, rewards, clustering — stays classical. This is deliberate: the policy has a fixed, low-dimensional I/O (5→2) that maps cleanly to qubits, while distance estimation demands incremental O(1) updates that quantum circuits cannot provide. See [Why classical vs. quantum](docs/wiki/justifications.md) for the full component-by-component analysis.

---

## Quick Start

```bash
# Install
cd q_rlstc
pip install -e ".[dev]"

# Run the synthetic demo
python experiments/run_synth_demo.py

# Run tests
pytest tests/ -v
```

---

## Documentation

> **📖 All deep-dive documents live in [`docs/wiki/`](docs/wiki/).** This README is the entry point.

| Document | What it covers |
|---|---|
| **[System Architecture](docs/wiki/architecture.md)** | Three-layer design, data flow, design philosophy, quantum scope boundary |
| **[MDP & Reward Engineering](docs/wiki/mdp_and_rewards.md)** | State space, action space, anti-gaming constraints, reward function design |
| **[Quantum Circuit Design](docs/wiki/quantum_circuit.md)** | Angle encoding, HEA ansatz, data re-uploading, Q-value extraction, Version A vs B circuits |
| **[Training Pipeline](docs/wiki/training_pipeline.md)** | SPSA optimizer, experience replay, Double DQN, target networks, hyperparameters |
| **[Distance & Clustering](docs/wiki/distance_and_clustering.md)** | IED metric, incremental computation, OD proxy, k-means, swap test (optional) |
| **[Classical vs. Quantum Justifications](docs/wiki/justifications.md)** | Component-by-component analysis: why each part is classical or quantum |
| **[RLSTC vs. Q-RLSTC Comparison](docs/wiki/comparison.md)** | Side-by-side technical comparison across 13 dimensions |
| **[Noise & Hardware Simulation](docs/wiki/noise_and_hardware.md)** | Backend factory, Eagle/Heron profiles, readout error mitigation |
| **[Experimental Design](docs/wiki/experimental_design.md)** | Cross-comparable baselines, metrics, experimental matrix |
| **[Debugging Guide](docs/wiki/debugging.md)** | Common failure modes, diagnostic functions, extension points |
| **[API Reference](docs/wiki/api_reference.md)** | Key classes, functions, and configuration dataclasses |

---

## Project Structure

```
q_rlstc/
├── README.md                      ← You are here
├── pyproject.toml
├── q_rlstc/
│   ├── config.py                  # All configuration dataclasses
│   ├── data/
│   │   ├── features.py            # State feature extraction (Version A + B)
│   │   └── synthetic.py           # Trajectory generation with ground truth
│   ├── quantum/
│   │   ├── vqdqn_circuit.py       # VQ-DQN circuit builder
│   │   ├── backends.py            # Aer backend factory (ideal, Eagle, Heron)
│   │   └── mitigation.py          # Readout error mitigation
│   ├── rl/
│   │   ├── vqdqn_agent.py         # Agent wrapper (ε-greedy, target network)
│   │   ├── spsa.py                # SPSA optimizer
│   │   ├── train.py               # Training loop + MDP environment
│   │   └── replay_buffer.py       # Experience replay buffer
│   └── clustering/
│       ├── classical_kmeans.py    # K-means for episode-end evaluation
│       └── metrics.py             # OD, silhouette, F1 metrics
├── experiments/
│   └── run_synth_demo.py
├── tests/
│   ├── test_angle_encoding.py
│   ├── test_hea_depth.py
│   ├── test_kmeans_update.py
│   └── test_training_smoke.py
└── docs/
    └── wiki/                      # ← Deep-dive documentation
        ├── architecture.md
        ├── mdp_and_rewards.md
        ├── quantum_circuit.md
        ├── training_pipeline.md
        ├── distance_and_clustering.md
        ├── justifications.md
        ├── comparison.md
        ├── noise_and_hardware.md
        ├── experimental_design.md
        ├── debugging.md
        └── api_reference.md
```

---

## Version A vs. Version B

Q-RLSTC ships with two configurations to answer different research questions:

| | Version A — _Close Comparison_ | Version B — _Quantum-Native_ |
|---|---|---|
| **Purpose** | Controlled experiment: isolate the effect of switching MLP → VQC | Explore whether a larger Hilbert space yields better policies |
| **Qubits** | 5 | 8 |
| **State features** | 5 (same dimensionality as classical RLSTC) | 8 (adds `angle_spread`, `curvature_gradient`, `segment_density`) |
| **Trainable params** | 20 | 32 |
| **Readout** | `⟨Z₀⟩`, `⟨Z₁⟩` | `w₀⟨Z₀⟩ + w₁⟨Z₂Z₃⟩`, `w₂⟨Z₁⟩ + w₃⟨Z₄Z₅⟩` |
| **Config** | `QRLSTCConfig(version="A")` | `QRLSTCConfig(version="B")` |

```python
from q_rlstc.config import QRLSTCConfig

# Scientific control — matches classical RLSTC dimensions
config_a = QRLSTCConfig(version="A")  # 5 qubits, 20 params

# Quantum-native — exploits larger Hilbert space
config_b = QRLSTCConfig(version="B")  # 8 qubits, 32 params
```

See [RLSTC vs. Q-RLSTC Comparison](docs/wiki/comparison.md) for the full breakdown across 13 dimensions.

---

## Key Design Decisions

| Decision | Choice | Rationale | Deep dive |
|---|---|---|---|
| Quantum scope | Policy network only | Small fixed I/O (5→2); distance needs O(1) incremental updates | [Justifications](docs/wiki/justifications.md) |
| Encoding | Angle (RY) | 1 feature → 1 qubit; bounded via `arctan`; no normalization needed | [Circuit Design](docs/wiki/quantum_circuit.md) |
| Ansatz | HEA (RY-RZ + linear CNOT) | NISQ-friendly; sufficient expressivity for 5D state | [Circuit Design](docs/wiki/quantum_circuit.md) |
| Optimizer | SPSA | 2 evals per step vs. 40 for parameter-shift; shot-noise robust | [Training Pipeline](docs/wiki/training_pipeline.md) |
| Reward | OD Δ + boundary sharpness − segment penalty | Markov-safe; dense signal on both EXTEND and CUT actions | [MDP & Rewards](docs/wiki/mdp_and_rewards.md) |
| Target network | Double DQN | Prevents Q-value overestimation | [Training Pipeline](docs/wiki/training_pipeline.md) |

---

## Comparative Systems

| System | Domain | Quantum Component | Relationship to Q-RLSTC |
|---|---|---|---|
| **RLSTCcode** | Trajectory clustering | None (classical MLP) | Direct predecessor; same MDP, different approximator |
| **TheFinalQRLSTC** | Trajectory clustering | VQ-DQN | Earlier prototype; Q-RLSTC is the modular rewrite |
| **qDINA** | Database indexing | BQN / TwoLocal | Similar quantum RL; SPSA-only, larger action space |
| **qmeans** | General clustering | Swap test | Unsupervised; shares amplitude encoding patterns |

See [Comparison](docs/wiki/comparison.md) for the full matrix.

---

## NISQ Constraints

| Constraint | Value | Rationale |
|---|---|---|
| Qubits | 5 (Version A) / 8 (Version B) | Matches state dimensionality |
| Circuit depth | ~11 layers | Below decoherence threshold for Eagle/Heron |
| Trainable parameters | 20 / 32 | Below barren plateau threshold |
| Shots (training) | 512 | Balance noise vs. iteration speed |
| Shots (evaluation) | 1024 | Lower variance for metric reporting |
| Entanglement | Linear CNOT chain | Fewer 2-qubit gates = less noise accumulation |

---

## References

1. Liang et al. — "Sub-trajectory clustering with deep reinforcement learning"
2. Chen et al. — "Variational Quantum Circuits for Deep Reinforcement Learning"
3. Schuld et al. — "Evaluating analytic gradients on quantum hardware"
4. Pérez-Salinas et al. — "Data re-uploading for a universal quantum classifier"
5. Spall, J.C. — "Implementation of the Simultaneous Perturbation Algorithm for Stochastic Optimization" (SPSA)

---

## License

MIT License — Research code for academic use.
