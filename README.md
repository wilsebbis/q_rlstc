<p align="center">
  <img src="https://img.shields.io/badge/Qiskit-1.x-6929C4?logo=qiskit&logoColor=white" alt="Qiskit 1.x" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License" />
  <img src="https://img.shields.io/badge/Platform-NISQ-orange" alt="NISQ Platform" />
</p>

# Q-RLSTC

**Quantum-Enhanced Reinforcement Learning for Sub-Trajectory Clustering**

A hybrid quantum-classical framework that replaces the classical Deep Q-Network in trajectory segmentation with a Variational Quantum Circuit — achieving **22× parameter reduction** (20–30 vs. ~450) while targeting comparable clustering quality on NISQ hardware.

---

## Why Q-RLSTC Exists

Sub-trajectory clustering groups portions of GPS trajectories that share similar movement patterns. Classical [RLSTC](https://github.com/llianga/RLSTCcode) solves this with a Deep Q-Network that learns _where_ to segment. Q-RLSTC asks: **can a 5–8 qubit quantum circuit learn the same policy with 95% fewer parameters?**

| | Classical RLSTC | Q-RLSTC |
|---|---|---|
| **Policy network** | Dense 5→64→2 MLP | 5–8 qubit VQ-DQN circuit |
| **Trainable parameters** | ~450 | 20 (A) / 32 (B) / ~24 (C) / 30 (D) |
| **Optimizer** | SGD + backprop | SPSA / m-SPSA (gradient-free) |
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
│   IED / OD distance    │   Angle encoding → HEA/EQC → Z-expect.    │
│   MDL simplification   │   SPSA / m-SPSA parameter updates          │
│   Reward computation   │   (Optional) Swap test verification        │
│   K-means clustering   │   (C only) Shadow qubit memory             │
│   Incremental updates  │                                            │
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

# Run cross-system comparison (needs T-Drive data)
python experiments/run_cross_comparison.py \
    --traj-path ../RLSTCcode/data/Tdrive_norm_traj \
    --centers-path ../RLSTCcode/data/tdrive_clustercenter \
    --amount 500

# Run tests
pytest tests/ -v
```

---

## Documentation

> **📖 All deep-dive documents live in [`docs/wiki/`](docs/wiki/).** This README is the entry point.

| Document | What it covers |
|---|---|
| **[System Architecture](docs/wiki/architecture.md)** | Three-layer design, data flow, quantum scope boundary |
| **[Technical Deep Dive](docs/wiki/technical_deep_dive.md)** | Versions A/B/C/D architecture, VQ-DQN design, SPSA, dina-quantum comparison |
| **[MDP & Reward Engineering](docs/wiki/mdp_and_rewards.md)** | State space, action space, anti-gaming constraints, reward design |
| **[Quantum Circuit Design](docs/wiki/quantum_circuit.md)** | Angle encoding, HEA/EQC ansatz, data re-uploading, Q-value extraction |
| **[Training Pipeline](docs/wiki/training_pipeline.md)** | SPSA/m-SPSA optimizer, experience replay, Double DQN, target networks |
| **[Distance & Clustering](docs/wiki/distance_and_clustering.md)** | IED metric, incremental computation, OD proxy, k-means, swap test |
| **[Classical vs. Quantum Justifications](docs/wiki/justifications.md)** | Component-by-component analysis: why each part is classical or quantum |
| **[RLSTC vs. Q-RLSTC Comparison](docs/wiki/comparison.md)** | Side-by-side technical comparison across 13 dimensions |
| **[Noise & Hardware Simulation](docs/wiki/noise_and_hardware.md)** | Backend factory, Eagle/Heron profiles, readout error mitigation |
| **[Experimental Design](docs/wiki/experimental_design.md)** | Cross-comparable baselines, metrics, experimental matrix |
| **[Debugging Guide](docs/wiki/debugging.md)** | Common failure modes, diagnostic functions, extension points |
| **[API Reference](docs/wiki/api_reference.md)** | Key classes, functions, and configuration dataclasses |
| **[Benchmarking](docs/wiki/benchmarking.md)** | Performance benchmarks and resource profiling |
| **[Compute Backends](docs/wiki/compute_backends.md)** | Backend configuration and hardware targeting |
| **[Visualization & Plotting](docs/wiki/visualization_and_plotting.md)** | Visualization tools for results analysis |

---

## Project Structure

```
q_rlstc/
├── README.md                      ← You are here
├── pyproject.toml
├── q_rlstc/
│   ├── config.py                  # All configuration dataclasses
│   ├── data/
│   │   ├── features.py            # State feature extraction (Versions A, B, D)
│   │   ├── synthetic.py           # Trajectory generation with ground truth
│   │   └── preprocessing.py       # MDL simplification + TRACLUS pipeline
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
│       ├── classical_kmeans.py    # K-means + incremental cluster updates
│       ├── metrics.py             # OD, silhouette, F1 metrics
│       ├── trajdistance.py        # IED, Fréchet, DTW trajectory distances
│       └── pickle_loader.py       # Load RLSTCcode pickle data files
├── experiments/
│   ├── run_synth_demo.py          # Synthetic data demo
│   ├── run_cross_comparison.py    # Classical vs. quantum comparison runner
│   └── data_bridge.py             # RLSTCcode → Q-RLSTC data conversion
├── tests/
│   ├── test_angle_encoding.py
│   ├── test_hea_depth.py
│   ├── test_kmeans_update.py
│   └── test_training_smoke.py
└── docs/
    └── wiki/                      # ← Deep-dive documentation
```

---

## Versions A / B / C / D

Q-RLSTC ships with four configurations to answer different research questions:

| | **Version A** | **Version B** | **Version C** | **Version D** |
|---|---|---|---|---|
| **Label** | Classical Parity | Quantum Enhanced | Next-Gen Q-RNN | VLDB Aligned |
| **Purpose** | Controlled experiment: isolate MLP → VQC effect | Explore larger Hilbert space | Full quantum-native architecture | Strict 1:1 VLDB paper mapping |
| **Qubits** | 5 | 8 | 6 (5 data + 1 shadow) | 5 |
| **State features** | 5D (matches classical) | 8D (+angle, curvature, density) | 5D + memory signal | 5D (VLDB exact: OD_s, OD_n, OD_b, L_b, L_f) |
| **Parameters** | 20 | 32 | ~24 | 30 |
| **Ansatz** | HEA (RY-RZ + linear CNOT) | HEA | EQC (SO(2)-equivariant) | HEA (3 layers) |
| **Readout** | ⟨Z₀⟩, ⟨Z₁⟩ | w·⟨Z⟩ + w·⟨ZZ⟩ | Soft π(a\|s) via softmax | ⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩ |
| **Actions** | 2 (EXTEND, CUT) | 2 (EXTEND, CUT) | 3 (EXTEND, CUT, DROP) | 2–3 (EXTEND, CUT, opt. SKIP) |
| **Agent** | ε-greedy DQN | ε-greedy DQN | SAC (entropy-regularised) | ε-greedy DQN |
| **Optimizer** | SPSA | SPSA | m-SPSA (momentum) | SPSA |
| **Config** | `version="A"` | `version="B"` | `version="C"` | `version="D"` |

```python
from q_rlstc.config import QRLSTCConfig

config_a = QRLSTCConfig(version="A")  # 5 qubits, 20 params — scientific control
config_b = QRLSTCConfig(version="B")  # 8 qubits, 32 params — quantum-native features
config_c = QRLSTCConfig(version="C")  # 6 qubits, ~24 params — Q-RNN with shadow qubit
config_d = QRLSTCConfig(version="D")  # 5 qubits, 30 params — VLDB paper-exact MDP
```

See [Technical Deep Dive](docs/wiki/technical_deep_dive.md) for detailed version specifications and [Comparison](docs/wiki/comparison.md) for the full breakdown.

---

## Key Design Decisions

| Decision | Choice | Rationale | Deep dive |
|---|---|---|---|
| Quantum scope | Policy network only | Small fixed I/O (5→2); distance needs O(1) incremental updates | [Justifications](docs/wiki/justifications.md) |
| Encoding | Angle (RY) | 1 feature → 1 qubit; bounded via `arctan`; no normalization needed | [Circuit Design](docs/wiki/quantum_circuit.md) |
| Ansatz | HEA / EQC | NISQ-friendly; sufficient expressivity for 5–8D state | [Circuit Design](docs/wiki/quantum_circuit.md) |
| Optimizer | SPSA / m-SPSA | 2 evals per step vs. 40 for parameter-shift; shot-noise robust | [Training Pipeline](docs/wiki/training_pipeline.md) |
| Reward | OD Δ + boundary sharpness − segment penalty | Markov-safe; dense signal on both EXTEND and CUT actions | [MDP & Rewards](docs/wiki/mdp_and_rewards.md) |
| Target network | Double DQN | Prevents Q-value overestimation | [Training Pipeline](docs/wiki/training_pipeline.md) |
| Distance | IED (incremental) | Full RLSTCcode parity; O(1) per step; temporal overlap-aware | [Distance & Clustering](docs/wiki/distance_and_clustering.md) |

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
| Qubits | 5 (A/D) / 8 (B) / 6 (C) | Matches state dimensionality |
| Circuit depth | ~11 layers (A/B/D), ~9 (C) | Below decoherence threshold for Eagle/Heron |
| Trainable parameters | 20–32 | Below barren plateau threshold |
| Shots (training) | 512 (fixed) or 32–512 (adaptive, C) | Balance noise vs. iteration speed |
| Shots (evaluation) | 1024–4096 | Lower variance for metric reporting |
| Entanglement | Linear CNOT (A/B/D), circular (C) | Fewer 2-qubit gates = less noise accumulation |

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
