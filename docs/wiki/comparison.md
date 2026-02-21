# RLSTC vs. Q-RLSTC: Technical Comparison

[← Back to README](../../README.md) · [Justifications](justifications.md) · **Comparison** · [Noise & Hardware →](noise_and_hardware.md)

---

A side-by-side analysis across 13 dimensions, covering architecture, design decisions, and all four Q-RLSTC versions (**A**, **B**, **C**, **D**).

## 1. Architecture Overview

> [!IMPORTANT]
> The "RLSTC" column below describes the **original RLSTC paper's** architecture (SGD, soft-update, single DQN). For **controlled experiments**, the classical MLP baselines intentionally mirror Q-RLSTC's training setup (SPSA, hard-copy, Double DQN) so that the function approximator is the **only** independent variable. See [Experimental Design](experimental_design.md) for the controlled comparison specification.

| Dimension | RLSTC (Original Paper) | Q-RLSTC (This Implementation) |
|---|---|---|
| **Policy network** | Classical DQN (TF 1.x / Keras) | VQ-DQN (Qiskit parameterised circuit) |
| **Optimizer** | SGD (lr = 0.001) | SPSA / m-SPSA (gradient-free, NISQ-suitable) |
| **Distance / clustering** | Incremental IED (custom) | IED (ported) + classical k-means + incremental OD proxy |
| **Loss function** | Huber loss | Huber loss (same) |
| **Target network** | Soft-update (τ = 0.05) | Periodic hard copy (`target_update_freq`) |
| **Double DQN** | No | Yes |

## 2. State Representation

### Classical RLSTC — 5 Features

| # | Feature | Source |
|---|---------|--------|
| 0 | `overall_sim` — OD to nearest cluster centre | `MDP.py` |
| 1 | `min_sim` — Minimum per-point similarity | `MDP.py` |
| 2 | `segment_len` — Current segment point count | `MDP.py` |
| 3 | `traj_progress` — Fraction consumed | `MDP.py` |
| 4 | `seg_count` — Segments created so far | `MDP.py` |

### Q-RLSTC Version A — 5 Features

| # | Feature | How it differs from RLSTC |
|---|---------|-----------------------------|
| 0 | `od_segment` — Projected OD if we split | Proxy-based, not full IED |
| 1 | `od_continue` — Projected OD if we extend | Running average, not global recalc |
| 2 | `baseline_cost` — MDL compression score | TRACLUS-inspired, replaces `min_sim` |
| 3 | `len_backward` — Normalised segment length | Same concept, different normalisation |
| 4 | `len_forward` — Remaining trajectory | Same concept |

### Q-RLSTC Version B — 8 Features

Adds three quantum-native features to Version A:

| # | Feature | Rationale |
|---|---------|-----------| 
| 5 | `angle_spread` — Variance of arctan-encoded features | Bloch sphere spread |
| 6 | `curvature_gradient` — Rate of curvature change | 2nd-order geometric signal |
| 7 | `segment_density` — Points per unit distance | Congestion without explicit speed |

### Q-RLSTC Version C — 5D + Memory

Same 5 features as Version A, plus a **shadow qubit** (qubit 0) that persists quantum state across time steps, creating recurrent memory without additional classical features.

### Q-RLSTC Version D — 5 Features (VLDB Exact)

| # | Feature | Source |
|---|---------|-----------| 
| 0 | `OD_s` — OD if we CUT here | Equation (19) of VLDB paper |
| 1 | `OD_n` — OD if we EXTEND | Same |
| 2 | `OD_b` — TRACLUS expert baseline | Ablation-confirmed improvement |
| 3 | `L_b` — Normalised backward segment length | Same |
| 4 | `L_f` — Normalised forward remaining length | Same |

## 3. Action Space

| Version | Actions | Description |
|---|---|---|
| **A, B** | 2 | EXTEND (0) or CUT (1) |
| **C** | 3 | EXTEND (0), CUT (1), DROP (2) — actively filters noise |
| **D** | 2–3 | EXTEND, CUT, optional SKIP(S) that fast-forwards S points |

## 4. Reward Functions

| Component | RLSTC | Q-RLSTC (A/B) | Q-RLSTC (C) | Q-RLSTC (D) |
|---|---|---|---|---|
| **Main signal** | `ΔOD = last_od − current_od` (full IED) | `α · od_improvement` (proxy) | Same + DROP penalty | `OD(s_t) − OD(s_{t+1})` (paper exact) |
| **Boundary quality** | None | `β · boundary_sharpness` | Same | None (paper doesn't use it) |
| **Over-segmentation** | Implicit via `MIN_SEGMENT_LEN` | Explicit `−penalty` in reward | Same + DROP micro-penalty (−0.05) | Implicit |
| **SKIP reward** | N/A | N/A | N/A | +0.05 × S (linear, low-var segments) |
| **Markov safety** | Depends on global cluster state | Uses only incremental quantities | Same | Same |

## 5. Quantum Circuit

| Aspect | Version A | Version B | Version C | Version D |
|---|---|---|---|---|
| **Qubits** | 5 | 8 | 6 (5+1 shadow) | 5 |
| **Encoding** | Angle (RY) | Angle (RY) | Angle (RY) | Angle (RY) |
| **Ansatz** | HEA (RY-RZ + linear CNOT) | HEA | EQC (RZ + circular CNOT) | HEA (3 layers) |
| **Variational layers** | 2 | 2 | 2 | 3 |
| **Trainable params** | 20 | 32 | ~24 | 30 |
| **Entanglement** | 4 CNOTs (linear) | 7 CNOTs (linear) | 6 CNOTs (circular) | 4 CNOTs (linear) |
| **Data re-uploading** | Yes | Yes | Yes | Yes |
| **Readout** | ⟨Z₀⟩, ⟨Z₁⟩ | w·⟨Z⟩ + w·⟨ZZ⟩ | Softmax π(a\|s) | ⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩ |

## 6. Optimizer

| | RLSTC | Q-RLSTC (A/B/D) | Q-RLSTC (C) |
|---|---|---|---|
| **Method** | SGD with backprop | SPSA (gradient-free) | m-SPSA (momentum-averaged) |
| **Evals per step** | 1 (forward + backward) | 2 (forward only) | 2 + EMA smoothing |
| **Shot noise?** | N/A | Robust by design | Extra-robust via momentum |
| **Gradient clipping** | No | Yes (max norm 1.0) | Yes |

## 7. Distance Computation

| | RLSTC | Q-RLSTC |
|---|---|---|
| **Primary metric** | Incremental IED | IED (ported in `trajdistance.py`) + OD proxy |
| **Per-step cost** | O(1) amortised | O(1) |
| **Full computation** | Every CUT action | Episode-end (k-means) or incremental update |
| **Available metrics** | IED, Fréchet, DTW | IED, Fréchet, DTW, OD, Silhouette, F1 |
| **Incremental updates** | `cluster.py` | `classical_kmeans.py` (ported) |

## 8. Data Structures

| | RLSTC | Q-RLSTC |
|---|---|---|
| **Point** | Plain class with `x, y, t` | `@dataclass` with `distance()`, `to_array()` |
| **Segment** | Class with distance methods | Implicit (index range) |
| **Trajectory** | `Traj(points, size, ts, te)` | `@dataclass` with `boundaries`, `labels` |
| **Replay buffer** | `deque(maxlen=2000)` in DQN class | Separate `ReplayBuffer(5000)` |
| **Cluster state** | Mutable dict `{id: [data]}` | Same format (ported), `@dataclass ClusterState` |
| **Config** | Hardcoded constants | Nested `@dataclass` hierarchy |

## 9. Version Comparison Summary

| Dimension | A (Classical Parity) | B (Quantum Enhanced) | C (Next-Gen Q-RNN) | D (VLDB Aligned) |
|---|---|---|---|---|
| **Goal** | Isolate quantum vs. classical | Leverage larger Hilbert space | Full quantum-native architecture | Strict VLDB paper reproduction |
| **Qubits** | 5 | 8 | 6 | 5 |
| **Features** | 5D (matches RLSTC) | 8D (3 quantum-native) | 5D + shadow memory | 5D (VLDB exact) |
| **Readout** | Single-qubit Z | Multi-observable (Z + ZZ) | Softmax distribution | Multi-qubit Z |
| **Params** | 20 | 32 | ~24 | 30 |
| **Actions** | 2 | 2 | 3 (+ DROP) | 2–3 (+ opt. SKIP) |
| **Agent** | ε-greedy DQN | ε-greedy DQN | SAC | ε-greedy DQN |
| **Optimizer** | SPSA | SPSA | m-SPSA | SPSA |
| **Shots** | Fixed (512/4096) | Fixed | Adaptive (32→512) | Fixed |
| **Config** | `version="A"` | `version="B"` | `version="C"` | `version="D"` |

## 10. Noise & Hardware

| | RLSTC | Q-RLSTC |
|---|---|---|
| **Noise simulation** | None | Full stack (ideal, simple, Eagle, Heron) |
| **Error mitigation** | None | Readout calibration matrix |
| **Backend** | CPU (TensorFlow) | Qiskit Aer (configurable) |

## 11. Training Pipeline

| Aspect | RLSTC | Q-RLSTC (A/B/D) | Q-RLSTC (C) |
|---|---|---|---|
| **Loop** | Iterate points → EXTEND/CUT | Same | Same + DROP/SKIP |
| **Replay** | Internal to DQN (2,000) | Separate buffer (5,000) | Same |
| **Target update** | Soft (τ = 0.05 every batch) | Hard copy every N episodes | Same |
| **Double DQN** | No | Yes | Yes |
| **Anti-gaming** | `MIN_SEGMENT_LEN` | Same + explicit reward penalty | Same + DROP penalty |

## 12. Design Rationale

| Decision | Rationale |
|---|---|
| Only policy is quantum | Fixed I/O (5→2); distance needs O(1) updates |
| SPSA over parameter-shift | 2 evals vs. 40; scales to larger circuits |
| Angle encoding | 1 feature → 1 qubit; bounded via `arctan` |
| Data re-uploading | Expressivity without depth; proven technique |
| Version A exists | Scientific control: isolate the approximator |
| Version B exists | Explore whether more qubits + richer features helps |
| Version C exists | Full quantum-native: shadow memory, EQC, SAC, adaptive shots |
| Version D exists | VLDB paper reproduction: exact MDP → VQC substitution |
| IED ported to Q-RLSTC | Classical parity: identical distance metric for fair comparison |
| MDL simplification ported | Ensures identical preprocessing between systems |
| Pickle loader | Direct data sharing between RLSTCcode and Q-RLSTC |

## 13. File Reference

### Classical RLSTC

| File | Purpose |
|---|---|
| `rl_nn.py` | DQN: model, training, target network |
| `MDP.py` | Environment: state features, reward, step logic |
| `rl_train.py` | Training loop orchestration |
| `rl_estimate.py` | Evaluation / inference |
| `cluster.py` | Incremental IED clustering |
| `trajdistance.py` | IED, Fréchet, DTW distances |
| `segment.py` | Segment distance metrics |
| `point.py` | Point data structure |
| `preprocessing.py` | MDL simplification, normalisation |

### Q-RLSTC

| File | Purpose |
|---|---|
| `quantum/vqdqn_circuit.py` | Circuit: encoding, HEA, measurement |
| `rl/vqdqn_agent.py` | Agent: ε-greedy, Double DQN, target network |
| `rl/train.py` | Training loop + MDP environment |
| `rl/spsa.py` | SPSA optimizer |
| `rl/replay_buffer.py` | Experience replay |
| `config.py` | Configuration dataclasses (A/B/C/D) |
| `data/features.py` | State feature extraction (A, B, D) |
| `data/preprocessing.py` | MDL simplification + TRACLUS pipeline |
| `data/synthetic.py` | Trajectory generation |
| `clustering/classical_kmeans.py` | K-means + incremental cluster updates |
| `clustering/metrics.py` | OD, silhouette, F1 |
| `clustering/trajdistance.py` | IED, Fréchet, DTW (ported from RLSTC) |
| `clustering/pickle_loader.py` | Load RLSTCcode pickle data files |
| `quantum/backends.py` | Noise models (ideal, Eagle, Heron) |
| `quantum/mitigation.py` | Readout error mitigation |
| `experiments/run_cross_comparison.py` | Classical ↔ quantum comparison runner |
| `experiments/data_bridge.py` | RLSTCcode → Q-RLSTC data conversion |

---

**Next:** [Noise & Hardware Simulation →](noise_and_hardware.md)
