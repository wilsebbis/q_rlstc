# RLSTC vs. Q-RLSTC: Technical Comparison

[← Back to README](../../README.md) · [Justifications](justifications.md) · **Comparison** · [Noise & Hardware →](noise_and_hardware.md)

---

A side-by-side analysis across 13 dimensions, covering architecture, design decisions, and the split between **Version A** (close comparison) and **Version B** (quantum-native).

## 1. Architecture Overview

| Dimension | RLSTC | Q-RLSTC |
|---|---|---|
| **Policy network** | Classical DQN (TF 1.x / Keras) | VQ-DQN (Qiskit parameterised circuit) |
| **Optimizer** | SGD (lr = 0.001) | SPSA (gradient-free, NISQ-suitable) |
| **Distance / clustering** | Incremental IED (custom) | Classical k-means + incremental OD proxy |
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
|---|---------|--------------------------|
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

## 3. Action Space

Both systems: **binary** — EXTEND (0) or CUT (1). Identical semantics.

## 4. Reward Functions

| Component | RLSTC | Q-RLSTC |
|---|---|---|
| **Main signal** | `ΔOD = last_od − current_od` (full IED) | `α · od_improvement` (lightweight proxy) |
| **Boundary quality** | None | `β · boundary_sharpness` (angle-based) |
| **Over-segmentation** | Implicit via `MIN_SEGMENT_LEN` | Explicit `−penalty` in reward |
| **Markov safety** | Depends on global cluster state | Uses only incremental quantities |

## 5. Quantum Circuit

| Aspect | Version A | Version B |
|---|---|---|
| **Qubits** | 5 | 8 |
| **Encoding** | Angle (RY) | Angle (RY) |
| **Variational layers** | 2 × (RY-RZ + CNOT chain) | 2 × (RY-RZ + CNOT chain) |
| **Trainable params** | 20 | 32 |
| **Entanglement** | 4 CNOTs (linear) | 7 CNOTs (linear) |
| **Data re-uploading** | Yes | Yes |
| **Readout** | `⟨Z₀⟩`, `⟨Z₁⟩` | `w₀⟨Z₀⟩ + w₁⟨Z₂Z₃⟩`, `w₂⟨Z₁⟩ + w₃⟨Z₄Z₅⟩` |

## 6. Optimizer

| | RLSTC | Q-RLSTC |
|---|---|---|
| **Method** | SGD with backprop | SPSA (gradient-free) |
| **Evals per step** | 1 (forward + backward) | 2 (forward only) |
| **Shot noise?** | N/A | Robust by design |
| **Gradient clipping** | No | Yes (max norm 1.0) |

## 7. Distance Computation

| | RLSTC | Q-RLSTC |
|---|---|---|
| **Primary metric** | Incremental IED | Lightweight OD proxy |
| **Per-step cost** | O(1) amortised | O(1) |
| **Full computation** | Every CUT action | Episode-end only (k-means) |
| **Available metrics** | IED, Fréchet, DTW | OD, Silhouette, F1 |

## 8. Data Structures

| | RLSTC | Q-RLSTC |
|---|---|---|
| **Point** | Plain class with `x, y, t` | `@dataclass` with `distance()`, `to_array()` |
| **Segment** | Class with distance methods | Implicit (index range) |
| **Trajectory** | `Traj(points, size, ts, te)` | `@dataclass` with `boundaries`, `labels` |
| **Replay buffer** | `deque(maxlen=2000)` in DQN class | Separate `ReplayBuffer(10000)` |
| **Cluster state** | Mutable dict `{id: [data]}` | `@dataclass ClusterState` |
| **Config** | Hardcoded constants | Nested `@dataclass` hierarchy |

## 9. Version A vs. Version B Summary

| Dimension | Version A (Close) | Version B (Unique) |
|---|---|---|
| **Goal** | Isolate quantum vs. classical approximator | Leverage larger Hilbert space |
| **Qubits** | 5 | 8 |
| **Features** | 5D (matches RLSTC) | 8D (3 quantum-native added) |
| **Readout** | Single-qubit Z | Multi-observable (Z + ZZ parity) |
| **Params** | 20 | 32 |
| **Config** | `version="A"` | `version="B"` |

## 10. Noise & Hardware

| | RLSTC | Q-RLSTC |
|---|---|---|
| **Noise simulation** | None | Full stack (ideal, simple, Eagle, Heron) |
| **Error mitigation** | None | Readout calibration matrix |
| **Backend** | CPU (TensorFlow) | Qiskit Aer (configurable) |

## 11. Training Pipeline

| Aspect | RLSTC | Q-RLSTC |
|---|---|---|
| **Loop** | Iterate points → EXTEND/CUT | Same |
| **Replay** | Internal to DQN (2,000) | Separate buffer (10,000) |
| **Target update** | Soft (τ = 0.05 every batch) | Hard copy every N episodes |
| **Double DQN** | No | Yes |
| **Anti-gaming** | `MIN_SEGMENT_LEN` | Same + explicit reward penalty |

## 12. Design Rationale

| Decision | Rationale |
|---|---|
| Only policy is quantum | Fixed I/O (5→2); distance needs O(1) updates |
| SPSA over parameter-shift | 2 evals vs. 40; scales to larger circuits |
| Angle encoding | 1 feature → 1 qubit; bounded via `arctan` |
| Data re-uploading | Expressivity without depth; proven technique |
| Version A exists | Scientific control: isolate the approximator |
| Version B exists | Explore whether more qubits + richer features helps |

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

### Q-RLSTC

| File | Purpose |
|---|---|
| `quantum/vqdqn_circuit.py` | Circuit: encoding, HEA, measurement |
| `rl/vqdqn_agent.py` | Agent: ε-greedy, Double DQN, target network |
| `rl/train.py` | Training loop + MDP environment |
| `rl/spsa.py` | SPSA optimizer |
| `rl/replay_buffer.py` | Experience replay |
| `config.py` | Configuration dataclasses |
| `data/features.py` | State feature extraction (A + B) |
| `clustering/metrics.py` | OD, silhouette, F1 |
| `clustering/classical_kmeans.py` | K-means evaluation |
| `quantum/backends.py` | Noise models (ideal, Eagle, Heron) |
| `quantum/mitigation.py` | Readout error mitigation |
| `data/synthetic.py` | Trajectory generation |

---

**Next:** [Noise & Hardware Simulation →](noise_and_hardware.md)
